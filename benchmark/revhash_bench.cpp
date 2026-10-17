// revhash_bench: speed/quality tradeoffs for invertible 64-bit hashes.
// Built by the repo Makefile pattern rule (benchmark/%.cpp -> ./revhash_bench).
// Usage: ./revhash_bench [quick]
#include "sketch/hash.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

using namespace sketch::hash;
using clk = std::chrono::steady_clock;

static double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }
static uint64_t sink_val;
static inline void sink(uint64_t v) { sink_val ^= v; }

// local 1-multiply experimental candidate (NOT shipped in hash.h: fails avalanche)
struct Xor1Mul: sketch::hash::CEIFused<RShiftXor<33>, CEIMul<0xff51afd7ed558ccd>, RShiftXor<31>> {};

// ---------------- speed ----------------
template<typename F>
double latency_ns(const F &f, uint64_t ops) {
    uint64_t v = 0x9e3779b97f4a7c15ull;
    double t0 = now_s();
    for(uint64_t i = 0; i < ops; ++i) v = f(v);
    double t1 = now_s();
    sink(v);
    return (t1 - t0) / ops * 1e9;
}
template<typename F>
double throughput_mhps(const F &f, const std::vector<uint64_t> &in, uint32_t reps) {
    uint64_t acc = 0;
    double t0 = now_s();
    for(uint32_t r = 0; r < reps; ++r)
        for(uint64_t v: in) acc ^= f(v);
    double t1 = now_s();
    sink(acc);
    return double(in.size()) * reps / (t1 - t0) / 1e6;
}
#ifndef VEC_DISABLED__
using SpaceL = vec::SIMDTypes<uint64_t>;
using VTypeL = typename SpaceL::VType;
using RawTypeL = typename SpaceL::Type;
static constexpr size_t LANES = SpaceL::COUNT;
template<typename F>
double simd_throughput_mhps(const F &f, const std::vector<uint64_t> &in, uint32_t reps) {
    uint64_t acc = 0;
    double t0 = now_s();
    for(uint32_t r = 0; r < reps; ++r) {
        size_t i = 0;
        for(; i + LANES <= in.size(); i += LANES) {
            VTypeL v = SpaceL::load(reinterpret_cast<const RawTypeL *>(&in[i]));
            VTypeL hv = f(v);
            RawTypeL raw = hv.simd_;
            uint64_t tmp[LANES];
            std::memcpy(tmp, &raw, sizeof(tmp));
            for(size_t j = 0; j < LANES; ++j) acc ^= tmp[j];
        }
        for(; i < in.size(); ++i) acc ^= 0; // tail skipped (inputs are a multiple of LANES here)
    }
    double t1 = now_s();
    sink(acc);
    return double(in.size()) * reps / (t1 - t0) / 1e6;
}
#endif

// ---------------- quality ----------------
struct AvalancheRes { double maxdev; double worst_chi2; };
template<typename F>
AvalancheRes avalanche(const F &f, uint32_t trials) {
    std::mt19937_64 gen(0xA5A5A5 + trials);
    static uint64_t counts[64][64];
    for(auto &row: counts) for(auto &c: row) c = 0;
    for(uint32_t t = 0; t < trials; ++t) {
        uint64_t base = gen();
        uint64_t hb = f(base);
        for(unsigned i = 0; i < 64; ++i) {
            uint64_t d = f(base ^ (uint64_t(1) << i)) ^ hb;
            for(unsigned j = 0; j < 64; ++j) counts[i][j] += (d >> j) & 1;
        }
    }
    AvalancheRes r{0, 0};
    for(unsigned i = 0; i < 64; ++i) for(unsigned j = 0; j < 64; ++j) {
        double dev = std::fabs(double(counts[i][j]) / trials - 0.5);
        if(dev > r.maxdev) r.maxdev = dev;
        double z = double(counts[i][j]) - 0.5 * trials;
        double chi2 = 4. * z * z / trials;
        if(chi2 > r.worst_chi2) r.worst_chi2 = chi2;
    }
    return r;
}
template<typename F>
double topq_chi2(const F &f, unsigned q, const std::vector<uint64_t> &in, std::vector<uint32_t> &hist) {
    const size_t cells = size_t(1) << q;
    hist.assign(cells, 0);
    for(uint64_t x: in) ++hist[f(x) >> (64 - q)];
    double exp = double(in.size()) / cells, chi2 = 0;
    for(uint32_t c: hist) { double d = double(c) - exp; chi2 += d * d / exp; }
    return chi2;
}
struct RemRes { double worst1pct_chi2; double max_chi2; double mean_chi2; };
template<typename F>
RemRes remainder_independence(const F &f, unsigned q, unsigned rem_bits, const std::vector<uint64_t> &in) {
    const size_t buckets = size_t(1) << q, cells = size_t(1) << rem_bits;
    std::vector<uint32_t> cnt(buckets * cells, 0);
    for(uint64_t x: in) {
        uint64_t hv = f(x);
        ++cnt[size_t(hv >> (64 - q)) * cells + (hv & (cells - 1))];
    }
    std::vector<double> chi2s(buckets, 0.);
    const double exp = double(in.size()) / buckets / cells;
    for(size_t b = 0; b < buckets; ++b) {
        double chi2 = 0;
        for(size_t c = 0; c < cells; ++c) { double d = double(cnt[b * cells + c]) - exp; chi2 += d * d / exp; }
        chi2s[b] = chi2;
    }
    std::sort(chi2s.begin(), chi2s.end());
    const size_t n1 = buckets / 100 ? buckets / 100 : 1;
    double s = 0; for(size_t i = chi2s.size() - n1; i < chi2s.size(); ++i) s += chi2s[i];
    double tot = 0; for(double c: chi2s) tot += c;
    return {s / n1, chi2s.back(), tot / buckets};
}
template<typename F>
bool bijection_spotcheck(const F &f, const std::vector<uint64_t> &in) {
    std::vector<uint64_t> hv(in.size());
    for(size_t i = 0; i < in.size(); ++i) hv[i] = f(in[i]);
    std::sort(hv.begin(), hv.end());
    return std::adjacent_find(hv.begin(), hv.end()) == hv.end();
}

int main(int argc, char **argv) {
    const bool quick = argc > 1 && std::strcmp(argv[1], "quick") == 0;
    const uint64_t NINPUT = 1ull << 23; // 8M
    std::vector<uint64_t> inputs(NINPUT);
    {
        std::mt19937_64 gen(20261017);
        for(auto &x: inputs) x = gen();
    }
    struct Cand {
        std::string name;
        std::function<uint64_t(uint64_t)> fn;
#ifndef VEC_DISABLED__
        std::function<VTypeL(VTypeL)> simd_fn;
#endif
    };
    std::vector<Cand> cands;
    auto add = [&](const std::string &n, std::function<uint64_t(uint64_t)> fn
#ifndef VEC_DISABLED__
                   , std::function<VTypeL(VTypeL)> simd_fn = {}
#endif
                   ) {
        cands.push_back({n, std::move(fn)
#ifndef VEC_DISABLED__
                         , std::move(simd_fn)
#endif
                        });
    };
    add("WangHash (baseline)", [](uint64_t x){ return WangHash()(x); }
#ifndef VEC_DISABLED__
        , [](VTypeL v){ return VTypeL{WangHash()(v.simd_)}; }
#endif
        );
    add("SplitMix64Hash (2 mul)", [](uint64_t x){ return SplitMix64Hash()(x); });
    add("Xor1Mul<33,C1,31> local (1 mul)", [](uint64_t x){ return Xor1Mul()(x); });
    add("CEHasher (xor,mul,xor)", [](uint64_t x){ return CEHasher()(x); });
    add("XorMultiply (xor,mul)", [h = XorMultiply()](uint64_t x){ return h(x); }
#ifndef VEC_DISABLED__
        , [h = XorMultiply()](VTypeL v) mutable { return h(v); }
#endif
        );
    add("MultiplyAdd (mul,add)", [h = MultiplyAdd()](uint64_t x){ return h(x); }
#ifndef VEC_DISABLED__
        , [h = MultiplyAdd()](VTypeL v) mutable { return h(v); }
#endif
        );
    add("MultiplyAddXor (a,mul,x)", [h = MultiplyAddXor()](uint64_t x){ return h(x); }
#ifndef VEC_DISABLED__
        , [h = MultiplyAddXor()](VTypeL v) mutable { return h(v); }
#endif
        );
    add("MultiplyAddXoRot<23>", [h = MultiplyAddXoRot<23>()](uint64_t x){ return h(x); });
    add("MultiplyAddXoRot<29>", [h = MultiplyAddXoRot<29>()](uint64_t x){ return h(x); });
    add("XorMultiplyNVec n=2", [h = XorMultiplyNVec(2)](uint64_t x){ return h(x); });
    add("XorMultiplyNVec n=3", [h = XorMultiplyNVec(3)](uint64_t x){ return h(x); });
    add("MultiplyAddNVec n=2", [h = MultiplyAddNVec(2)](uint64_t x){ return h(x); });
    add("MultiplyAddNVec n=3", [h = MultiplyAddNVec(3)](uint64_t x){ return h(x); });
    add("MultiplyAddXorNVec n=2", [h = MultiplyAddXorNVec(2)](uint64_t x){ return h(x); });
    add("MultiplyAddXorNVec n=3", [h = MultiplyAddXorNVec(3)](uint64_t x){ return h(x); });
    add("MultiplyAddXoRotNVec<23> n=2", [h = MultiplyAddXoRotNVec<23>(2)](uint64_t x){ return h(x); });
    add("TriangularReversibleHash", [](uint64_t x){ return TriangularReversibleHash<>()(x); });
    add("MurFinHash (reference)", [](uint64_t x){ return MurFinHash()(x); });

    const uint32_t speed_reps = quick ? 3 : 20;
    const uint64_t lat_ops = quick ? (1ull << 24) : (1ull << 26);
    const uint32_t av_trials = quick ? 4096 : 16384;
    const size_t qn = quick ? (1ull << 22) : size_t(NINPUT);
    std::vector<uint64_t> qin(inputs.begin(), inputs.begin() + qn);

    std::fprintf(stderr, "revhash_bench: %zu candidates, %llu inputs, quick=%d, LANES=%zu\n",
                 cands.size(), (unsigned long long)NINPUT, int(quick),
#ifndef VEC_DISABLED__
                 LANES
#else
                 0
#endif
                 );
    std::fprintf(stderr, "lat=dependent-chain ns/op (median of 3 x %llu ops); thr=M hashes/s over 8M inputs (median of 3, x%u reps); simd=SIMD M/s where provided\n", (unsigned long long)lat_ops, speed_reps);

    std::printf("%-34s %8s %9s %9s | %7s %8s | %10s | %8s %8s %8s | %s\n",
                "candidate", "lat_ns", "thr_Mh/s", "simd_Mh/s", "maxdev", "chi2max", "top22_chi2", "rem_w1pc", "rem_mean", "rem_max", "bijective");
    std::fflush(stdout);

    std::vector<uint32_t> hist;
    for(auto &c: cands) {
        double lat[3], thr[3], simd[3] = {0, 0, 0};
        for(int r = 0; r < 3; ++r) {
            lat[r] = latency_ns(c.fn, lat_ops);
            thr[r] = throughput_mhps(c.fn, inputs, speed_reps);
        }
        std::sort(lat, lat + 3); std::sort(thr, thr + 3);
#ifndef VEC_DISABLED__
        if(c.simd_fn) {
            for(int r = 0; r < 3; ++r) simd[r] = simd_throughput_mhps(c.simd_fn, inputs, speed_reps);
            std::sort(simd, simd + 3);
        }
#endif
        AvalancheRes av = avalanche(c.fn, av_trials);
        double chi2 = topq_chi2(c.fn, 22, qin, hist);
        RemRes rem = remainder_independence(c.fn, 18, 6, qin);
        const size_t BJ = 1ull << 20;
        bool bij = qn >= BJ ? bijection_spotcheck(c.fn, std::vector<uint64_t>(qin.begin(), qin.begin() + BJ)) : bijection_spotcheck(c.fn, qin);
        std::printf("%-34s %8.3f %9.1f %9.1f | %7.4f %8.1f | %10.0f | %8.1f %8.1f %8.1f | %s\n",
                    c.name.c_str(), lat[1], thr[1], simd[1], av.maxdev, av.worst_chi2, chi2,
                    rem.worst1pct_chi2, rem.mean_chi2, rem.max_chi2, bij ? "ok" : "COLLISION");
        std::fflush(stdout);
    }
    std::fprintf(stderr, "reference: 3/sqrt(trials)=%.4f at trials=%u; top22 chi2 df=2^22-1=%.4g (std %.3g); remainder per-bucket chi2 df=63 (expect ~63); bijection on 1M random keys.\n",
                 3. / std::sqrt(double(av_trials)), av_trials, double((1ull << 22) - 1), std::sqrt(2. * double((1ull << 22) - 1)));
    std::fprintf(stderr, "note: WangHash itself measures maxdev ~0.10-0.11 (structural, rows 9/63) at these trials -- 'comparable to WangHash' is the quality bar.\n");
    std::fprintf(stderr, "sink: %llu\n", (unsigned long long)sink_val);
    return 0;
}
