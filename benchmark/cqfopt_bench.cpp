// cqfopt_bench: before/after driver for the CQF/BCQF optimization pass.
// Per-op tables at a realistic table (default 1<<20 slots for the wide
// configs) and fills 0.5/0.75/0.9 of each structure's reachable cap:
// insert (single random-order + bulk sorted), query hit/miss, update
// (counter bump on existing), delete (dec to zero), memory bytes/key.
// Comparators: std::unordered_map with the repo's WangHash. References are
// fp-space aware for the inexact/compact configs (mask + saturation), so
// every phase is count-verified after it runs. Seeded throughout.
#include "cqf.h"
#include "hash.h"
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <type_traits>

using namespace sketch;
using clk = std::chrono::high_resolution_clock;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

static double tsec(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// distinct pre-hashed stream (splitmix64 advance + WangHash finalize)
static std::vector<uint64_t> make_keys(uint64_t n, uint64_t seed) {
    std::vector<uint64_t> v(n);
    uint64_t st = seed;
    for(uint64_t i = 0; i < n; ++i) {
        st += 0x9E3779B97F4A7C15ULL;
        uint64_t z = st;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        v[i] = sketch::hash::WangHash{}(z ^ (z >> 31));
    }
    return v;
}

// family-tagged construction: bcqf/bcf take an explicit max_load; cqfbase_t
// takes a fuse policy flag; the map takes neither
template<typename F, bool FUSE>
struct maker { static F make(uint32_t q, double) { return F(q, FUSE); } };
template<size_t KB, size_t VB, class H, class L, bool FUSE>
struct maker<cqf::bcqf_t<KB,VB,H,L>, FUSE> {
    static cqf::bcqf_t<KB,VB,H,L> make(uint32_t q, double ml) { return cqf::bcqf_t<KB,VB,H,L>(q, ml); }
};
template<size_t FP, size_t VB, class H, bool FUSE>
struct maker<cqf::bcf_t<FP,VB,H>, FUSE> {
    static cqf::bcf_t<FP,VB,H> make(uint32_t q, double ml) { return cqf::bcf_t<FP,VB,H>(q, ml); }
};

// detect a native count_many (bcqf has one; older trees do not)
template<typename T, typename = void>
struct has_count_many: std::false_type {};
template<typename T>
struct has_count_many<T, std::void_t<decltype(std::declval<const T &>().template count_many<std::vector<uint64_t>::const_iterator, std::vector<uint64_t>::iterator>(std::declval<std::vector<uint64_t>::const_iterator>(), std::declval<std::vector<uint64_t>::iterator>(), std::declval<std::vector<uint64_t>::iterator>()))>>: std::true_type {};

template<typename F, bool FUSE = true>
struct SketchAdapter {
    F f;
    SketchAdapter(uint32_t q, double ml): f(maker<F,FUSE>::make(q, ml)) {}
    static constexpr uint64_t kmask = F::key_bits >= 64 ? ~uint64_t(0): ((uint64_t(1) << F::key_bits) - 1);
    static constexpr uint64_t sat = F::val_bits >= 64 ? ~uint64_t(0): ((uint64_t(1) << F::val_bits) - 1);
    static uint64_t slots(uint32_t q) { return uint64_t(1) << q; }
    static uint64_t cap(uint32_t q, double) { return (uint64_t(1) << q) - ((uint64_t(1) << q) / 10); } // 0.9 ceiling
    static uint64_t maskk(uint64_t k) { return k & kmask; }
    void add(uint64_t k) { f.add(maskk(k)); }
    uint64_t count(uint64_t k) const { return f.count(maskk(k)); }
    void dec(uint64_t k) { f.dec(maskk(k)); }
    uint64_t bytes() const { return f.est_memory_usage().second; }
    uint64_t used() const { return f.size(); }
    uint32_t qbits() const { return f.q(); }
    void check() const { f.structure_check(); }
    template<typename It> void bulk(It a, It b) { f.insert_sorted_counts(a, b); }
    template<typename It, typename Ot> void batch_count(It a, It b, Ot o) const { batch_count_impl(a, b, o, std::integral_constant<bool, has_count_many<F>::value>{}); }
    template<typename It, typename Ot> void batch_count_impl(It a, It b, Ot o, std::true_type) const { f.count_many(a, b, o); }
    template<typename It, typename Ot> void batch_count_impl(It a, It b, Ot o, std::false_type) const { for(; a != b; ++a, ++o) *o = count(*a); }
    static void ref_add(std::unordered_map<uint64_t, uint64_t> &ref, uint64_t k) {
        const uint64_t kk = maskk(k);
        const uint64_t nv = ref.count(kk) ? ref[kk] + 1: 1;
        ref[kk] = nv > sat ? sat: nv;
    }
};

template<typename V>
struct UMapAdapter {
    std::unordered_map<uint64_t, V, sketch::hash::WangHash> f;
    UMapAdapter(uint32_t, double) {}
    static uint64_t slots(uint32_t q) { return uint64_t(1) << q; }
    static uint64_t cap(uint32_t q, double) { return (uint64_t(1) << q) - ((uint64_t(1) << q) / 10); } // mirror the CQF cap
    static uint64_t maskk(uint64_t k) { return k; }
    void add(uint64_t k) { ++f[k]; }
    uint64_t count(uint64_t k) const { auto it = f.find(k); return it == f.end() ? 0: uint64_t(it->second); }
    void dec(uint64_t k) {
        auto it = f.find(k);
        if(it == f.end()) return;
        if(it->second > 1) --it->second; else f.erase(it);
    }
    uint64_t bytes() const { // documented libstdc++ node + bucket footprint
        const size_t node = (24 + sizeof(V) + 7) / 8 * 8;
        return f.size() * (node + 8);
    }
    uint64_t used() const { return f.size(); }
    uint32_t qbits() const { return 0; }
    void check() const {}
    template<typename It> void bulk(It a, It b) { for(; a != b; ++a) add(a->first); }
    template<typename It, typename Ot> void batch_count(It a, It b, Ot o) const {
        for(; a != b; ++a, ++o) *o = count(*a);
    }
    static void ref_add(std::unordered_map<uint64_t, uint64_t> &ref, uint64_t k) {
        const uint64_t nv = ref.count(k) ? ref[k] + 1: 1;
        ref[k] = nv > uint64_t(V(~0)) ? uint64_t(V(~0)): nv;
    }
};

static void row(const char *tag, uint32_t q, double fill, const char *op, double n, double dt) {
    std::printf("| %s | %u | %.2f | %s | %.2f | %.1f |\n", tag, q, fill, op, n / dt / 1e6, dt / n * 1e9);
}

template<typename Ad>
static void verify_sampled(const Ad &ad, const std::unordered_map<uint64_t, uint64_t> &ref,
                           const std::vector<uint64_t> &keys, uint64_t begin, uint64_t n, uint64_t want_add, const char *tag) {
    const uint64_t step = n > 8192 ? n / 8192: 1;
    for(uint64_t i = 0; i < n; i += step) {
        const uint64_t k = keys[begin + i];
        const uint64_t want = (ref.count(Ad::maskk(k)) ? ref.at(Ad::maskk(k)): 0) + want_add;
        const uint64_t got = ad.count(k);
        if(got != want) { CHECK(false, "%s: count mismatch (got %llu want %llu)", tag, (unsigned long long)got, (unsigned long long)want); return; }
    }
}

// verify exactly the keys a strided op phase touched (i in [0,n): keys[(i*7919)%target]);
// want_add must be the PER-KEY multiplicity of touches when the masked space
// collides (fp-compact configs: several raw keys share one slot, each touch
// bumps it), 1 for collision-free configs
template<typename Ad>
static void verify_strided(const Ad &ad, const std::unordered_map<uint64_t, uint64_t> &ref,
                           const std::vector<uint64_t> &keys, uint64_t target, uint64_t n, uint64_t want_add, const char *tag) {
    // multiplicity of each masked key among the touched set (1 unless collisions)
    std::unordered_map<uint64_t, uint64_t> mult;
    if(want_add) {
        for(uint64_t i = 0; i < n; ++i) ++mult[Ad::maskk(keys[(i * 7919) % target])];
    }
    const uint64_t step = n > 8192 ? n / 8192: 1;
    for(uint64_t i = 0; i < n; i += step) {
        const uint64_t k = keys[(i * 7919) % target];
        const uint64_t want = (ref.count(Ad::maskk(k)) ? ref.at(Ad::maskk(k)): 0)
                              + (want_add ? mult[Ad::maskk(k)]: 0);
        const uint64_t got = ad.count(k);
        if(got != want) { CHECK(false, "%s: count mismatch (got %llu want %llu)", tag, (unsigned long long)got, (unsigned long long)want); return; }
    }
}

template<typename Ad>
static void run_line(const char *tag, uint32_t q, double maxload,
                     const std::vector<uint64_t> &keys, bool quick) {
    const uint64_t cap = Ad::cap(q, maxload);
    const uint64_t nq = quick ? 50000: 200000;
    const uint64_t M = quick ? 20000: 50000; // fresh keys per insert phase
    const uint64_t nhave = keys.size();
    // The built stream STRIDE-SAMPLES the key stream (stride 2): the sorted
    // built set must stay uniform over the whole key space -- taking the
    // smallest values of a dense uniform stream would concentrate the
    // quotient range (mega-clusters for the CQF) and would let high-bit
    // miss-key flips survive masking into the built set.
    const uint64_t miss_lo = 2 * (cap + 16), fresh1 = 2 * (cap + 32 + nq), fresh2 = fresh1 + 2 * M;
    std::unordered_map<uint64_t, uint64_t> ref;
    ref.reserve(cap);
    std::vector<uint64_t> miss;
    std::vector<std::pair<uint64_t, uint64_t>> sorted; // the canonical built stream
    sorted.reserve(cap);
    for(uint64_t i = 0; i < cap; ++i) sorted.emplace_back(Ad::maskk(keys[2 * i]), 1);
    std::sort(sorted.begin(), sorted.end());
    std::fprintf(stderr, "== %s q=%u (cap %llu slots) ==\n", tag, q, (unsigned long long)cap);
    Ad ad(q, maxload);
    for(const double fill: {0.5, 0.75, 0.9}) {
        const uint64_t target = uint64_t(fill * cap);
        const uint64_t Mf = std::min<uint64_t>(M, target / 4); // fresh inserts per insert phase
        // canonical built keys: sorted[j].first for j < target (insert order is
        // the sorted order; ALL phases sample this array, never the raw stream)
        const uint64_t *built = &sorted[0].first; // pairs are 16-byte; use stride below
        (void)built;
        std::vector<uint64_t> bk;
        bk.reserve(target);
        for(uint64_t i = 0; i < target; ++i) bk.push_back(sorted[i].first);
        // build via the sorted bulk path (excluded from timing), reference in lockstep
        ad = Ad(q, maxload);
        ref.clear();
        const auto tb0 = clk::now();
        ad.bulk(sorted.begin(), sorted.begin() + target);
        const double tbuild = tsec(tb0, clk::now());
        for(uint64_t i = 0; i < target; ++i) Ad::ref_add(ref, bk[i]);
        CHECK(ad.used() > 0, "%s: empty after build", tag);
        const double bytes_per_key = double(ad.bytes()) / double(ad.used());
        std::fprintf(stderr, "%s fill=%.2f: built %llu keys (%.2f M/s bulk-build), mem %.2f B/key\n",
                     tag, fill, (unsigned long long)ad.used(), target / tbuild / 1e6, bytes_per_key);
        miss.clear();
        for(uint64_t i = 0; i < nq; ++i) miss.push_back(keys[miss_lo + 2 * i]);
        volatile uint64_t sink = 0;
        double thit = 0, tmiss = 0;
        for(int rep = 0; rep < 2; ++rep) {
            auto t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) sink += ad.count(bk[(i * 7919) % target]);
            thit += tsec(t0, clk::now());
            t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) sink += ad.count(miss[i]);
            tmiss += tsec(t0, clk::now());
        }
        row(tag, q, fill, "query-hit", nq, thit / 2);
        row(tag, q, fill, "query-miss", nq, tmiss / 2);
        // batched queries (in-binary A/B: count_many API vs the one-by-one loop above)
        {
            std::vector<uint64_t> hitq(nq), out(nq);
            for(uint64_t i = 0; i < nq; ++i) hitq[i] = bk[(i * 7919) % target];
            double tb = 0;
            for(int rep = 0; rep < 2; ++rep) {
                const auto t0 = clk::now();
                ad.batch_count(hitq.begin(), hitq.end(), out.begin());
                tb += tsec(t0, clk::now());
            }
            for(uint64_t i = 0; i < nq; i += nq / 1024 + 1)
                CHECK(out[i] == ad.count(hitq[i]), "%s: batch hit mismatch", tag);
            row(tag, q, fill, "query-hit-batch", nq, tb / 2);
            double tbm = 0;
            for(int rep = 0; rep < 2; ++rep) {
                const auto t0 = clk::now();
                ad.batch_count(miss.begin(), miss.end(), out.begin());
                tbm += tsec(t0, clk::now());
            }
            for(uint64_t i = 0; i < nq; i += nq / 1024 + 1)
                CHECK(out[i] == 0, "%s: batch miss nonzero", tag);
            row(tag, q, fill, "query-miss-batch", nq, tbm / 2);
        }
        {   // realized false-positive rate over the absent probes (0 for exact modes)
            uint64_t fp = 0;
            for(uint64_t i = 0; i < nq; ++i) fp += ad.count(miss[i]) != 0;
            std::fprintf(stderr, "%s fill=%.2f: realized FP on %llu absent probes: %.6f\n", tag, fill, (unsigned long long)nq, double(fp) / nq);
        }
        {
            const auto t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) ad.add(bk[(i * 7919) % target]);
            row(tag, q, fill, "update-inc", nq, tsec(t0, clk::now()));
        }
        verify_strided<Ad>(ad, ref, bk, target, nq, 1, "update-inc");
        {
            const auto t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) ad.dec(bk[(i * 7919) % target]);
            row(tag, q, fill, "delete-dec", nq, tsec(t0, clk::now()));
        }
        verify_strided<Ad>(ad, ref, bk, target, nq, 0, "delete-dec");
        ad.check();
        // insert fresh singles: rebuild to target-M, time the last M
        {
            ad = Ad(q, maxload);
            ad.bulk(sorted.begin(), sorted.begin() + (target - Mf));
            std::unordered_map<uint64_t, uint64_t> r2;
            for(uint64_t i = 0; i < target - Mf; ++i) Ad::ref_add(r2, bk[i]);
            const auto t0 = clk::now();
            for(uint64_t i = 0; i < Mf; ++i) ad.add(keys[fresh1 + 2 * i]);
            row(tag, q, fill, "insert-single", Mf, tsec(t0, clk::now()));
            for(uint64_t i = 0; i < Mf; ++i) Ad::ref_add(r2, keys[fresh1 + 2 * i]);
            verify_sampled<Ad>(ad, r2, bk, 0, target - Mf, 0, "insert-single(old)");
            for(uint64_t i = 0; i < Mf; i += Mf / 2048 + 1) {
                const uint64_t k = keys[fresh1 + 2 * i];
                CHECK(ad.count(k) == (r2.count(Ad::maskk(k)) ? r2.at(Ad::maskk(k)): 0), "%s: insert-single(new) count mismatch", tag);
            }
            if(ad.qbits() != 0 && ad.qbits() != q) std::fprintf(stderr, "%s fill=%.2f: NOTE insert-single triggered a grow (q=%u)\n", tag, fill, ad.qbits());
            ad.check();
        }
        // bulk sorted: rebuild to target-M, time a sorted M-key append
        {
            ad = Ad(q, maxload);
            ad.bulk(sorted.begin(), sorted.begin() + (target - Mf));
            std::vector<std::pair<uint64_t, uint64_t>> b2;
            b2.reserve(Mf);
            for(uint64_t i = 0; i < Mf; ++i) b2.emplace_back(Ad::maskk(keys[fresh2 + 2 * i]), 1);
            std::sort(b2.begin(), b2.end());
            const auto t0 = clk::now();
            ad.bulk(b2.begin(), b2.end());
            row(tag, q, fill, "insert-bulk-sorted", Mf, tsec(t0, clk::now()));
            std::unordered_map<uint64_t, uint64_t> r2;
            for(uint64_t i = 0; i < target - Mf; ++i) Ad::ref_add(r2, bk[i]);
            for(uint64_t i = 0; i < Mf; ++i) Ad::ref_add(r2, keys[fresh2 + 2 * i]);
            for(uint64_t i = 0; i < Mf; i += Mf / 2048 + 1) {
                const uint64_t k = keys[fresh2 + 2 * i];
                CHECK(ad.count(k) == (r2.count(Ad::maskk(k)) ? r2.at(Ad::maskk(k)): 0), "%s: insert-bulk(new) count mismatch", tag);
            }
            if(ad.qbits() != 0 && ad.qbits() != q) std::fprintf(stderr, "%s fill=%.2f: NOTE insert-bulk triggered a grow (q=%u)\n", tag, fill, ad.qbits());
            ad.check();
        }
        std::fprintf(stderr, "%s fill=%.2f done\n", tag, fill);
    }
    (void)ad;
}

int main(int argc, char **argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "quick";
    std::fprintf(stderr, "cqfopt_bench (seed 0xC0FFEE, %s mode)\n", quick ? "quick": "full");
    const uint64_t nhave = uint64_t(1) << 22; // 4M distinct keys in reserve (stride-2 sampling)
    const std::vector<uint64_t> keys = make_keys(nhave, 0xC0FFEEULL);
    std::puts("| config | q | fill | op | Mops/s | ns/op |");
    std::puts("|---|---|---|---|---|---|");
    run_line<SketchAdapter<cqf::bcqf_t<48,16>>>("bcqf48_16", 20, 0.95, keys, quick);
    run_line<SketchAdapter<cqf::bcqf_t<16,8>>>("bcqf16_8", 14, 0.95, keys, quick);
    run_line<SketchAdapter<cqf::cqf64_16>>("cqf64_16", 20, -1., keys, quick);
    run_line<SketchAdapter<cqf::cqf64_64>>("cqf64_64", 20, -1., keys, quick);
    run_line<SketchAdapter<cqf::cqf32_8>>("cqf32_8", 20, -1., keys, quick);
    // packed regime (fuse=false) full vs truncated remainder: the Phase-2
    // variable-effective-remainder space win lives in the packed arrays
    run_line<SketchAdapter<cqf::cqfbase_t<32,8>, false>>("cqf32_8p", 20, -1., keys, quick);
    run_line<SketchAdapter<cqf::cqfbase_t<32,8,uint64_t,sketch::hash::WangHash,8>, false>>("cqf32_8p_t8", 20, -1., keys, quick);
    // position-encoded fingerprint filters (Phase 2 mode): FP bits stored,
    // the block position disambiguates negatives
    run_line<SketchAdapter<cqf::bcf_t<12,8>>>("bcf12_8", 16, 0.95, keys, quick); // FP=12: q <= FP + BLK_SHIFT
    run_line<SketchAdapter<cqf::bcf_t<16,8>>>("bcf16_8", 20, 0.95, keys, quick);
    run_line<UMapAdapter<uint16_t>>("umap_u16", 20, -1., keys, quick);
    run_line<UMapAdapter<uint64_t>>("umap_u64", 20, -1., keys, quick);
    if(failures) { std::fprintf(stderr, "%d FAILURES\n", failures); return 1; }
    std::fprintf(stderr, "cqfopt_bench done\n");
    return 0;
}
