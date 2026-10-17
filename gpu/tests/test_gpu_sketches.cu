// Bit-identity correctness tests for the sketch GPU layer.
// For each sketch type: build the reference on CPU (the regular headers,
// compiled by nvcc's host pass), run the GPU kernel on the same pre-hashed
// stream, and require byte-identical sketch state (or exact integer equality
// for fmh kept-set summaries; kept hashes are compared sorted, since GPU
// compaction order is nondeterministic).
//
// Exit status: number of failed checks (0 = all pass).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hash.h"
#include "hll.h"
#include "bf.h"
#include "ccm.h"
#include "update.h"
#include "bbmh.h"
#include "fracminhash.h"

#include "sketch/gpu/hash.cuh"
#include "sketch/gpu/hll.cuh"
#include "sketch/gpu/bloom.cuh"
#include "sketch/gpu/count.cuh"
#include "sketch/gpu/minhash.cuh"

namespace {
int g_pass = 0, g_fail = 0;
void check(bool cond, const char *name) {
    if(cond) { ++g_pass; std::fprintf(stderr, "[PASS] %s\n", name); }
    else     { ++g_fail; std::fprintf(stderr, "[FAIL] %s\n", name); }
}
std::vector<std::uint64_t> mk_stream(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 mt(seed);
    std::vector<std::uint64_t> ret(n);
    for(auto &v: ret) v = mt();
    return ret;
}
// Stream with edge-heavy low-entropy values (exercises lzt == q+1 and the
// all-threads-one-register contention path).
std::vector<std::uint64_t> mk_edgy_stream(std::size_t n) {
    std::mt19937_64 mt(42);
    std::vector<std::uint64_t> ret(n);
    for(std::size_t i = 0; i < n; ++i) {
        switch(i & 7) {
            case 0: ret[i] = i; break;                                  // tiny values -> lzt saturates
            case 1: ret[i] = std::uint64_t(1) << 63; break;             // single top bit
            case 2: ret[i] = ~std::uint64_t(0); break;                  // all ones
            case 3: ret[i] = std::uint64_t(1); break;
            case 4: ret[i] = ~std::uint64_t(0) >> 20; break;
            default: ret[i] = mt(); break;
        }
    }
    return ret;
}
std::size_t first_diff_u8(const std::uint8_t *a, const std::uint8_t *b, std::size_t n) {
    for(std::size_t i = 0; i < n; ++i) if(a[i] != b[i]) return i;
    return n;
}
std::size_t first_diff_u64(const std::uint64_t *a, const std::uint64_t *b, std::size_t n) {
    for(std::size_t i = 0; i < n; ++i) if(a[i] != b[i]) return i;
    return n;
}
} // anonymous namespace

static void test_hash() {
    const std::size_t n = std::size_t(1) << 20;
    std::vector<std::uint64_t> in = mk_edgy_stream(n), out(n, 0);
    sketch::hash::SplitMix64Hash sm;
    for(std::size_t i = 0; i < 256; ++i) in[i] = i; // small integers too
    std::uint64_t *d_in = skgpu::memcpy_h2d(in.data(), in.size());
    std::uint64_t *d_out = static_cast<std::uint64_t *>(skgpu::malloc_dev(n * sizeof(std::uint64_t)));
    skgpu::hash(d_in, d_out, n);
    skgpu::memcpy_d2h(out.data(), d_out, n);
    skgpu::free_dev(d_in); skgpu::free_dev(d_out);
    std::size_t bad = n;
    for(std::size_t i = 0; i < n; ++i) if(out[i] != sm(in[i])) { bad = i; break; }
    check(bad == n, "splitmix64_hash bit-identical to CPU SplitMix64Hash");
    if(bad != n) std::fprintf(stderr, "  first mismatch at %zu: in=%016llx gpu=%016llx cpu=%016llx\n",
                              bad, (unsigned long long)in[bad], (unsigned long long)out[bad], (unsigned long long)sm(in[bad]));
}

static void test_hll_update() {
    const std::uint64_t seeds[] {137, 42};
    for(const std::uint32_t np: {4u, 10u, 16u, 20u}) {
        for(const std::uint64_t seed: seeds) {
            const std::size_t n = std::size_t(1) << 20;
            const std::size_t m = std::size_t(1) << np;
            std::vector<std::uint64_t> in = seed == 137 ? mk_edgy_stream(n): mk_stream(n, seed);
            sketch::hll::hll_t ref(np);
            for(const auto hv: in) ref.add(hv);
            std::uint64_t *d_in = skgpu::memcpy_h2d(in.data(), in.size());
            std::uint8_t *d_core = static_cast<std::uint8_t *>(skgpu::malloc_dev(m));
            SKGPU_CK(cudaMemset(d_core, 0, m));
            skgpu::hll_update(d_in, n, np, d_core);
            SKGPU_CK(cudaDeviceSynchronize());
            std::vector<std::uint8_t> gpu(m);
            skgpu::memcpy_d2h(gpu.data(), d_core, m);
            skgpu::free_dev(d_in); skgpu::free_dev(d_core);
            const std::size_t bad = first_diff_u8(gpu.data(), ref.core().data(), m);
            char buf[128];
            std::snprintf(buf, sizeof buf, "hll_update np=%u seed=%llu bit-identical (n=%zu, m=%zu)",
                          np, (unsigned long long)seed, n, m);
            check(bad == m, buf);
            if(bad != m) std::fprintf(stderr, "  first diff at register %zu: gpu=%u cpu=%u\n", bad, unsigned(gpu[bad]), unsigned(ref.core()[bad]));
        }
    }
}

static void test_hll_update_hashed() {
    const std::size_t n = std::size_t(1) << 20;
    std::vector<std::uint64_t> in = mk_stream(n, 137);
    for(const std::uint32_t np: {10u, 16u}) {
        const std::size_t m = std::size_t(1) << np;
        sketch::hll::hllbase_t<sketch::hash::SplitMix64Hash> ref(np);
        for(const auto x: in) ref.addh(x);
        std::uint64_t *d_in = skgpu::memcpy_h2d(in.data(), in.size());
        std::uint8_t *d_core = static_cast<std::uint8_t *>(skgpu::malloc_dev(m));
        SKGPU_CK(cudaMemset(d_core, 0, m));
        skgpu::hll_update_hashed(d_in, n, np, d_core);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> gpu(m);
        skgpu::memcpy_d2h(gpu.data(), d_core, m);
        skgpu::free_dev(d_in); skgpu::free_dev(d_core);
        char buf[128];
        std::snprintf(buf, sizeof buf, "hll_update_hashed np=%u bit-identical", np);
        check(first_diff_u8(gpu.data(), ref.core().data(), m) == m, buf);
    }
}

static void test_hll_merge() {
    const std::size_t n = std::size_t(1) << 18;
    for(const std::uint32_t np: {10u, 16u}) {
        const std::size_t m = std::size_t(1) << np;
        std::vector<sketch::hll::hll_t> refs;
        std::vector<std::uint8_t> all(3 * m);
        for(std::size_t s = 0; s < 3; ++s) {
            sketch::hll::hll_t ref(np);
            for(const auto hv: mk_stream(n, 137 + s)) ref.add(hv);
            std::memcpy(all.data() + s * m, ref.core().data(), m);
            refs.emplace_back(std::move(ref));
        }
        sketch::hll::hll_t merged(refs[0]);
        merged += refs[1];
        merged += refs[2];
        std::uint8_t *d_all = skgpu::memcpy_h2d(all.data(), all.size());
        std::uint8_t *d_out = static_cast<std::uint8_t *>(skgpu::malloc_dev(m));
        skgpu::hll_merge(d_out, d_all, 3, m);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> gpu(m);
        skgpu::memcpy_d2h(gpu.data(), d_out, m);
        char buf[128];
        std::snprintf(buf, sizeof buf, "hll_merge (3 cores, np=%u) bit-identical", np);
        check(first_diff_u8(gpu.data(), merged.core().data(), m) == m, buf);
        // Estimation stays on CPU: load the merged registers into an hll_t
        // and require the same estimate as the CPU-merged sketch.
        sketch::hll::hll_t est(np);
        std::memcpy(est.mutable_core().data(), gpu.data(), m);
        const double a = est.report(), b = merged.report();
        std::snprintf(buf, sizeof buf, "hll merged registers report same cardinality (np=%u: %lf vs %lf)", np, a, b);
        check(a == b, buf);
        // In-place merge over row 0.
        {
            std::vector<std::uint8_t> all2 = all;
            std::uint8_t *d_all2 = skgpu::memcpy_h2d(all2.data(), all2.size());
            skgpu::hll_merge(d_all2, d_all2, 3, m);
            SKGPU_CK(cudaDeviceSynchronize());
            std::vector<std::uint8_t> gpu0(m);
            skgpu::memcpy_d2h(gpu0.data(), d_all2, m);
            skgpu::free_dev(d_all2);
            std::snprintf(buf, sizeof buf, "hll_merge in-place on row 0 (np=%u) bit-identical", np);
            check(first_diff_u8(gpu0.data(), merged.core().data(), m) == m, buf);
        }
        skgpu::free_dev(d_all); skgpu::free_dev(d_out);
    }
}

static void test_bf() {
    for(const std::uint32_t nh: {4u, 6u}) { // 6 exercises the partial seed group (npw = 4 -> groups 4 + 2)
        const std::uint32_t l2sz = 16;
        const std::size_t ne = std::size_t(1) << 16;
        sketch::bf_t ref(l2sz, nh, 137);
        const auto &seeds = ref.seeds();
        const std::uint32_t npw = skgpu::bf_npw(l2sz), ng = skgpu::bf_ngroups(nh, npw);
        if(seeds.size() != ng) {
            std::fprintf(stderr, "[FAIL] bf seed count %zu != ngroups %u\n", seeds.size(), ng);
            ++g_fail;
            continue;
        }
        sketch::hash::WangHash wh;
        std::vector<std::uint64_t> els = mk_stream(ne, 137), flat(ne * ng);
        for(std::size_t e = 0; e < ne; ++e) {
            ref.addh(els[e]);
            for(std::size_t g = 0; g < ng; ++g)
                flat[e * ng + g] = wh(els[e] ^ seeds[g]);
        }
        std::uint64_t *d_flat = skgpu::memcpy_h2d(flat.data(), flat.size());
        const std::size_t nwords = ref.core().size();
        std::uint64_t *d_core = static_cast<std::uint64_t *>(skgpu::malloc_dev(nwords * sizeof(std::uint64_t)));
        SKGPU_CK(cudaMemset(d_core, 0, nwords * sizeof(std::uint64_t)));
        skgpu::bf_update(d_flat, ne, l2sz, nh, d_core);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> gpu(nwords);
        skgpu::memcpy_d2h(gpu.data(), d_core, nwords);
        skgpu::free_dev(d_flat); skgpu::free_dev(d_core);
        char buf[128];
        std::snprintf(buf, sizeof buf, "bf_update l2sz=%u nh=%u bit-identical", l2sz, nh);
        check(first_diff_u64(gpu.data(), ref.core().data(), nwords) == nwords, buf);
    }
}

// Reference: ccmbase_t<Increment, ..., conservative_update=false>. Two
// ccm.h quirks shape the parameters:
//  - The non-conservative path passes maxv = 1 << nbits to range_check where
//    a bit width is expected, so for nbits >= 6 the shift count is a UB
//    power of two >= 64 and every update is silently suppressed on x86.
//    nbits <= 5 keeps the CPU reference functional.
//  - The compact vector masks on store, so CPU counts wrap mod 2^nbits; the
//    GPU kernels take nbits and reproduce the wrap exactly.
template<bool conservative>
static void count_test(const char *name, std::uint32_t l2sz, std::uint32_t nh, std::uint64_t seed,
                       const std::vector<std::uint64_t> &els, std::uint32_t nbits) {
    using Ref = sketch::ccmbase_t<sketch::update::Increment, sketch::DefaultCompactVectorType,
                                  sketch::hash::WangHash, conservative>;
    Ref ref(int(nbits), int(l2sz), nh, seed);
    std::mt19937_64 mt(seed + 4); // mirrors ccmbase_t seed generation
    std::vector<std::uint64_t> seeds(nh);
    for(auto &s: seeds) s = mt();
    sketch::hash::WangHash wh;
    std::vector<std::uint64_t> flat(els.size() * nh);
    for(std::size_t e = 0; e < els.size(); ++e) {
        ref.addh(els[e]);
        for(std::size_t i = 0; i < nh; ++i)
            flat[e * nh + i] = wh(els[e] ^ seeds[i]);
    }
    std::uint64_t *d_flat = skgpu::memcpy_h2d(flat.data(), flat.size());
    const std::size_t ncnt = std::size_t(nh) << l2sz;
    std::uint32_t *d_cnt = static_cast<std::uint32_t *>(skgpu::malloc_dev(ncnt * sizeof(std::uint32_t)));
    SKGPU_CK(cudaMemset(d_cnt, 0, ncnt * sizeof(std::uint32_t)));
    if(std::strcmp(name, "cbf") == 0) skgpu::cbf_add(d_flat, els.size(), nh, l2sz, d_cnt, nbits);
    else                              skgpu::ccm_add(d_flat, els.size(), nh, l2sz, d_cnt, nbits);
    SKGPU_CK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> gpu(ncnt);
    skgpu::memcpy_d2h(gpu.data(), d_cnt, ncnt);
    skgpu::free_dev(d_flat); skgpu::free_dev(d_cnt);
    std::vector<std::uint32_t> cpu(ncnt);
    std::size_t j = 0;
    ref.for_each_register([&](const auto v) { cpu[j++] = std::uint32_t(v); });
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s l2sz=%u nh=%u bit-identical (n=%zu)", name, l2sz, nh, els.size());
    check(j == ncnt && std::equal(gpu.begin(), gpu.end(), cpu.begin()), buf);
}

static void test_cbf_ccm() {
    count_test<false>("cbf", 14, 3, 137, mk_stream(std::size_t(1) << 16, 137), 5);
    count_test<false>("ccm", 12, 5, 42,  mk_stream(std::size_t(1) << 16, 4242), 5);
    // Duplication-heavy stream: totals must match exactly (atomicAdd commutes).
    std::vector<std::uint64_t> dup(std::size_t(1) << 17);
    std::mt19937_64 mt(7);
    for(auto &v: dup) v = mt() & 1023;
    count_test<false>("ccm", 6, 2, 99, dup, 5);
}

static void test_bbmh() {
    const std::uint32_t p = 12, nsk = 512;
    const std::size_t nreg = std::size_t(1) << p;
    for(int heavy = 0; heavy < 2; ++heavy) {
        const std::size_t n = std::size_t(1) << 18;
        std::vector<std::uint32_t> sids(n);
        std::vector<std::uint64_t> hvs(n);
        std::mt19937_64 mts(137), mth(7331);
        std::vector<std::uint64_t> smallset(64);
        for(auto &v: smallset) v = mth();
        for(std::size_t i = 0; i < n; ++i) {
            sids[i] = std::uint32_t(mts() % nsk);
            hvs[i] = heavy ? smallset[mts() % smallset.size()]: mth();
        }
        std::vector<sketch::BBitMinHasher<std::uint64_t>> refs;
        refs.reserve(nsk);
        for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(p);
        for(std::size_t i = 0; i < n; ++i) refs[sids[i]].add(hvs[i]);
        std::uint32_t *d_sids = skgpu::memcpy_h2d(sids.data(), sids.size());
        std::uint64_t *d_hvs = skgpu::memcpy_h2d(hvs.data(), hvs.size());
        std::uint64_t *d_core = static_cast<std::uint64_t *>(skgpu::malloc_dev(nsk * nreg * sizeof(std::uint64_t)));
        SKGPU_CK(cudaMemset(d_core, 0xFF, nsk * nreg * sizeof(std::uint64_t))); // default_val<T>() = ~0
        skgpu::bbmh_batch_update(d_sids, d_hvs, n, p, d_core);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> gpu(nsk * nreg);
        skgpu::memcpy_d2h(gpu.data(), d_core, nsk * nreg);
        skgpu::free_dev(d_sids); skgpu::free_dev(d_hvs); skgpu::free_dev(d_core);
        std::size_t bad = nsk * nreg;
        for(std::uint32_t s = 0; s < nsk && bad == nsk * nreg; ++s) {
            const std::size_t d = first_diff_u64(gpu.data() + s * nreg, refs[s].core().data(), nreg);
            if(d != nreg) bad = s * nreg + d;
        }
        char buf[128];
        std::snprintf(buf, sizeof buf, "bbmh_batch_update %s bit-identical (nsk=%u p=%u n=%zu)",
                      heavy ? "duplicates-heavy": "random", nsk, p, n);
        check(bad == nsk * nreg, buf);
        if(bad != nsk * nreg)
            std::fprintf(stderr, "  first diff at sketch %zu register %zu\n", bad / nreg, bad % nreg);
    }
}

static void test_fmh() {
    const std::uint32_t nsk = 64;
    const std::size_t n = std::size_t(1) << 20;
    struct Case { std::uint64_t max_hash; std::size_t n; };
    for(const Case c: {Case{~std::uint64_t(0) >> 8, n}, Case{~std::uint64_t(0), std::size_t(1) << 16}}) {
        std::vector<std::uint32_t> sids(c.n);
        std::vector<std::uint64_t> hvs(c.n);
        std::mt19937_64 mts(137), mth(4242);
        for(std::size_t i = 0; i < c.n; ++i) {
            sids[i] = std::uint32_t(mts() % nsk);
            hvs[i] = mth();
            if((i & 1023) == 0) hvs[i] = ~std::uint64_t(0); // sentinel must be dropped on both sides
        }
        std::vector<sketch::fracminhashbase_t<>> refs;
        refs.reserve(nsk);
        for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(c.max_hash);
        // add(), not addh(): the GPU consumes pre-hashed values, so the CPU
        // reference must see the same hash values (addh would hash again).
        for(std::size_t i = 0; i < c.n; ++i) refs[sids[i]].add(hvs[i]);
        std::uint32_t *d_sids = skgpu::memcpy_h2d(sids.data(), sids.size());
        std::uint64_t *d_hvs = skgpu::memcpy_h2d(hvs.data(), hvs.size());
        std::uint64_t *d_kept = static_cast<std::uint64_t *>(skgpu::malloc_dev(c.n * sizeof(std::uint64_t)));
        std::uint32_t *d_ksid = static_cast<std::uint32_t *>(skgpu::malloc_dev(c.n * sizeof(std::uint32_t)));
        std::uint32_t *d_cnt = static_cast<std::uint32_t *>(skgpu::malloc_dev(nsk * sizeof(std::uint32_t)));
        std::uint64_t *d_min = static_cast<std::uint64_t *>(skgpu::malloc_dev(nsk * sizeof(std::uint64_t)));
        std::uint32_t *d_tick = static_cast<std::uint32_t *>(skgpu::malloc_dev(sizeof(std::uint32_t)));
        SKGPU_CK(cudaMemset(d_cnt, 0, nsk * sizeof(std::uint32_t)));
        SKGPU_CK(cudaMemset(d_min, 0xFF, nsk * sizeof(std::uint64_t)));
        SKGPU_CK(cudaMemset(d_tick, 0, sizeof(std::uint32_t)));
        skgpu::fmh_batch_filter(d_sids, d_hvs, c.n, c.max_hash, d_kept, d_ksid, d_cnt, d_min, d_tick);
        SKGPU_CK(cudaDeviceSynchronize());
        std::uint32_t total = 0;
        skgpu::memcpy_d2h(&total, d_tick, 1);
        std::vector<std::uint64_t> kept;
        std::vector<std::uint32_t> ksid;
        if(total) {
            kept.resize(total); ksid.resize(total);
            skgpu::memcpy_d2h(kept.data(), d_kept, total);
            skgpu::memcpy_d2h(ksid.data(), d_ksid, total);
        }
        std::vector<std::uint32_t> cnt(nsk);
        std::vector<std::uint64_t> mins(nsk);
        skgpu::memcpy_d2h(cnt.data(), d_cnt, nsk);
        skgpu::memcpy_d2h(mins.data(), d_min, nsk);
        skgpu::free_dev(d_sids); skgpu::free_dev(d_hvs); skgpu::free_dev(d_kept); skgpu::free_dev(d_ksid);
        skgpu::free_dev(d_cnt); skgpu::free_dev(d_min); skgpu::free_dev(d_tick);
        std::vector<std::vector<std::uint64_t>> gpu_sets(nsk);
        for(std::size_t i = 0; i < total; ++i) gpu_sets[ksid[i]].push_back(kept[i]);
        bool ok = true;
        for(std::uint32_t s = 0; s < nsk && ok; ++s) {
            std::vector<std::uint64_t> cpu_set;
            refs[s].for_each_kept([&](std::uint64_t v) { cpu_set.push_back(v); });
            std::sort(gpu_sets[s].begin(), gpu_sets[s].end());
            std::sort(cpu_set.begin(), cpu_set.end());
            ok = gpu_sets[s] == cpu_set && cnt[s] == refs[s].size() && mins[s] == refs[s].min_hash();
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "fmh_batch_filter max_hash=0x%llx bit-identical kept sets/counts/mins (kept %u)",
                      (unsigned long long)c.max_hash, total);
        check(ok, buf);
    }
}

int main() {
    std::fprintf(stderr, "sketch gpu bit-identity tests\n");
    try {
        test_hash();
        test_hll_update();
        test_hll_update_hashed();
        test_hll_merge();
        test_bf();
        test_cbf_ccm();
        test_bbmh();
        test_fmh();
    } catch(const std::exception &e) {
        std::fprintf(stderr, "[FAIL] uncaught exception: %s\n", e.what());
        ++g_fail;
    }
    std::fprintf(stderr, "summary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? EXIT_FAILURE: EXIT_SUCCESS;
}
