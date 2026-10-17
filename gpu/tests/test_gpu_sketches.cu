// Bit-identity correctness tests for the sketch GPU layer.
//
// This TU is compiled by nvcc and includes ONLY gpu/*.cuh + reference.h —
// never the CPU sketch headers (nvcc frontend ICEs / include-order-dependent
// constexpr errors; see tests/reference.{h,cpp}, where all CPU-header
// reference computation lives, built as a plain-g++ object).
//
// Per sketch type: build the reference via skref:: on the same pre-hashed
// stream, run the GPU kernel, and require byte-identical sketch state (fmh
// kept hashes are compared sorted, since GPU compaction order is
// nondeterministic). Exit status: nonzero on any failed check.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

#include "sketch/gpu/hash.cuh"
#include "sketch/gpu/hll.cuh"
#include "sketch/gpu/bloom.cuh"
#include "sketch/gpu/count.cuh"
#include "sketch/gpu/minhash.cuh"

#include "reference.h"

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
    for(std::size_t i = 0; i < 256; ++i) in[i] = i; // small integers too
    std::uint64_t *d_in = skgpu::memcpy_h2d(in.data(), in.size());
    std::uint64_t *d_out = static_cast<std::uint64_t *>(skgpu::malloc_dev(n * sizeof(std::uint64_t)));
    skgpu::hash(d_in, d_out, n);
    skgpu::memcpy_d2h(out.data(), d_out, n);
    skgpu::free_dev(d_in); skgpu::free_dev(d_out);
    std::size_t bad = n;
    for(std::size_t i = 0; i < n; ++i) if(out[i] != skref::splitmix64(in[i])) { bad = i; break; }
    check(bad == n, "splitmix64_hash bit-identical to CPU SplitMix64Hash");
    if(bad != n) std::fprintf(stderr, "  first mismatch at %zu: in=%016llx gpu=%016llx cpu=%016llx\n",
                              bad, (unsigned long long)in[bad], (unsigned long long)out[bad], (unsigned long long)skref::splitmix64(in[bad]));
}

static void test_hll_update() {
    const std::uint64_t seeds[] {137, 42};
    for(const std::uint32_t np: {4u, 10u, 16u, 20u}) {
        for(const std::uint64_t seed: seeds) {
            const std::size_t n = std::size_t(1) << 20;
            const std::size_t m = std::size_t(1) << np;
            std::vector<std::uint64_t> in = seed == 137 ? mk_edgy_stream(n): mk_stream(n, seed);
            std::vector<std::uint8_t> ref(m);
            skref::hll_update(in.data(), n, np, ref.data());
            std::uint64_t *d_in = skgpu::memcpy_h2d(in.data(), in.size());
            std::uint8_t *d_core = static_cast<std::uint8_t *>(skgpu::malloc_dev(m));
            SKGPU_CK(cudaMemset(d_core, 0, m));
            skgpu::hll_update(d_in, n, np, d_core);
            SKGPU_CK(cudaDeviceSynchronize());
            std::vector<std::uint8_t> gpu(m);
            skgpu::memcpy_d2h(gpu.data(), d_core, m);
            skgpu::free_dev(d_in); skgpu::free_dev(d_core);
            const std::size_t bad = first_diff_u8(gpu.data(), ref.data(), m);
            char buf[128];
            std::snprintf(buf, sizeof buf, "hll_update np=%u seed=%llu bit-identical (n=%zu, m=%zu)",
                          np, (unsigned long long)seed, n, m);
            check(bad == m, buf);
            if(bad != m) std::fprintf(stderr, "  first diff at register %zu: gpu=%u cpu=%u\n", bad, unsigned(gpu[bad]), unsigned(ref[bad]));
        }
    }
}

static void test_hll_update_hashed() {
    const std::size_t n = std::size_t(1) << 20;
    std::vector<std::uint64_t> in = mk_stream(n, 137);
    for(const std::uint32_t np: {10u, 16u}) {
        const std::size_t m = std::size_t(1) << np;
        std::vector<std::uint8_t> ref(m);
        skref::hll_update_hashed(in.data(), n, np, ref.data());
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
        check(first_diff_u8(gpu.data(), ref.data(), m) == m, buf);
    }
}

static void test_hll_merge() {
    const std::size_t n = std::size_t(1) << 18;
    for(const std::uint32_t np: {10u, 16u}) {
        const std::size_t m = std::size_t(1) << np;
        std::vector<std::uint64_t> s0 = mk_stream(n, 137), s1 = mk_stream(n, 138), s2 = mk_stream(n, 139);
        std::vector<std::uint8_t> all(3 * m), ref(m);
        double ref_report = 0.;
        skref::hll_merge3(s0.data(), s1.data(), s2.data(), n, np, ref.data(), &ref_report);
        {   // The GPU merge consumes the three register arrays; build them
            // with the same reference entry point, one stream each.
            std::vector<std::uint8_t> c0(m), c1(m), c2(m);
            skref::hll_update(s0.data(), n, np, c0.data());
            skref::hll_update(s1.data(), n, np, c1.data());
            skref::hll_update(s2.data(), n, np, c2.data());
            std::memcpy(all.data(), c0.data(), m);
            std::memcpy(all.data() + m, c1.data(), m);
            std::memcpy(all.data() + 2 * m, c2.data(), m);
        }
        std::uint8_t *d_all = skgpu::memcpy_h2d(all.data(), all.size());
        std::uint8_t *d_out = static_cast<std::uint8_t *>(skgpu::malloc_dev(m));
        skgpu::hll_merge(d_out, d_all, 3, m);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint8_t> gpu(m);
        skgpu::memcpy_d2h(gpu.data(), d_out, m);
        char buf[160];
        std::snprintf(buf, sizeof buf, "hll_merge (3 cores, np=%u) bit-identical", np);
        check(first_diff_u8(gpu.data(), ref.data(), m) == m, buf);
        // Estimation stays on CPU: merged registers must give the same
        // cardinality as the CPU-merged sketch.
        const double a = skref::hll_report(gpu.data(), np);
        std::snprintf(buf, sizeof buf, "hll merged registers report same cardinality (np=%u: %lf vs %lf)", np, a, ref_report);
        check(a == ref_report, buf);
        // In-place merge over row 0.
        {
            std::uint8_t *d_all2 = skgpu::memcpy_h2d(all.data(), all.size());
            skgpu::hll_merge(d_all2, d_all2, 3, m);
            SKGPU_CK(cudaDeviceSynchronize());
            std::vector<std::uint8_t> gpu0(m);
            skgpu::memcpy_d2h(gpu0.data(), d_all2, m);
            skgpu::free_dev(d_all2);
            std::snprintf(buf, sizeof buf, "hll_merge in-place on row 0 (np=%u) bit-identical", np);
            check(first_diff_u8(gpu0.data(), ref.data(), m) == m, buf);
        }
        skgpu::free_dev(d_all); skgpu::free_dev(d_out);
    }
}

static void test_bf() {
    for(const std::uint32_t nh: {4u, 6u}) { // 6 exercises the partial seed group (npw = 4 -> groups 4 + 2)
        const std::uint32_t l2sz = 16;
        const std::size_t ne = std::size_t(1) << 16;
        const std::uint32_t ng = skref::bf_ngroups(l2sz, nh);
        std::vector<std::uint64_t> els = mk_stream(ne, 137), flat(ne * ng), ref(std::size_t(1) << (l2sz - 6));
        skref::bf(els.data(), ne, l2sz, nh, flat.data(), ref.data());
        std::uint64_t *d_flat = skgpu::memcpy_h2d(flat.data(), flat.size());
        const std::size_t nwords = ref.size();
        std::uint64_t *d_core = static_cast<std::uint64_t *>(skgpu::malloc_dev(nwords * sizeof(std::uint64_t)));
        SKGPU_CK(cudaMemset(d_core, 0, nwords * sizeof(std::uint64_t)));
        skgpu::bf_update(d_flat, ne, l2sz, nh, d_core);
        SKGPU_CK(cudaDeviceSynchronize());
        std::vector<std::uint64_t> gpu(nwords);
        skgpu::memcpy_d2h(gpu.data(), d_core, nwords);
        skgpu::free_dev(d_flat); skgpu::free_dev(d_core);
        char buf[128];
        std::snprintf(buf, sizeof buf, "bf_update l2sz=%u nh=%u bit-identical", l2sz, nh);
        check(first_diff_u64(gpu.data(), ref.data(), nwords) == nwords, buf);
    }
}

// Reference: skref::ccm (ccmbase_t conservative_update=false; nbits <= 5
// keeps the CPU reference functional — see reference.h / gpu/README.md).
static void count_test(const char *name, std::uint32_t l2sz, std::uint32_t nh, std::uint64_t seed,
                       const std::vector<std::uint64_t> &els, std::uint32_t nbits) {
    std::vector<std::uint64_t> flat(els.size() * nh);
    std::vector<std::uint32_t> ref(std::size_t(nh) << l2sz);
    skref::ccm(els.data(), els.size(), l2sz, nh, seed, nbits, flat.data(), ref.data());
    std::uint64_t *d_flat = skgpu::memcpy_h2d(flat.data(), flat.size());
    const std::size_t ncnt = ref.size();
    std::uint32_t *d_cnt = static_cast<std::uint32_t *>(skgpu::malloc_dev(ncnt * sizeof(std::uint32_t)));
    SKGPU_CK(cudaMemset(d_cnt, 0, ncnt * sizeof(std::uint32_t)));
    if(std::strcmp(name, "cbf") == 0) skgpu::cbf_add(d_flat, els.size(), nh, l2sz, d_cnt, nbits);
    else                              skgpu::ccm_add(d_flat, els.size(), nh, l2sz, d_cnt, nbits);
    SKGPU_CK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> gpu(ncnt);
    skgpu::memcpy_d2h(gpu.data(), d_cnt, ncnt);
    skgpu::free_dev(d_flat); skgpu::free_dev(d_cnt);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s l2sz=%u nh=%u bit-identical (n=%zu)", name, l2sz, nh, els.size());
    check(std::equal(gpu.begin(), gpu.end(), ref.begin()), buf);
}

static void test_cbf_ccm() {
    count_test("cbf", 14, 3, 137, mk_stream(std::size_t(1) << 16, 137), 5);
    count_test("ccm", 12, 5, 42,  mk_stream(std::size_t(1) << 16, 4242), 5);
    // Duplication-heavy stream: totals must match exactly (atomicAdd commutes).
    std::vector<std::uint64_t> dup(std::size_t(1) << 17);
    std::mt19937_64 mt(7);
    for(auto &v: dup) v = mt() & 1023;
    count_test("ccm", 6, 2, 99, dup, 5);
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
        std::vector<std::uint64_t> ref(nsk * nreg);
        skref::bbmh(sids.data(), hvs.data(), n, p, nsk, ref.data());
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
            const std::size_t d = first_diff_u64(gpu.data() + s * nreg, ref.data() + s * nreg, nreg);
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
        // CPU reference over the identical stream.
        std::vector<std::uint64_t> cpukept(c.n);
        std::vector<std::uint32_t> cpusid(c.n), cpucnt(nsk);
        std::vector<std::uint64_t> cpumin(nsk);
        const std::size_t cpu_total = skref::fmh(sids.data(), hvs.data(), c.n, c.max_hash, nsk,
                                                 cpukept.data(), cpusid.data(), cpucnt.data(), cpumin.data());
        bool ok = total == cpu_total;
        std::vector<std::vector<std::uint64_t>> gpu_sets(nsk), cpu_sets(nsk);
        for(std::size_t i = 0; i < total; ++i) gpu_sets[ksid[i]].push_back(kept[i]);
        for(std::size_t i = 0; i < cpu_total; ++i) cpu_sets[cpusid[i]].push_back(cpukept[i]);
        for(std::uint32_t s = 0; s < nsk && ok; ++s) {
            std::sort(gpu_sets[s].begin(), gpu_sets[s].end());
            std::sort(cpu_sets[s].begin(), cpu_sets[s].end());
            ok = gpu_sets[s] == cpu_sets[s] && cnt[s] == cpucnt[s] && mins[s] == cpumin[s];
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
