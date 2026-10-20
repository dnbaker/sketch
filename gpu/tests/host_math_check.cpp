// Host-only verification of the GPU layer's scalar semantics.
//
// This machine has no CUDA toolchain (PI rule: never install dev tools), so
// the kernels in gpu/sketch.cu cannot be compiled or run here. What CAN be
// verified on the host is the bit-identity contract itself: every kernel is
// an order-insensitive fold (max/min/add/or) of the SKGPU_HD scalar helpers
// in include/sketch/gpu/*.cuh, so a serial fold of the same helpers over the
// same pre-hashed stream must equal both the CPU reference sketch AND the GPU
// kernel output (which test_gpu_sketches.cu checks on a real device).
//
// Build (plain C++, no CUDA): see `make hostcheck` in gpu/Makefile.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
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
std::vector<std::uint64_t> mk_edgy_stream(std::size_t n) {
    std::mt19937_64 mt(42);
    std::vector<std::uint64_t> ret(n);
    for(std::size_t i = 0; i < n; ++i) {
        switch(i & 7) {
            case 0: ret[i] = i; break;
            case 1: ret[i] = std::uint64_t(1) << 63; break;
            case 2: ret[i] = ~std::uint64_t(0); break;
            case 3: ret[i] = std::uint64_t(1); break;
            case 4: ret[i] = ~std::uint64_t(0) >> 20; break;
            default: ret[i] = mt(); break;
        }
    }
    return ret;
}
} // anonymous namespace

static void check_hash() {
    sketch::hash::SplitMix64Hash sm;
    bool ok = true;
    std::mt19937_64 mt(137);
    for(std::size_t i = 0; ok && i < (std::size_t(1) << 20); ++i) {
        const std::uint64_t x = i < 256 ? i: mt();
        ok = skgpu::splitmix64_hash(x) == sm(x);
    }
    check(ok, "skgpu::splitmix64_hash == sketch::hash::SplitMix64Hash");
}

static void check_hll() {
    for(const std::uint32_t np: {4u, 10u, 16u, 20u}) {
        const std::size_t n = std::size_t(1) << 18, m = std::size_t(1) << np;
        std::vector<std::uint64_t> in = mk_edgy_stream(n);
        sketch::hll::hll_t ref(np);
        std::vector<std::uint8_t> gpu(m, 0);
        for(const auto hv: in) {
            ref.add(hv);
            const std::uint32_t idx = skgpu::hll_index(hv, np);
            const std::uint8_t lzt = skgpu::hll_lzt(hv, np);
            gpu[idx] = gpu[idx] > lzt ? gpu[idx]: lzt; // serial fold of hll_byte_atomic_max
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "hll update semantics np=%u", np);
        check(std::equal(gpu.begin(), gpu.end(), ref.core().begin()), buf);
    }
    // Merge = per-byte max over cores.
    {
        const std::uint32_t np = 12;
        const std::size_t m = std::size_t(1) << np;
        std::vector<std::uint8_t> rows(3 * m);
        sketch::hll::hll_t merged(np);
        for(std::size_t s = 0; s < 3; ++s) {
            sketch::hll::hll_t ref(np);
            for(const auto hv: mk_stream(std::size_t(1) << 14, 137 + s)) ref.add(hv);
            std::copy(ref.core().begin(), ref.core().end(), rows.begin() + s * m);
            merged += ref;
        }
        std::vector<std::uint8_t> gpu(m);
        for(std::size_t i = 0; i < m; ++i) {
            std::uint8_t v = rows[i];
            for(std::size_t s = 1; s < 3; ++s) v = rows[s * m + i] > v ? rows[s * m + i]: v;
            gpu[i] = v;
        }
        check(std::equal(gpu.begin(), gpu.end(), merged.core().begin()), "hll merge semantics (per-byte max)");
    }
}

static void check_bf() {
    for(const std::uint32_t nh: {4u, 6u}) {
        const std::uint32_t l2sz = 16;
        const std::size_t ne = std::size_t(1) << 14;
        sketch::bf_t ref(l2sz, nh, 137);
        const auto &seeds = ref.seeds();
        const std::uint32_t npw = skgpu::bf_npw(l2sz), ng = skgpu::bf_ngroups(nh, npw);
        const std::uint64_t mask = (std::uint64_t(1) << l2sz) - 1;
        sketch::hash::WangHash wh;
        std::vector<std::uint64_t> gpu(ref.core().size(), 0);
        std::mt19937_64 mt(137);
        for(std::size_t e = 0; e < ne; ++e) {
            const std::uint64_t x = mt();
            ref.addh(x);
            for(std::size_t g = 0; g < ng; ++g) {
                const std::uint64_t hv = wh(x ^ seeds[g]);
                const std::uint32_t gsize = skgpu::bf_group_size(std::uint32_t(g), nh, npw);
                for(std::uint32_t i = 0; i < gsize; ++i)
                    skgpu::bf_set1(gpu.data(), mask, hv >> (i * l2sz));
            }
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "bf sub_set1 semantics nh=%u", nh);
        check(std::equal(gpu.begin(), gpu.end(), ref.core().begin()), buf);
    }
}

// Reference: ccmbase_t conservative_update=false. nbits <= 5 keeps the CPU
// reference functional (the non-conservative path passes maxv = 1 << nbits
// to range_check where a bit width belongs; >= 6 makes the shift count UB
// and suppresses every update on x86 — a ccm.h quirk). The compact vector
// masks on store, so CPU counts wrap mod 2^nbits; the serial fold below
// applies the same wrap the GPU kernel reproduces with counter_add_wrap.
template<bool conservative>
static void check_count(const char *name, std::uint32_t l2sz, std::uint32_t nh, std::uint64_t seed,
                        const std::vector<std::uint64_t> &els, std::uint32_t nbits) {
    using Ref = sketch::ccmbase_t<sketch::update::Increment, sketch::DefaultCompactVectorType,
                                  sketch::hash::WangHash, conservative>;
    Ref ref(int(nbits), int(l2sz), nh, seed);
    std::mt19937_64 mt(seed + 4);
    std::vector<std::uint64_t> seeds(nh);
    for(auto &s: seeds) s = mt();
    sketch::hash::WangHash wh;
    std::vector<std::uint32_t> gpu(std::size_t(nh) << l2sz, 0);
    const std::uint32_t wrapmask = (std::uint32_t(1) << nbits) - 1;
    for(const auto x: els) {
        ref.addh(x);
        for(std::uint32_t i = 0; i < nh; ++i) {
            auto &c = gpu[skgpu::ccm_index(wh(x ^ seeds[i]), i, l2sz)];
            c = (c + 1) & wrapmask;
        }
    }
    std::vector<std::uint32_t> cpu(gpu.size());
    std::size_t j = 0;
    ref.for_each_register([&](const auto v) { cpu[j++] = std::uint32_t(v); });
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s counter-add semantics l2sz=%u nh=%u", name, l2sz, nh);
    check(j == gpu.size() && std::equal(gpu.begin(), gpu.end(), cpu.begin()), buf);
}

static void check_bbmh() {
    const std::uint32_t p = 12, nsk = 64;
    const std::size_t nreg = std::size_t(1) << p, n = std::size_t(1) << 16;
    std::vector<std::uint64_t> gpu(nsk * nreg, ~std::uint64_t(0));
    std::vector<sketch::BBitMinHasher<std::uint64_t>> refs;
    for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(p);
    std::mt19937_64 mts(137), mth(7331);
    for(std::size_t i = 0; i < n; ++i) {
        const std::uint32_t sid = std::uint32_t(mts() % nsk);
        const std::uint64_t hv = mth();
        refs[sid].add(hv);
        std::uint64_t &slot = gpu[(std::size_t(sid) << p) + skgpu::bbmh_bucket(hv, p)];
        const std::uint64_t val = skgpu::bbmh_val(hv, p);
        slot = slot < val ? slot: val;
    }
    bool ok = true;
    for(std::uint32_t s = 0; s < nsk && ok; ++s)
        ok = std::equal(gpu.begin() + (std::size_t(s) << p), gpu.begin() + (std::size_t(s + 1) << p),
                        refs[s].core().begin());
    check(ok, "bbmh batch min semantics");
}

static void check_fmh() {
    const std::uint64_t max_hash = ~std::uint64_t(0) >> 8;
    const std::uint32_t nsk = 16;
    const std::size_t n = std::size_t(1) << 18;
    std::vector<sketch::fracminhashbase_t<>> refs;
    for(std::uint32_t s = 0; s < nsk; ++s) refs.emplace_back(max_hash);
    std::vector<std::uint64_t> mins(nsk, ~std::uint64_t(0));
    std::vector<std::uint32_t> counts(nsk, 0);
    std::vector<std::vector<std::uint64_t>> kept(nsk);
    std::mt19937_64 mts(137), mth(4242);
    for(std::size_t i = 0; i < n; ++i) {
        const std::uint32_t sid = std::uint32_t(mts() % nsk);
        const std::uint64_t h = mth();
        refs[sid].add(h); // pre-hashed stream, like the GPU kernel consumes
        if(skgpu::fmh_keep(h, max_hash)) {
            kept[sid].push_back(h);
            mins[sid] = h < mins[sid] ? h: mins[sid];
            ++counts[sid];
        }
    }
    bool ok = true;
    for(std::uint32_t s = 0; s < nsk && ok; ++s) {
        std::vector<std::uint64_t> cpu_set;
        refs[s].for_each_kept([&](std::uint64_t v) { cpu_set.push_back(v); });
        std::sort(kept[s].begin(), kept[s].end());
        std::sort(cpu_set.begin(), cpu_set.end());
        ok = kept[s] == cpu_set && counts[s] == refs[s].size() && mins[s] == refs[s].min_hash();
    }
    check(ok, "fmh filter semantics (keep/count/min)");
}

int main() {
    std::fprintf(stderr, "sketch gpu host scalar-semantics checks (no CUDA required)\n");
    check_hash();
    check_hll();
    check_bf();
    check_count<false>("cbf", 14, 3, 137, mk_stream(std::size_t(1) << 14, 137), 5);
    // nbits <= 5: see the note above count_test in test_gpu_sketches.cu —
    // the CPU non-conservative path's range_check gets maxv, so >= 6
    // suppresses every update on x86 (UB shift by a power of two >= 64).
    check_count<false>("ccm", 6, 2, 99, mk_stream(std::size_t(1) << 14, 7), 5);
    check_bbmh();
    check_fmh();
    std::fprintf(stderr, "summary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? EXIT_FAILURE: EXIT_SUCCESS;
}
