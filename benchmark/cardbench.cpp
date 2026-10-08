/*
 * Cardinality sketch micro-benchmark: insert, merge, and estimation
 * throughput for hll_t, ultraloglog_t, and exaloglog_t.
 *
 * Built via the repo Makefile (make cardbench). For hll merge before/after
 * comparisons against a pre-cleanup revision, stage the old include tree
 * and compile with -I pointing at it first:
 *   git archive <rev> include | tar -x -C /tmp/oldsketch
 *   g++ ... -I/tmp/oldsketch/include/sketch -I/tmp/oldsketch/include ...
 */
#include "hll.h"
// SKETCH_BENCH_HLL_ONLY: build against revisions predating the
// register-compressed sketches to A/B the hll merge path.
#ifndef SKETCH_BENCH_HLL_ONLY
#include "ultraloglog.h"
#include "exaloglog.h"
#endif
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace sketch;
using clk = std::chrono::high_resolution_clock;

static double now_s(clk::time_point t0) {
    return std::chrono::duration<double>(t0 - clk::now()).count();
}

int main(int argc, char **argv) {
    const uint64_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10): 2000000ull;
    const unsigned p = argc > 2 ? std::atoi(argv[2]): 14;
    std::mt19937_64 rng(1337);
    std::vector<uint64_t> vals(n);
    for(auto &v: vals) v = rng();

    std::fprintf(stderr, "cardinality sketch bench, n=%llu p=%u\n", (unsigned long long)n, p);
    // --- insert throughput ---
    {
        auto t0 = clk::now();
        hll::hll_t h(p);
        for(const auto v: vals) h.addh(v);
        const double dt = -now_s(t0);
        std::fprintf(stderr, "hll  addh : %8.3f M/s (est %.0f)\n", n / dt / 1e6, h.report());
    }
#ifndef SKETCH_BENCH_HLL_ONLY
    {
        auto t0 = clk::now();
        ull::ultraloglog_t h(p);
        for(const auto v: vals) h.addh(v);
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ull  addh : %8.3f M/s (est %.0f)\n", n / dt / 1e6, h.report());
    }
    {
        auto t0 = clk::now();
        ell::exaloglog_t h(2, 20, p);
        for(const auto v: vals) h.addh(v);
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ell  addh : %8.3f M/s (est %.0f)\n", n / dt / 1e6, h.report());
    }
#endif
    // --- merge throughput (m = 2^p registers) ---
    {
        hll::hll_t a(p), b(p);
        for(size_t i = 0; i < n; ++i) {a.addh(vals[i]); b.addh(vals[(i + n / 2) % n]);}
        const unsigned iters = 2000;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) {a += b;}
        const double dt = -now_s(t0);
        std::fprintf(stderr, "hll  merge: %8.3f Mreg/s (%u iters)\n", iters * double(size_t(1) << p) / dt / 1e6, iters);
    }
#ifndef SKETCH_BENCH_HLL_ONLY
    {
        ull::ultraloglog_t a(p), b(p);
        for(size_t i = 0; i < n; ++i) {a.addh(vals[i]); b.addh(vals[(i + n / 2) % n]);}
        const unsigned iters = 2000;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) {a += b;}
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ull  merge: %8.3f Mreg/s (%u iters)\n", iters * double(size_t(1) << p) / dt / 1e6, iters);
    }
#endif
#ifndef SKETCH_BENCH_HLL_ONLY
    {
        ell::exaloglog_t a(2, 20, p), b(2, 20, p);
        for(size_t i = 0; i < n; ++i) {a.addh(vals[i]); b.addh(vals[(i + n / 2) % n]);}
        const unsigned iters = 1000;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) {a += b;}
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ell  merge: %8.3f Mreg/s (%u iters)\n", iters * double(size_t(1) << p) / dt / 1e6, iters);
    }
#endif
    // --- estimation cost ---
    {
        hll::hll_t h(p);
        for(const auto v: vals) h.addh(v);
        const unsigned iters = 50;
        double sink = 0;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) {h.sum(); sink += h.report();}
        const double dt = -now_s(t0);
        std::fprintf(stderr, "hll  mle-est: %7.3f Mreg/s (%u iters, sink=%g)\n", iters * double(size_t(1) << p) / dt / 1e6, iters, sink);
    }
#ifndef SKETCH_BENCH_HLL_ONLY
    {
        ull::ultraloglog_t h(p);
        for(const auto v: vals) h.addh(v);
        const unsigned iters = 200;
        double sink = 0;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) sink += h.fgra_estimate();
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ull  fgra-est: %7.3f Mreg/s (%u iters, sink=%g)\n", iters * double(size_t(1) << p) / dt / 1e6, iters, sink);
        auto t1 = clk::now();
        for(unsigned i = 0; i < iters / 4; ++i) sink += h.mle_estimate();
        const double dt2 = -now_s(t1);
        std::fprintf(stderr, "ull  mle-est: %7.3f Mreg/s (%u iters, sink=%g)\n", iters / 4 * double(size_t(1) << p) / dt2 / 1e6, iters / 4, sink);
    }
#endif
#ifndef SKETCH_BENCH_HLL_ONLY
    {
        ell::exaloglog_t h(2, 20, p);
        for(const auto v: vals) h.addh(v);
        const unsigned iters = 100;
        double sink = 0;
        auto t0 = clk::now();
        for(unsigned i = 0; i < iters; ++i) sink += h.estimate();
        const double dt = -now_s(t0);
        std::fprintf(stderr, "ell  mle-est: %7.3f Mreg/s (%u iters, sink=%g)\n", iters * double(size_t(1) << p) / dt / 1e6, iters, sink);
    }
#endif
    return 0;
}
