// Round 15 HyperMinHash benchmark: accuracy ladder (old vs new collision
// correction vs brute force), comparison throughput (4 types, corrected +
// raw), update throughput, merge. Ours only -- the Rust twin lives in
// /tmp/hyperminhash-rs/examples/bench_sketch.rs.
#include "hmh.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <unordered_set>
#include <vector>
using namespace sketch;
using clk = std::chrono::high_resolution_clock;

int main(int argc, char **argv) {
    const uint32_t p = argc > 1 ? atoi(argv[1]) : 14;
    const uint32_t card = argc > 2 ? atoi(argv[2]) : 100000;
    const size_t reps = argc > 3 ? atol(argv[3]) : 30;
    DefaultRNGType rng(42);
    std::vector<uint64_t> pool(card * 2);
    for(auto &v: pool) v = rng();

    // ---- accuracy ladder: J = 0.1..0.9, old-ec vs new-ec vs brute force;
    // plus intersection accuracy: MH-union vs histogram-union (both x new ji)
    std::fprintf(stderr, "accuracy ladder p=%u card=%u (reps=%zu, mean |err| and bias vs brute force)\n", p, card, reps);
    for(double J: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}) {
        double old_abs = 0, new_abs = 0, old_bias = 0, new_bias = 0, denom = 0;
        double is_mh_abs = 0, is_hist_abs = 0, is_mh_bias = 0, is_hist_bias = 0;
        for(size_t rep = 0; rep < reps; ++rep) {
            const size_t shared = size_t(J * card), only = card - shared;
            HyperMinHash a(p, 16), b(p, 16);
            std::unordered_set<uint64_t> sa, sb;
            for(size_t i = 0; i < only; ++i) {sa.insert(pool[i]); a.add(pool[i]);}
            for(size_t i = 0; i < shared; ++i) {sa.insert(pool[only + i]); sb.insert(pool[only + i]); a.add(pool[only + i]); b.add(pool[only + i]);}
            for(size_t i = 0; i < only; ++i) {sb.insert(pool[only + shared + i]); b.add(pool[only + shared + i]);}
            const double truth = double(shared) / (2. * only + shared);
            const double mycard = a.getcard(), ocard = b.getcard();
            const auto ccnc = a.overlap_counts(b);
            const double cc = ccnc.first, nc = ccnc.second;
            const double ec_old = a.hll_lazy_collision_estimate(mycard, ocard);
            const double ec_new = a.rank_interval_collision_estimate(mycard, ocard);
            auto ji = [&](double ec) {return (1. - ec / nc) * cc / nc;};
            if(cc == 0) continue;
            if(cc == nc) {old_abs += std::abs(1. - truth); new_abs += std::abs(1. - truth); old_bias += 1. - truth; new_bias += 1. - truth; denom += 1; continue;}
            old_abs += std::abs(ji(ec_old) - truth); new_abs += std::abs(ji(ec_new) - truth);
            old_bias += ji(ec_old) - truth; new_bias += ji(ec_new) - truth;
            const double jn = ji(ec_new);
            const double truth_is = double(shared);
            const double is_mh = a.union_size_mh(b) * jn;
            const double is_hist = a.union_size(b) * jn;
            is_mh_abs += std::abs(is_mh - truth_is) / truth_is; is_hist_abs += std::abs(is_hist - truth_is) / truth_is;
            is_mh_bias += (is_mh - truth_is) / truth_is; is_hist_bias += (is_hist - truth_is) / truth_is;
            denom += 1;
        }
        std::fprintf(stderr, "  J=%.1f jac: |err| old %.5f new %.5f bias %+.5f/%+.5f | inter: relerr mh %.5f hist %.5f bias %+.5f/%+.5f\n",
                     J, old_abs / denom, new_abs / denom, old_bias / denom, new_bias / denom,
                     is_mh_abs / denom, is_hist_abs / denom, is_mh_bias / denom, is_hist_bias / denom);
    }
    // ---- comparison throughput on one fixed pair
    HyperMinHash a(p, 16), b(p, 16);
    for(size_t i = 0; i < card; ++i) { a.add(pool[i]); b.add(pool[i]); }
    for(size_t i = card; i < 2 * card; ++i) { a.add(pool[i]); }
    for(size_t i = 0; i < card; ++i) { b.add(rng()); }
    const size_t npairs = 30000;
    auto bench = [&](const char *tag, auto &&fn) {
        auto t0 = clk::now();
        volatile double sink = 0;
        for(size_t i = 0; i < npairs; ++i) sink += fn(a, b);
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "cmp %-22s p=%u: %8.0f pairs/s\n", tag, p, npairs / dt);
    };
    std::fprintf(stderr, "comparison throughput (card=%u each, ~50%% overlap)\n", card);
    bench("jaccard(corrected)", [](const auto &x, const auto &y){return x.jaccard_index(y);});
    bench("intersection(corrected)", [](const auto &x, const auto &y){return x.intersection_size(y);});
    bench("union(corrected)", [](const auto &x, const auto &y){return x.union_size(y);});
    bench("containment(corrected)", [](const auto &x, const auto &y){return x.containment_index(y);});
    bench("compare(all four)", [](const auto &x, const auto &y){auto c = x.compare(y); return c.jaccard + c.intersection + c.union_cardinality + c.containment;});
    bench("jaccard_raw(cc/nc)", [](const auto &x, const auto &y){auto o = x.overlap_counts(y); return o.second ? double(o.first) / o.second: 1.;});
    // ---- update throughput
    const size_t n = 20000000;
    std::vector<uint64_t> vals(n);
    for(auto &v: vals) v = rng();
    {
        HyperMinHash h(p, 16);
        auto t0 = clk::now();
        for(size_t i = 0; i < n; ++i) h.add(vals[i]);
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "update single p=%u: %.2f M adds/s, card %.0f\n", p, n / dt / 1e6, h.getcard());
    }
    {
        HyperMinHash h(p, 16);
        auto t0 = clk::now();
        h.add(vals.data(), n);
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "update batch  p=%u: %.2f M adds/s, card %.0f\n", p, n / dt / 1e6, h.getcard());
    }
    // ---- merge throughput
    {
        HyperMinHash acc(a);
        auto t0 = clk::now();
        const size_t nm = 200000;
        for(size_t i = 0; i < nm; ++i) acc += b;
        const double dt = std::chrono::duration<double>(clk::now() - t0).count();
        std::fprintf(stderr, "merge p=%u: %.0f merges/s\n", p, nm / dt);
    }
    return 0;
}
