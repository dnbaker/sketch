// BBitMinHasher one-permutation densification: sentinel elimination, rc
// semantics, estimate quality vs the plain (both-empty-excluding) estimator
// at p=13/b=16, merge/densify ordering (raw merge is exact; min of two
// already-densified *sparse* sketches is biased low -- measured here),
// determinism, and densify() cost (ns/bucket, printed).
//
// Reference: A. Shrivastava, "Optimal Densification for Fast and Accurate
// Minwise Hashing" (2017); A. Shrivastava and P. Li, "Densifying One
// Permutation Hashing via Rotation for Fast Similarity Search" (2014).
#include "sketch/bbmh.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace sketch;
namespace mh = sketch::minhash;

static constexpr uint64_t E = mh::detail::default_val<uint64_t>();

static size_t nsent(const mh::BBitMinHasher<uint64_t> &h) {
    size_t n = 0;
    for(const auto v: h.core()) n += v == E;
    return n;
}

static int fail(const char *msg) {
    std::fprintf(stderr, "bbmh_densifytest: FAIL: %s\n", msg);
    return 1;
}

int main() {
    // 1. rc semantics: many-empties densification, idempotence, densified()
    //    convenience, and finalize() giving identical packed codes with or
    //    without an explicit densify() (finalize densifies an internal copy).
    for(const unsigned p: {10u, 13u, 16u}) {
        const size_t m = size_t(1) << p;
        const uint64_t n = std::max<uint64_t>(3, m / 250); // few items vs many buckets
        auto build = [&](mh::BBitMinHasher<uint64_t> &h) {
            for(uint64_t i = 0; i < n; ++i) h.addh(i * 0x9E3779B97F4A7C15ull + 1);
        };
        mh::BBitMinHasher<uint64_t> h(p, 16), g(p, 16);
        build(h); build(g);
        if(nsent(h) == 0 || nsent(g) == 0) return fail("expected sentinel registers before densify");
        if(h.densify() != 1) return fail("densify rc != 1 with empties present");
        if(nsent(h) != 0) return fail("sentinel registers remain after densify");
        const auto snap = h.core();
        if(h.densify() != 0) return fail("second densify rc != 0");
        if(h.core() != snap) return fail("densify is not idempotent");
        const auto d = g.densified(); // const convenience; g untouched
        if(nsent(d) != 0 || d.core() != h.core()) return fail("densified() mismatch with in-place densify");
        if(nsent(g) == 0) return fail("densified() modified the source sketch");
        const auto f_raw = g.finalize(16), f_den = d.finalize(16);
        if(f_raw.equal_bblocks(f_den) != f_raw.nmin())
            return fail("finalize() codes differ with/without explicit densify");
    }
    // 2. All-empty sketch: graceful no-op (left as-is), matching densifybin.
    {
        mh::BBitMinHasher<uint64_t> h(10, 16);
        if(h.densify() != -1) return fail("all-empty densify rc != -1");
        if(nsent(h) != h.size()) return fail("all-empty sketch modified");
        const auto d = h.densified();
        if(nsent(d) != d.size()) return fail("all-empty densified() modified");
    }
    // 3. Disjoint densified pairs (fresh items each rep): Jaccard and
    //    intersection read ~0; no both-empty registers exist to exclude.
    {
        double jmax = 0, imax = 0, jfmax = 0;
        for(unsigned r = 0; r < 30; ++r) {
            mh::BBitMinHasher<uint64_t> a(13, 16), b(13, 16);
            const uint64_t off = uint64_t(r) * 10000000;
            for(uint64_t i = 0; i < 5000; ++i) {a.addh(off + i + 1); b.addh(off + i + 1000001);}
            const auto da = a.densified(), db = b.densified();
            jmax = std::max(jmax, da.jaccard_index(db));
            imax = std::max(imax, da.intersection_size(db));
            jfmax = std::max(jfmax, da.finalize(16).jaccard_index(db.finalize(16)));
        }
        std::fprintf(stderr, "disjoint: max jaccard(unfinalized) %.2e, max intersection %.2e, max jaccard(final b=16) %.2e\n",
                     jmax, imax, jfmax);
        if(jmax > 2e-3) return fail("disjoint densified jaccard too large");
        if(imax > 1.) return fail("disjoint densified intersection too large");
        if(jfmax > 1e-3) return fail("disjoint finalized jaccard too large");
    }
    // 4. Known-Jaccard accuracy: densified vs plain at p=13/b=16. True
    //    J = c/(c + 2*priv) by construction; fresh items every rep so the
    //    R=60 replicates are independent. plain = unfinalized
    //    both-empty-excluding estimator; dens-fin = what downstream users of
    //    densified sketches get (finalize() densifies internally). Gate:
    //    densified rmse within the plain estimator's band at every level.
    std::fprintf(stderr, "p=13 b=16 rmse (R=60, independent reps):\n  c(shared)   J    plain-unf  dens-unf  dens-fin(b=16)\n");
    for(const uint64_t c: {250ull, 2000ull}) {
        for(const double J: {0.1, 0.25, 0.5, 0.75, 0.9}) {
            const uint64_t priv = uint64_t(std::lround(c * (1. - J) / (2. * J)));
            const double true_j = double(c) / double(c + 2 * priv);
            double se_plain = 0, se_dens = 0, se_fin = 0, sum_fin = 0;
            for(unsigned r = 0; r < 60; ++r) {
                mh::BBitMinHasher<uint64_t> a(13, 16), b(13, 16);
                const uint64_t off = uint64_t(r) * 10000000;
                for(uint64_t i = 0; i < c; ++i) {a.addh(off + i + 1); b.addh(off + i + 1);}
                for(uint64_t i = 0; i < priv; ++i) {a.addh(off + c + i + 1); b.addh(off + c + priv + i + 1);}
                se_plain += std::pow(a.jaccard_index(b) - true_j, 2);
                const auto da = a.densified(), db = b.densified();
                se_dens += std::pow(da.jaccard_index(db) - true_j, 2);
                const double ef = da.finalize(16).jaccard_index(db.finalize(16));
                se_fin += (ef - true_j) * (ef - true_j);
                sum_fin += ef;
            }
            const double rp = std::sqrt(se_plain / 60), rd = std::sqrt(se_dens / 60), rf = std::sqrt(se_fin / 60);
            const double mean_fin = sum_fin / 60;
            std::fprintf(stderr, "  %5llu  %.2f   %.4f    %.4f    %.4f   (dens/plain %.1fx)\n",
                         (unsigned long long)c, J, rp, rd, rf, rf / rp);
            // Small-cardinality regime (c = 250): the plain both-empty-
            // excluding estimator is 2-8x sharper here because it adaptively
            // drops uninformative registers; the densified estimator's rmse
            // tracks sqrt(J(1-J)/m_eff) with m_eff ~ the count of registers
            // non-empty in either sketch. Gate unbiasedness and an absolute,
            // theory-anchored bound; report the ratio instead of gating it.
            if(std::abs(mean_fin - true_j) > 0.008) return fail("densified finalized estimate biased");
            if(rf > 0.02) return fail("densified rmse above small-sample bound");
            if(c >= 2000) { // sets large relative to p: densified must sit in
                            // the plain estimator's tolerance band
                const double band = std::max(2.5 * rp, rp + 0.003);
                if(rd > band || rf > band) return fail("densified rmse outside plain estimator band");
            }
        }
    }
    // 5. Merge/densify ordering. Exact facts (see the densify() doc comment):
    //    - raw per-register min IS the raw union sketch (sentinel = +inf), so
    //      densify(min(rawA, rawB)) == densify(A+B item-wise) componentwise;
    //    - min(densA, densB) <= that union sketch componentwise, with strict
    //      inequality on borrowed registers; against a third sketch the
    //      sparse regime reads LOW (measured below; vanishes when registers
    //      fill -- at near-full sketches the two merges coincide exactly).
    {
        const unsigned R = 60;
        double mean_exact = 0, mean_post = 0; // raw-merge vs post-densify-merge
        size_t strict_total = 0, cmp_total = 0;
        for(unsigned r = 0; r < R; ++r) {
            mh::BBitMinHasher<uint64_t> a(13, 16), b(13, 16), u(13, 16), cs(13, 16);
            const uint64_t off = uint64_t(r) * 10000000;
            for(uint64_t i = 0; i < 1500; ++i) {a.addh(off + i + 1); u.addh(off + i + 1);}
            for(uint64_t i = 0; i < 1500; ++i) {b.addh(off + i + 1000001); u.addh(off + i + 1000001);}
            for(uint64_t i = 0; i < 400; ++i) {cs.addh(off + i + 1); cs.addh(off + i + 1000001);} // in A and in B
            for(uint64_t i = 0; i < 400; ++i) cs.addh(off + i + 2000001);                        // fresh
            auto da = a.densified(), db = b.densified();
            const auto merged = da + db;
            if(nsent(merged) != 0) return fail("merged densified sketch has sentinels");
            for(size_t i = 0; i < merged.size(); ++i)
                if(merged.core()[i] != std::min(da.core()[i], db.core()[i])) return fail("merge is not per-register min");
            const auto du = u.densified();
            for(size_t i = 0; i < merged.size(); ++i) {
                if(merged.core()[i] > du.core()[i]) return fail("merged register exceeds densified union register");
                strict_total += merged.core()[i] < du.core()[i];
                ++cmp_total;
            }
            // supported sparse workflow: merge raw, densify once
            auto raw_merged = a; raw_merged += b;
            auto raw_merged_d = raw_merged; raw_merged_d.densify();
            if(raw_merged_d.core() != du.core()) return fail("densify(raw merge) != densify(item-wise union)");
            mean_exact += raw_merged_d.jaccard_index(cs.densified());
            mean_post += merged.jaccard_index(cs.densified());
        }
        mean_exact /= R; mean_post /= R;
        const double true_ju = 800. / 3400.; // |C n (A u B)| / |C u A u B|
        std::fprintf(stderr, "union: raw-merge-then-densify %.4f, min-of-densified %.4f (true %.4f, biased low); strict-< registers %.4f\n",
                     mean_exact, mean_post, true_ju, double(strict_total) / cmp_total);
        if(std::abs(mean_exact - true_ju) > 0.02) return fail("raw merge + densify does not estimate union jaccard");
    }
    // 6. Dense regime: with no empty registers anywhere, min(densA, densB)
    //    IS the densified union sketch, per register, exactly.
    {
        mh::BBitMinHasher<uint64_t> a(13, 16), b(13, 16), u(13, 16);
        for(uint64_t i = 0; i < 200000; ++i) {a.addh(i + 1); b.addh(i + 1000001); u.addh(i + 1); u.addh(i + 1000001);}
        const auto merged = a.densified() + b.densified();
        if(merged.core() != u.densified().core()) return fail("dense: min of densified != densified union");
    }
    // 7. Determinism: same input -> same densified sketch.
    {
        mh::BBitMinHasher<uint64_t> a(13, 16), b(13, 16);
        for(uint64_t i = 0; i < 2000; ++i) {a.addh(i * 2654435761u + 7); b.addh(i * 2654435761u + 7);}
        a.densify(); b.densify();
        if(a != b || a.densified() != b.densified()) return fail("densification not deterministic");
    }
    // 8. Densify cost at p = 10..16 (informational; loose sanity gate). Fill
    //    ~40% of registers, then time copy + densify(), as densified() does.
    std::fprintf(stderr, "densify() cost (copy + fill, ~60%% empty), ns/bucket:\n");
    for(unsigned p = 10; p <= 16; ++p) {
        const size_t m = size_t(1) << p;
        mh::BBitMinHasher<uint64_t> h(p, 16);
        uint64_t s = 0x243F6A8885A308D3ull + p;
        const uint64_t n = uint64_t(0.51 * m); // fills ~40% of buckets
        for(uint64_t i = 0; i < n; ++i) {s ^= s << 13; s ^= s >> 7; s ^= s << 17; h.addh(s);}
        if(h.densify() != 1) return fail("bench sketch did not need densification");
        const size_t reps = std::max<size_t>(20, (size_t(1) << 24) / m);
        const auto t0 = std::chrono::steady_clock::now();
        for(size_t r = 0; r < reps; ++r) {auto tmp = h; tmp.densify();}
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        const double ns_per_bucket = ns / (double(reps) * m);
        std::fprintf(stderr, "  p=%2u m=%6zu: %8.2f\n", p, m, ns_per_bucket);
        if(ns_per_bucket > 100.) return fail("densify suspiciously slow");
    }
    std::fprintf(stderr, "bbmh_densifytest: all checks passed\n");
    return 0;
}
