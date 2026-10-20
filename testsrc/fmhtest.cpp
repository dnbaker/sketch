#include "fracminhash.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>

using namespace sketch;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

// Cardinality at several sizes; scale chosen to keep ~n/scale samples.
void test_cardinality() {
    const struct {double n; double scale;} cases[] = {
        {1e2, 1.}, {1e3, 1.}, {1e4, 2.}, {1e5, 8.}, {1e6, 64.}, {1e7, 512.}
    };
    for(const auto &c: cases) {
        std::vector<double> errs;
        for(unsigned seed = 1; seed <= 9; ++seed) {
            fmh::fracminhash_t h(double(c.scale));
            DefaultRNGType rng(seed * 911 + unsigned(c.n));
            for(uint64_t i = 0; i < uint64_t(c.n); ++i) h.addh(rng());
            errs.push_back(h.report() / c.n - 1.);
        }
        const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
        const double theory = std::sqrt(c.scale / c.n); // Poisson: std = sqrt(s/n) relative
        const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        std::fprintf(stderr, "fmh n=%8.0f scale=%4.0f rms=%.5f (theory %.5f) bias=%+.5f\n", c.n, c.scale, rms, theory, bias);
        CHECK(rms <= 1.9 * theory, "fmh rms %.5f vs theory %.5f (n=%g)", rms, theory, c.n);
        CHECK(std::abs(bias) <= 0.6 * theory, "fmh bias %f (n=%g)", bias, c.n);
        for(double e: errs) CHECK(std::abs(e) < 6. * theory + 1e-9, "fmh outlier %f (n=%g)", e, c.n);
        // min-hash cross-check: kept ~ max_hash / min_kept
        if(c.n >= 1e4) {
            fmh::fracminhash_t h(double(c.scale));
            DefaultRNGType rng(5);
            for(uint64_t i = 0; i < uint64_t(c.n); ++i) h.addh(rng());
            // min-of-kept estimates |kept| ~ max_hash / min; n^ = scale * that.
            // (single order statistic: tolerance is loose by nature)
            const double est_min = h.scale() * double(h.max_hash()) / double(h.min_hash());
            CHECK(std::abs(est_min / h.report() - 1.) < 9., "min-hash cross-check off: %f vs %f", est_min, h.report());
        }
    }
}

// Containment and Jaccard at known overlap.
void test_containment() {
    const double js[] = {0.1, 0.3, 0.5, 0.9};
    const double scale = 10., K = 1000000.; // B has K elements -> ~1e5 kept
    for(const double J: js) {
        std::vector<double> cerrs, jerrs;
        for(unsigned seed = 1; seed <= 5; ++seed) {
            // B = [0, K); A = [0, J*K) u [2^62, 2^62 + (1-J)*K): |A|=|B|=K, intersection = J*K
            const uint64_t C = uint64_t(J * K), D = uint64_t((1. - J) * K);
            fmh::fracminhash_t a{double(scale)}, b{double(scale)};
            DefaultRNGType rng(seed * 31337 + unsigned(J * 100));
            for(uint64_t i = 0; i < K; ++i) b.addh(i);
            for(uint64_t i = 0; i < C; ++i) a.addh(i);
            for(uint64_t i = 0; i < D; ++i) a.addh(i + (uint64_t(1) << 62));
            (void)rng;
            cerrs.push_back(a.containment_index(b) - J);
            // construction: |A|=|B|=K, |A&B|=J*K -> true jaccard J/(2-J)
            jerrs.push_back(a.jaccard_index(b) - J / (2. - J));
        }
        const double crms = std::sqrt(std::inner_product(cerrs.begin(), cerrs.end(), cerrs.begin(), 0.) / cerrs.size());
        const double jrms = std::sqrt(std::inner_product(jerrs.begin(), jerrs.end(), jerrs.begin(), 0.) / jerrs.size());
        const double ctheory = std::sqrt(J * (1. - J) / (K / scale)); // binomial over kept samples
        std::fprintf(stderr, "fmh J=%.1f containment rms=%.5f (binom %.5f) jaccard rms=%.5f\n", J, crms, ctheory, jrms);
        CHECK(crms <= 1.9 * ctheory + 1e-9, "containment rms %.5f vs %.5f (J=%g)", crms, ctheory, J);
        CHECK(std::abs(std::accumulate(cerrs.begin(), cerrs.end(), 0.) / cerrs.size()) <= 0.6 * ctheory + 5e-3, "containment bias (J=%g)", J);
        CHECK(jrms < 3. * ctheory + 5e-3, "jaccard rms %.5f (J=%g)", jrms, J);
        const double jbias = std::accumulate(jerrs.begin(), jerrs.end(), 0.) / jerrs.size();
        CHECK(std::abs(jbias) < 3. * ctheory + 5e-3, "jaccard bias %.5f (J=%g)", jbias, J);
    }
}

// Merge equality, idempotency, batch==scalar, serialization.
void test_merge_and_serialization() {
    fmh::fracminhash_t a(double(4)), b(double(4)), both(double(4));
    DefaultRNGType rng(42);
    for(unsigned i = 0; i < 200000; ++i) {const uint64_t v = rng(); a.addh(v); both.addh(v);}
    for(unsigned i = 0; i < 200000; ++i) {const uint64_t v = rng(); b.addh(v); both.addh(v);}
    a += b;
    CHECK(a == both, "fmh merge mismatch");
    // idempotency
    fmh::fracminhash_t h1(double(2)), h2(double(2));
    for(unsigned i = 0; i < 50000; ++i) {const uint64_t v = rng() % 5000; h1.addh(v); h2.addh(v);}
    for(unsigned i = 0; i < 50000; ++i) h1.addh(rng() % 5000);
    CHECK(h1 == h2, "fmh idempotency violated");
    // batch == scalar
    {
        std::vector<uint64_t> vals(100000);
        for(auto &v: vals) v = rng();
        fmh::fracminhash_t t1(double(8)), t2(double(8));
        t1.add(vals.data(), vals.size());
        for(const auto v: vals) t2.add(v);
        CHECK(t1 == t2, "fmh batch add != scalar add");
    }
    // serialization
    {
        const char *path = "/tmp/fmh_roundtrip.fmh";
        both.write(path);
        fmh::fracminhash_t r(path);
        CHECK(r == both, "fmh serialization round-trip mismatch");
        CHECK(std::abs(r.report() - both.report()) < 1e-9, "fmh estimate changed across round-trip");
        std::remove(path);
    }
    // merge of serialized/deserialized keeps equality of estimates
    std::fprintf(stderr, "fmh merge/serialization tests done\n");
}

int main() {
    test_cardinality();
    test_containment();
    test_merge_and_serialization();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all FracMinHash tests passed\n");
    return 0;
}
