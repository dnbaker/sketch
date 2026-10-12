#include "dartminhash.h"
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

// Jaccard accuracy at known overlap, several sketch sizes (the point of
// DartMinHash: unbiased estimation even at small k).
void test_jaccard() {
    const double js[] = {0.1, 0.3, 0.5, 0.9};
    const unsigned ks[] = {64, 256, 1024};
    const uint64_t n = 50000; // plenty: bucket-min statistics depend on k, not n
    for(const unsigned k: ks) {
        for(const double J: js) {
            std::vector<double> errs;
            const uint64_t i = uint64_t(2. * n * J / (1. + J) + .5); // |A n B|
            const double true_j = double(i) / double(2 * n - i);
            const unsigned nseeds = k <= 256 ? 15 : 5;
            for(unsigned seed = 1; seed <= nseeds; ++seed) {
                dmh::dartminhashbase_t<> a(k, seed), b(k, seed), both(k, seed);
                for(uint64_t v = 0; v < n; ++v) {a.addh(v); both.addh(v);}
                for(uint64_t v = 0; v < i; ++v) {b.addh(v); both.addh(v);}
                for(uint64_t v = n; v < 2 * n - i; ++v) {b.addh(v); both.addh(v);}
                errs.push_back(a.jaccard_index(b) - true_j);
            }
            {   // union size consistency on one instance
                dmh::dartminhashbase_t<> a(k, 1), b(k, 1);
                for(uint64_t v = 0; v < n; ++v) a.addh(v);
                for(uint64_t v = 0; v < i; ++v) b.addh(v);
                for(uint64_t v = n; v < 2 * n - i; ++v) b.addh(v);
                CHECK(std::abs(a.union_size(b) - (a + b).report()) < 1e-9, "union_size != merged report");
            }
            const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
            const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
            const double theory = std::sqrt(true_j * (1. - true_j) / k);
            std::fprintf(stderr, "dmh k=%4u J=%.1f rms=%.5f (theory %.5f) bias=%+.5f\n", k, J, rms, theory, bias);
            CHECK(rms <= 1.9 * theory + 1e-9, "dmh jaccard rms %.5f vs %.5f (k=%u,J=%g)", rms, theory, k, J);
            CHECK(std::abs(bias) <= 0.85 * theory + 1e-9, "dmh jaccard bias %.5f (k=%u,J=%g)", bias, k, J);
        }
    }
}

// Containment at known overlap.
void test_containment() {
    const uint64_t n = 50000;
    const unsigned k = 1024;
    for(const double J: {0.1, 0.3, 0.5, 0.9}) {
        const uint64_t i = uint64_t(2. * n * J / (1. + J) + .5);
        const double true_j = double(i) / double(2 * n - i);
        const double true_c = double(i) / n; // containment of B in A
        std::vector<double> errs;
        for(unsigned seed = 1; seed <= 15; ++seed) {
            dmh::dartminhashbase_t<> a(k, seed), b(k, seed);
            for(uint64_t v = 0; v < n; ++v) a.addh(v);
            for(uint64_t v = 0; v < i; ++v) b.addh(v);
            for(uint64_t v = n; v < 2 * n - i; ++v) b.addh(v);
            errs.push_back(a.containment_index(b) - true_c);
        }
        const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
        const double theory = std::sqrt(true_j * (1. - true_j) / k); // error propagated from J^
        // noise of the inclusion-exclusion containment is dominated by the
        // union-sketch cardinality error (CV ~ 1/sqrt(k)), scaled by n/i;
        // the binomial 'theory' is shown for reference only.
        const double cbias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        std::fprintf(stderr, "dmh containment J=%.1f rms=%.5f (binom %.5f, card-CV scaled %.5f) bias=%+.5f\n",
                     J, rms, theory, std::sqrt(2.) / std::sqrt(double(k)) / true_c, cbias);
        CHECK(rms < 6. * theory + 0.02, "dmh containment rms %.5f (J=%g)", rms, J);
        // max(0, .) censoring in inclusion-exclusion allows a small positive
        // bias on the order of the estimator's own noise (same convention
        // as hll::containment_index); require it to stay within that.
        CHECK(std::abs(cbias) < 1.1 * rms + 1e-9, "dmh containment bias %.5f vs own rms (J=%g)", cbias, J);
    }
}

// Cardinality at several sizes (k = 4096 registers; theory CV ~ 1/sqrt(m),
// m = min(n, k) filled registers for the moment estimator).
void test_cardinality() {
    const unsigned k = 1024;
    const double ns[] = {1e2, 1e3, 1e4, 1e5, 1e6};
    for(const double n: ns) {
        std::vector<double> errs;
        for(unsigned seed = 1; seed <= 7; ++seed) {
            dmh::dartminhashbase_t<> h(k, seed);
            DefaultRNGType rng(seed * 2017 + unsigned(n));
            for(uint64_t i = 0; i < uint64_t(n); ++i) h.addh(rng());
            errs.push_back(h.report() / n - 1.);
        }
        const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
        const double m = std::min(n, double(k));
        const double theory = 1. / std::sqrt(m);
        std::fprintf(stderr, "dmh n=%8.0f rms=%.5f (theory %.5f)\n", n, rms, theory);
        CHECK(rms <= 2.2 * theory + 1e-9, "dmh card rms %.5f vs %.5f (n=%g)", rms, theory, n);
        CHECK(std::abs(std::accumulate(errs.begin(), errs.end(), 0.) / errs.size()) <= 1.2 * theory + 1e-9, "dmh card bias (n=%g)", n);
    }
    { // single-seed large-n smoke (keeps runtime sane: cost is n*k/LANES)
        dmh::dartminhashbase_t<> h(k, 3);
        DefaultRNGType rng(3);
        const uint64_t n = 10000000;
        for(uint64_t i = 0; i < n; ++i) h.addh(rng());
        const double relerr = std::abs(h.report() / n - 1.);
        std::fprintf(stderr, "dmh n=1e7 (1 seed) relerr=%.5f (theory %.5f)\n", relerr, 1. / std::sqrt(double(k)));
        CHECK(relerr < 4. / std::sqrt(double(k)), "dmh n=1e7 relerr %.5f", relerr);
    }
}

// Merge equality/idempotency/commutativity, serialization round-trip.
void test_merge_and_serialization() {
    for(const unsigned k: {64u, 256u, 1000u}) {
        dmh::dartminhashbase_t<> a(k, 7), b(k, 7), both(k, 7);
        DefaultRNGType rng(k);
        for(unsigned i = 0; i < 60000; ++i) {const uint64_t v = rng(); a.addh(v); both.addh(v);}
        for(unsigned i = 0; i < 60000; ++i) {const uint64_t v = rng(); b.addh(v); both.addh(v);}
        a += b;
        CHECK(a == both, "dmh merge mismatch (k=%u)", k);
        // commutativity of merge
        dmh::dartminhashbase_t<> x(k, 7), y(k, 7);
        for(unsigned i = 0; i < 50000; ++i) {x.addh(rng()); y.addh(rng());}
        CHECK((x + y) == (y + x), "dmh merge not commutative (k=%u)", k);
        // idempotency
        dmh::dartminhashbase_t<> h1(k, 7), h2(k, 7);
        for(unsigned i = 0; i < 50000; ++i) {const uint64_t v = rng() % 1000; h1.addh(v); h2.addh(v);}
        for(unsigned i = 0; i < 50000; ++i) h1.addh(rng() % 1000);
        CHECK(h1 == h2, "dmh idempotency violated (k=%u)", k);
        // serialization
        const char *path = "/tmp/dmh_roundtrip.dmh";
        both.write(path);
        dmh::dartminhashbase_t<> r(k, 7);
        r.read(path);
        CHECK(r == both, "dmh serialization round-trip mismatch (k=%u)", k);
        CHECK(std::abs(r.report() - both.report()) < 1e-9, "dmh estimate changed across round-trip");
        std::remove(path);
    }
    std::fprintf(stderr, "dmh merge/serialization tests done\n");
}

// Bit-exact equivalence of the SIMD register update against an independent
// scalar reference (guards the vectorized WangHash/min/blend path).
void test_simd_equivalence() {
    for(const unsigned k: {3u, 64u, 255u, 1024u}) { // odd sizes exercise the tail
        dmh::dartminhashbase_t<> h(k, 99);
        DefaultRNGType rng(k);
        for(unsigned i = 0; i < 20000; ++i) h.addh(rng());
        for(unsigned j = 0; j < k; ++j) {
            const uint64_t seed_j = 99 + uint64_t(j) * dmh::detail::DMH_GOLDEN;
            uint64_t exp_rank = ~uint64_t(0), exp_id = 0;
            DefaultRNGType rng2(k);
            for(unsigned i = 0; i < 20000; ++i) {
                const uint64_t v = rng2();
                const uint64_t r = sketch::hash::WangHash::hash(h.hash(v) ^ seed_j);
                if(r < exp_rank) {exp_rank = r; exp_id = h.hash(v);}
            }
            CHECK(h.ranks()[j] == exp_rank && h.ids()[j] == exp_id,
                  "dmh SIMD/scalar mismatch at k=%u j=%u", k, j);
        }
    }
    std::fprintf(stderr, "dmh SIMD equivalence done\n");
}

int main() {
    test_simd_equivalence();
    test_jaccard();
    test_containment();
    test_cardinality();
    test_merge_and_serialization();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all DartMinHash tests passed\n");
    return 0;
}
