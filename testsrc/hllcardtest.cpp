#include "hll.h"
#include "bbmh.h"
#include "setsketch.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>

using namespace sketch;
using setsketch::SetSketch;
using SSType = setsketch::CSetSketch<double>;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

// ---------------- HLL ----------------
// Accuracy vs theory at several cardinalities, merge correctness,
// union size, serialization round-trips.
void test_hll() {
    const unsigned p = 12;
    const double m = double(size_t(1) << p);
    const double theory = 1.03896 / std::sqrt(m);
    const double ns[] = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7};
    for(const double n: ns) {
        std::vector<double> errs;
        for(unsigned seed = 1; seed <= 15; ++seed) {
            hll::hll_t h(p);
            std::mt19937_64 rng(seed * 4099 + unsigned(n));
            for(uint64_t i = 0; i < uint64_t(n); ++i) h.addh(rng());
            errs.push_back(h.report() / n - 1.);
        }
        const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
        const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
        std::fprintf(stderr, "hll n=%8.0f rms=%.5f (theory %.5f) bias=%+.5f\n", n, rms, theory, bias);
        CHECK(rms <= 1.75 * theory, "hll rms %.5f vs theory %.5f at n=%g", rms, theory, n);
        CHECK(std::abs(bias) <= 0.5 * theory, "hll bias %f at n=%g", bias, n);
        for(double e: errs) CHECK(std::abs(e) < 7. * theory, "hll outlier %f at n=%g", e, n);
    }
    // merge of disjoint streams must equal direct insertion, register-exact
    for(unsigned np: {5u, 10u, 14u}) {
        std::mt19937_64 rng(np);
        hll::hll_t a(np), b(np), both(np);
        for(unsigned i = 0; i < 100000; ++i) {const uint64_t v = rng(); a.addh(v); both.addh(v);}
        for(unsigned i = 0; i < 100000; ++i) {const uint64_t v = rng(); b.addh(v); both.addh(v);}
        a += b;
        CHECK(a == both, "hll merge mismatch at np=%u", np);
        CHECK(std::abs(a.report() - 200000.) < 4. * a.relative_error() * 200000., "hll merged estimate off: %f", a.report());
        // union_size of identical sketches must equal the cardinality
        const double us = a.union_size(a);
        CHECK(std::abs(us - a.report()) / a.report() < 1e-6, "hll union_size(self) %f != report %f", us, a.report());
    }
    // serialization round-trip (path-based gz and FILE*-based)
    {
        hll::hll_t h(10);
        std::mt19937_64 rng(2024);
        for(unsigned i = 0; i < 50000; ++i) h.addh(rng());
        h.sum();
        const char *path = "/tmp/hll_roundtrip.hll";
        h.write(path);
        hll::hll_t h2(path);
        CHECK(h == h2, "hll gz round-trip mismatch");
        CHECK(std::abs(h2.report() - h.report()) < 1e-9, "hll report changed across round-trip");
        // uncompressed FILE* path
        h.write("/tmp/hll_roundtrip.raw", false);
        hll::hll_t h3;
        h3.read("/tmp/hll_roundtrip.raw");
        CHECK(h == h3, "hll raw round-trip mismatch");
        std::remove(path);
        std::remove("/tmp/hll_roundtrip.raw");
    }
    std::fprintf(stderr, "hll tests done\n");
}

// ---------------- BBitMinHash ----------------
void test_bbmh() {
    const unsigned p = 14, b = 16;
    // controlled overlap: A = [0,300000) u [2^40, 2^40+100000),
    //                 B = [200000,500000);
    // |A| = 400000, |B| = 300000, |A&B| = 100000, |A u B| = 600000, J = 1/6.
    // NOTE: the library's harmonic cardinality estimator for finalized
    // b-bit minhash is only valid when the sketch is essentially full
    // (empty registers otherwise dominate the harmonic sum); k = n/m ~ 37
    // here keeps it in the valid regime.
    minhash::BBitMinHasher<uint64_t> c1(p, b), c2(p, b), cboth(p, b);
    for(uint64_t v = 0; v < 300000; ++v) {c1.addh(v); cboth.addh(v);}
    for(uint64_t v = 0; v < 100000; ++v) {c1.addh(v + (uint64_t(1) << 40)); cboth.addh(v + (uint64_t(1) << 40));}
    for(uint64_t v = 200000; v < 500000; ++v) {c2.addh(v); cboth.addh(v);}
    auto f1 = c1.finalize(), f2 = c2.finalize(), fboth = cboth.finalize();
    const double true_j = 100000. / 600000.;
    const double est_j = f1.jaccard_index(f2);
    CHECK(std::abs(est_j - true_j) < 0.02, "bbmh jaccard %f != %f", est_j, true_j);
    const double est_card = fboth.cardinality_estimate();
    CHECK(std::abs(est_card - 600000.) < 0.03 * 600000., "bbmh cardinality %f != 600000", est_card);
    // serialization round-trip of the final sketch
    const char *path = "/tmp/bbmh_roundtrip.bbmh";
    fboth.write(path);
    minhash::FinalBBitMinHash f3(fboth.p_, fboth.b_, fboth.est_cardinality_);
    f3.read(path);
    CHECK(f3 == fboth, "bbmh round-trip mismatch");
    std::remove(path);
    std::fprintf(stderr, "bbmh tests done\n");
}

// ---------------- SetSketch ----------------
void test_setsketch() {
    const size_t m = 1024;
    SSType s1(m), s2(m), both(m);
    for(uint64_t v = 0; v < 300000; ++v) {s1.addh(v); both.addh(v);}
    for(uint64_t v = 0; v < 100000; ++v) {s1.addh(v + (uint64_t(1) << 40)); both.addh(v + (uint64_t(1) << 40));}
    for(uint64_t v = 200000; v < 500000; ++v) {s2.addh(v); both.addh(v);}
    const double true_j = 100000. / 600000.;
    const double est_j = s1.jaccard_index(s2);
    const double est_card = both.cardinality();
    std::fprintf(stderr, "setsketch jaccard: est=%.4f true=%.4f card=%.0f (true 600000)\n", est_j, true_j, est_card);
    CHECK(std::abs(est_j - true_j) < 0.03, "setsketch jaccard %f != %f", est_j, true_j);
    CHECK(std::abs(est_card - 600000.) < 0.05 * 600000., "setsketch cardinality %f", est_card);
    // serialization round-trip
    const char *path = "/tmp/ss_roundtrip.ss";
    both.write(std::string(path));
    SSType both2(m);
    both2.read(std::string(path));
    CHECK(std::abs(both2.cardinality() - both.cardinality()) < 1e-6 * both.cardinality(), "setsketch round-trip mismatch");
    std::remove(path);
    std::fprintf(stderr, "setsketch tests done\n");
}

int main() {
    test_hll();
    test_bbmh();
    test_setsketch();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all hll/bbmh/setsketch gap tests passed\n");
    return 0;
}
