#include "ultraloglog.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace sketch;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

// 1. Register encoding invariants: pack/unpack round-trips through updates.
void test_encoding() {
    for(unsigned p = 3; p <= 10; ++p) {
        for(unsigned trial = 0; trial < 100; ++trial) {
            ull::ultraloglog_t h(p);
            DefaultRNGType rng(p * 1000 + trial);
            for(unsigned i = 0; i < 200; ++i) h.add(rng());
            // every nonzero register must be a fixed point of decode/encode
            // (pack(0) is intentionally undefined: the empty register has no
            // hash prefix)
            for(size_t i = 0; i < h.size(); ++i) {
                const uint8_t reg = h.core()[i];
                if(reg == 0) continue;
                CHECK(sketch::ull::detail::ull_pack(sketch::ull::detail::ull_unpack(reg)) == reg,
                      "pack/unpack not a fixed point: reg=%u", reg);
            }
        }
    }
    std::fprintf(stderr, "encoding invariants done\n");
}

// 2. Accuracy vs theory for both estimators, several cardinalities.
void test_accuracy() {
    const unsigned p = 10;
    const double m = double(size_t(1) << p);
    const double mle_theory = ull::ultraloglog_t::mle_relative_error(uint64_t(m));
    const double fgra_theory = ull::ultraloglog_t::fgra_relative_error(uint64_t(m));
    const double ns[] = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7};
    for(const double n: ns) {
        std::vector<double> mle_errs, fgra_errs;
        for(unsigned seed = 1; seed <= 15; ++seed) {
            ull::ultraloglog_t h(p);
            DefaultRNGType rng(seed * 1337 + unsigned(n));
            for(uint64_t i = 0; i < uint64_t(n); ++i) h.addh(rng());
            mle_errs.push_back(h.mle_estimate() / n - 1.);
            fgra_errs.push_back(h.fgra_estimate() / n - 1.);
        }
        const auto rms = [](const std::vector<double> &v) {
            double s = 0.; for(double e: v) s += e * e;
            return std::sqrt(s / double(v.size()));
        };
        const double mle_rms = rms(mle_errs), fgra_rms = rms(fgra_errs);
        const double mle_bias = std::accumulate(mle_errs.begin(), mle_errs.end(), 0.) / double(mle_errs.size());
        const double fgra_bias = std::accumulate(fgra_errs.begin(), fgra_errs.end(), 0.) / double(fgra_errs.size());
        std::fprintf(stderr, "n=%8.0f MLE  rms=%.5f (theory %.5f) bias=%+.5f | FGRA rms=%.5f (theory %.5f) bias=%+.5f\n",
                     n, mle_rms, mle_theory, mle_bias, fgra_rms, fgra_theory, fgra_bias);
        // rms over seeds should track theory; allow generous factor for seed noise
        CHECK(mle_rms <= 1.75 * mle_theory, "MLE rms %.5f exceeds 1.75x theory %.5f at n=%g", mle_rms, mle_theory, n);
        CHECK(fgra_rms <= 1.75 * fgra_theory, "FGRA rms %.5f exceeds 1.75x theory %.5f at n=%g", fgra_rms, fgra_theory, n);
        CHECK(std::abs(mle_bias) <= 0.5 * mle_theory, "MLE bias %f too large at n=%g", mle_bias, n);
        CHECK(std::abs(fgra_bias) <= 0.5 * fgra_theory, "FGRA bias %f too large at n=%g", fgra_bias, n);
        for(double e: mle_errs) CHECK(std::abs(e) < 7. * mle_theory, "MLE outlier %f at n=%g", e, n);
    }
}

// 3. Merge: union of disjoint streams must equal direct insertion, register-exact.
void test_merge() {
    for(unsigned p = 4; p <= 12; p += 2) {
        DefaultRNGType rng(p);
        ull::ultraloglog_t a(p), b(p), both(p);
        for(unsigned i = 0; i < 100000; ++i) {
            const uint64_t v = rng();
            a.addh(v);
            both.addh(v);
        }
        for(unsigned i = 0; i < 100000; ++i) {
            const uint64_t v = rng();
            b.addh(v);
            both.addh(v);
        }
        a += b;
        CHECK(a == both, "merge mismatch at p=%u", p);
        CHECK(std::abs(a.estimate() / both.estimate() - 1.) < 1e-12, "estimate mismatch after merge");
    }
    // idempotency: re-adding the same elements must not change the state
    {
        ull::ultraloglog_t h(8), h2(8);
        DefaultRNGType rng(999);
        for(unsigned i = 0; i < 10000; ++i) {const uint64_t v = rng() % 1000; h.addh(v); h2.addh(v);}
        for(unsigned i = 0; i < 10000; ++i) h.addh(rng() % 1000); // repeats of the same values
        CHECK(h == h2, "idempotency violated");
    }
    std::fprintf(stderr, "merge tests done\n");
}

// 4. Reducibility: merging a higher-precision sketch equals direct recording.
void test_downsize() {
    const unsigned p_lo = 6, p_hi = 9;
    ull::ultraloglog_t hi(p_hi), lo_direct(p_lo), lo_merged(p_lo);
    DefaultRNGType rng(31337);
    for(unsigned i = 0; i < 200000; ++i) {
        const uint64_t v = rng();
        hi.addh(v);
        lo_direct.addh(v);
    }
    lo_merged.add(hi);
    CHECK(lo_merged == lo_direct, "downsize merge not identical to direct recording");
    // explicit downsize
    const auto d = hi.downsize(p_lo);
    CHECK(d == lo_direct, "downsize() not identical to direct recording");
    std::fprintf(stderr, "downsize tests done\n");
}

// 5. Serialization round-trip.
void test_serialization() {
    ull::ultraloglog_t h(9);
    DefaultRNGType rng(777);
    for(unsigned i = 0; i < 50000; ++i) h.addh(rng());
    const char *path = "/tmp/ull_roundtrip.ull";
    h.write(path);
    ull::ultraloglog_t h2(path);
    CHECK(h == h2, "serialization round-trip mismatch");
    CHECK(h2.estimate() == h.estimate(), "estimate changed across serialization");
    std::remove(path);
    std::fprintf(stderr, "serialization tests done\n");
}

// 6. Martingale estimation (single-stream).
void test_martingale() {
    const unsigned p = 10;
    const double m = double(size_t(1) << p);
    const double theory = std::sqrt(3.47 / 8. / m); // martingale MVP = 3.47 bits
    double worst = 0.;
    for(unsigned seed = 1; seed <= 5; ++seed) {
        ull::ultraloglog_t h(p);
        mle::detail::MartingaleEstimator mart;
        DefaultRNGType rng(seed);
        const uint64_t n = 1000000;
        for(uint64_t i = 0; i < n; ++i) h.addh(rng(), mart);
        const double relerr = mart.estimate() / double(n) - 1.;
        worst = std::max(worst, std::abs(relerr));
        CHECK(std::abs(relerr) < 4. * theory, "martingale relerr %f exceeds 4x theory %f", relerr, theory);
    }
    std::fprintf(stderr, "martingale worst relerr %.5f (theory %.5f)\n", worst, theory);
}

int main() {
    test_encoding();
    test_accuracy();
    test_merge();
    test_downsize();
    test_serialization();
    test_martingale();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all UltraLogLog tests passed\n");
    return 0;
}
