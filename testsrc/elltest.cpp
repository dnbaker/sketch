#include "exaloglog.h"
#include "ultraloglog.h"
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

// 0. Theory constants: MVP and ML bias correction vs reference values.
void test_theory() {
    // Paper Fig. 4 MVPs
    CHECK(std::abs(ell::exaloglog_t::theoretical_mvp(2, 20) - 3.67) < 0.02, "MVP(t=2,d=20)=%f != 3.67", ell::exaloglog_t::theoretical_mvp(2, 20));
    CHECK(std::abs(ell::exaloglog_t::theoretical_mvp(2, 24) - 3.78) < 0.02, "MVP(t=2,d=24)=%f != 3.78", ell::exaloglog_t::theoretical_mvp(2, 24));
    CHECK(std::abs(ell::exaloglog_t::theoretical_mvp(1, 9) - 3.90) < 0.02, "MVP(t=1,d=9)=%f != 3.90", ell::exaloglog_t::theoretical_mvp(1, 9));
    // t=0 must reproduce the UltraLogLog numbers (b=2, d=2: MVP 4.6313, bias 0.48147...)
    CHECK(std::abs(ell::exaloglog_t::theoretical_mvp(0, 2) - 4.6313) < 0.001, "MVP(t=0,d=2)=%f != 4.6313", ell::exaloglog_t::theoretical_mvp(0, 2));
    CHECK(std::abs(ell::detail::ell_ml_bias_correction(0, 2) - 0.48147376527720065) < 1e-9, "bias(0,2)=%.17g", ell::detail::ell_ml_bias_correction(0, 2));
    // Reference table (MLBiasCorrectionConstants.java) spot checks
    CHECK(std::abs(ell::detail::ell_ml_bias_correction(0, 0) - 1.01015908095854) < 1e-9, "bias(0,0)=%.17g", ell::detail::ell_ml_bias_correction(0, 0));
    CHECK(std::abs(ell::detail::ell_ml_bias_correction(0, 1) - 0.6574064986454711) < 1e-9, "bias(0,1)=%.17g", ell::detail::ell_ml_bias_correction(0, 1));
    CHECK(std::abs(ell::detail::ell_ml_bias_correction(1, 9) - 0.19074089378272704) < 1e-9, "bias(1,9)=%.17g", ell::detail::ell_ml_bias_correction(1, 9));
    CHECK(std::abs(ell::detail::ell_ml_bias_correction(2, 20) - 0.1055382430717344) < 1e-9, "bias(2,20)=%.17g", ell::detail::ell_ml_bias_correction(2, 20));
    std::fprintf(stderr, "theory checks done (MVP(2,20)=%.4f bias(2,20)=%.6f)\n",
                 ell::exaloglog_t::theoretical_mvp(2, 20), ell::detail::ell_ml_bias_correction(2, 20));
}

// 1. Accuracy vs theory at several cardinalities and configurations.
void test_accuracy() {
    const struct {unsigned t, d, p;} configs[] = {{2, 20, 8}, {2, 24, 8}, {1, 9, 10}, {0, 2, 10}};
    for(const auto &cfg: configs) {
        ell::exaloglog_t h0(cfg.p); // constructor default is t=2,d=20; build explicitly below
        (void)h0;
        const double m = double(uint64_t(1) << cfg.p);
        const double theory = std::sqrt(ell::exaloglog_t::theoretical_mvp(cfg.t, cfg.d) / (m * ell::exaloglog_t::regbits(cfg.t, cfg.d)));
        const double ns[] = {1e2, 1e3, 1e4, 1e5, 1e6, 1e7};
        for(const double n: ns) {
            std::vector<double> errs;
            for(unsigned seed = 1; seed <= 25; ++seed) {
                ell::exaloglog_t h(uint8_t(cfg.t), uint8_t(cfg.d), cfg.p);
                DefaultRNGType rng(seed * 7331 + unsigned(n));
                for(uint64_t i = 0; i < uint64_t(n); ++i) h.addh(rng());
                errs.push_back(h.estimate() / n - 1.);
            }
            const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / double(errs.size()));
            const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / double(errs.size());
            std::fprintf(stderr, "t=%u d=%2u p=%2u n=%8.0f rms=%.5f (theory %.5f) bias=%+.5f\n", cfg.t, cfg.d, cfg.p, n, rms, theory, bias);
            CHECK(rms <= 1.75 * theory, "rms %.5f exceeds 1.75x theory %.5f (t=%u,d=%u,n=%g)", rms, theory, cfg.t, cfg.d, n);
            // Bias bound, statistically correct for a K-run ensemble mean:
            // the bias-reduced ML estimator's residual is O(1/m^2) -- at most
            // 0.1 of one standard deviation for these configurations -- and a
            // K-run mean of per-run errors (std ~ theory) carries theory/sqrt(K)
            // of ensemble noise. The previous fixed 0.5*theory bound equals
            // 2.5 standard errors of the mean, so it fired on ensemble noise
            // alone (P ~ 1.2% per cell, ~25% per run over the grid) and was
            // misread as stochastic flakiness. The bound below is a true-bias
            // allowance plus 3 measured standard errors (no gaussian assumption).
            const double bias_bound = 0.1 * theory + 3. * rms / std::sqrt(double(errs.size()));
            CHECK(std::abs(bias) <= bias_bound, "bias %f exceeds bound %f (t=%u,d=%u,n=%g)", bias, bias_bound, cfg.t, cfg.d, n);
            for(double e: errs) CHECK(std::abs(e) < 7. * theory, "outlier %f (t=%u,d=%u,n=%g)", e, cfg.t, cfg.d, n);
        }
    }
}

// 2. Merge: union of disjoint streams must equal direct insertion, register-exact.
void test_merge() {
    for(const auto cfg: {std::pair<unsigned, unsigned>{2, 20}, {1, 9}, {0, 2}}) {
        const unsigned p = 8;
        DefaultRNGType rng(p * 7 + cfg.first);
        ell::exaloglog_t a(uint8_t(cfg.first), uint8_t(cfg.second), p),
                         b(uint8_t(cfg.first), uint8_t(cfg.second), p),
                         both(uint8_t(cfg.first), uint8_t(cfg.second), p);
        for(unsigned i = 0; i < 100000; ++i) {const uint64_t v = rng(); a.addh(v); both.addh(v);}
        for(unsigned i = 0; i < 100000; ++i) {const uint64_t v = rng(); b.addh(v); both.addh(v);}
        a += b;
        CHECK(a == both, "merge mismatch t=%u d=%u", cfg.first, cfg.second);
    }
    // idempotency
    {
        ell::exaloglog_t h(8), h2(8);
        DefaultRNGType rng(999);
        for(unsigned i = 0; i < 10000; ++i) {const uint64_t v = rng() % 1000; h.addh(v); h2.addh(v);}
        for(unsigned i = 0; i < 10000; ++i) h.addh(rng() % 1000);
        CHECK(h == h2, "idempotency violated");
    }
    std::fprintf(stderr, "merge tests done\n");
}

// 3. Reducibility: merging a higher-precision (or higher-d) sketch equals
//    direct recording.
void test_downsize() {
    {
        const unsigned p_lo = 6, p_hi = 9;
        ell::exaloglog_t hi(2, 20, p_hi), lo_direct(2, 20, p_lo), lo_merged(2, 20, p_lo);
        DefaultRNGType rng(31337);
        for(unsigned i = 0; i < 200000; ++i) {const uint64_t v = rng(); hi.addh(v); lo_direct.addh(v);}
        lo_merged.add(hi);
        CHECK(lo_merged == lo_direct, "precision downsize not identical to direct recording");
        CHECK(hi.downsize(20, p_lo) == lo_direct, "downsize() not identical to direct recording");
    }
    {
        // d-reducibility: merge a d=24 sketch into d=20 at same p
        const unsigned p = 8;
        ell::exaloglog_t hi(2, 24, p), lo_direct(2, 20, p), lo_merged(2, 20, p);
        DefaultRNGType rng(5150);
        for(unsigned i = 0; i < 200000; ++i) {const uint64_t v = rng(); hi.addh(v); lo_direct.addh(v);}
        lo_merged.add(hi);
        CHECK(lo_merged == lo_direct, "d-downsize not identical to direct recording");
    }
    std::fprintf(stderr, "downsize tests done\n");
}

// 4. Serialization round-trip.
void test_serialization() {
    for(const auto cfg: {std::pair<unsigned, unsigned>{2, 20}, {1, 9}}) {
        ell::exaloglog_t h(uint8_t(cfg.first), uint8_t(cfg.second), 9);
        DefaultRNGType rng(777);
        for(unsigned i = 0; i < 50000; ++i) h.addh(rng());
        const char *path = "/tmp/ell_roundtrip.ell";
        h.write(path);
        ell::exaloglog_t h2(path);
        CHECK(h == h2, "serialization round-trip mismatch t=%u d=%u", cfg.first, cfg.second);
        CHECK(h2.estimate() == h.estimate(), "estimate changed across serialization");
        std::remove(path);
    }
    std::fprintf(stderr, "serialization tests done\n");
}

// 5b. Regression: the exact cell whose 25-run ensemble mean
// historically exceeded the (statistically incorrect) fixed bound --
// seed set, config, and n pinned. The ensemble mean must stay within
// the corrected true-bias + 3-standard-error bound, and the estimator
// must remain inside its theory RMS.
void test_bias_regression() {
    const unsigned t = 1, d = 9, p = 10;
    const double n = 1e6;
    const double m = double(uint64_t(1) << p);
    const double theory = std::sqrt(ell::exaloglog_t::theoretical_mvp(t, d) / (m * ell::exaloglog_t::regbits(t, d)));
    std::vector<double> errs;
    for(unsigned seed = 1; seed <= 25; ++seed) {
        ell::exaloglog_t h(uint8_t(t), uint8_t(d), p);
        DefaultRNGType rng(seed * 7331 + unsigned(n)); // the historical seed formula
        for(uint64_t i = 0; i < uint64_t(n); ++i) h.addh(rng());
        errs.push_back(h.estimate() / n - 1.);
    }
    const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / double(errs.size()));
    const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / double(errs.size());
    const double bound = 0.1 * theory + 3. * rms / std::sqrt(double(errs.size()));
    std::fprintf(stderr, "bias regression cell: bias=%+.5f bound=%.5f theory=%.5f rms=%.5f\n", bias, bound, theory, rms);
    CHECK(rms <= 1.75 * theory, "regression cell rms %.5f exceeds 1.75x theory %.5f", rms, theory);
    CHECK(std::abs(bias) <= bound, "regression cell bias %.5f exceeds corrected bound %.5f", bias, bound);
    // sensitivity anchor: a true bias of 0.8 standard deviations would
    // produce |mean| ~ 0.8*theory, which this ensemble size rejects
    CHECK(bound < 0.8 * theory, "corrected bound %.5f lost its sensitivity anchor (>= 0.8 theory %.5f)", bound, theory);
}

// 5. Martingale estimation (single-stream).
void test_martingale() {
    const unsigned p = 10;
    const double m = double(uint64_t(1) << p);
    // martingale MVP (paper eq. (6)) for t=2, d=20: (q+d)ln(b)/2 * (1 + b^-d/(b-1))
    const double b = ell::exaloglog_t::base_b(2);
    const double mvp = (26. * std::log(b) / 2.) * (1. + std::pow(b, -20.) / (b - 1.));
    const double theory = std::sqrt(mvp / (m * 28.));
    double worst = 0.;
    for(unsigned seed = 1; seed <= 5; ++seed) {
        ell::exaloglog_t h(2, 20, p);
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
    test_theory();
    test_accuracy();
    test_merge();
    test_downsize();
    test_serialization();
    test_martingale();
    test_bias_regression();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all ExaLogLog tests passed\n");
    return 0;
}
