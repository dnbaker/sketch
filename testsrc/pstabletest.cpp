#include "pstable.h"
#include "rnla.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <numeric>
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

template<typename F>
std::vector<double> sample(F &f, size_t n, uint64_t seed) {
    DefaultRNGType rng(seed);
    std::vector<double> v(n);
    for(auto &x: v) x = double(f(rng));
    return v;
}

// a=2: Gaussian with variance 2 (Chambers-Mallows-Stuck special case),
// regardless of skewness b.
void test_gaussian() {
    for(const double b: {0., -0.7, 0.9}) {
        pstable::cms_distribution<double> gen(2., b);
        const auto v = sample(gen, 100000, 42);
        const double mean = std::accumulate(v.begin(), v.end(), 0.) / v.size();
        const double var = std::inner_product(v.begin(), v.end(), v.begin(), 0.) / v.size() - mean * mean;
        std::fprintf(stderr, "a=2 b=%.1f: mean=%+.5f var=%.5f\n", b, mean, var);
        CHECK(std::abs(mean) < 5. * std::sqrt(2. / v.size()), "a=2 b=%g mean %f off", b, mean);
        CHECK(std::abs(var - 2.) < 0.1, "a=2 b=%g variance %f != 2", b, var);
    }
}

// a=1, b=0: standard Cauchy. Tail P(|X| > x) ~ 2/(pi x); median 0.
void test_cauchy() {
    pstable::cms_distribution<double> gen(1., 0.);
    const size_t n = 4000000;
    const auto v = sample(gen, n, 137);
    std::vector<double> w(v);
    std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
    const double median = w[w.size() / 2];
    CHECK(std::abs(median) < 0.01, "Cauchy median %f != 0", median);
    for(const double x: {100., 1000., 4000.}) {
        const size_t cnt = std::count_if(v.begin(), v.end(), [x](double y) {return std::abs(y) > x;});
        const double empirical = double(cnt) / n;
        const double theory = 2. / (M_PI * x);
        std::fprintf(stderr, "Cauchy tail x=%g: empirical %.3e theory %.3e (ratio %.3f)\n",
                     x, empirical, theory, empirical / theory);
        CHECK(std::abs(empirical / theory - 1.) < 0.15, "Cauchy tail at %g off: %f vs %f", x, empirical, theory);
        // exact density sanity: f(0) = 1/pi
    }
    // density at 0: fraction in [-h, h] ~ 2h/pi
    const size_t cnt0 = std::count_if(v.begin(), v.end(), [](double y) {return std::abs(y) < 0.001;});
    const double dens = double(cnt0) / n / 0.002;
    CHECK(std::abs(dens - 1. / M_PI) < 0.03, "Cauchy density at 0: %f != %f", dens, 1. / M_PI);
}

// a=0.5, b=0: symmetric stable with index 0.5. Tail index check:
// log(count(x1)/count(x2)) / log(x2/x1) ~ a.
void test_stable_tail() {
    pstable::cms_distribution<double> gen(0.5, 0.);
    const size_t n = 4000000;
    const auto v = sample(gen, n, 999);
    const double x1 = 1e4, x2 = 1e6;
    const size_t c1 = std::count_if(v.begin(), v.end(), [x1](double y) {return std::abs(y) > x1;});
    const size_t c2 = std::count_if(v.begin(), v.end(), [x2](double y) {return std::abs(y) > x2;});
    const double slope = std::log(double(c1) / c2) / std::log(x2 / x1);
    std::fprintf(stderr, "a=0.5 tail slope: %.3f (expect ~0.5, counts %zu/%zu)\n", slope, c1, c2);
    CHECK(std::abs(slope - 0.5) < 0.1, "stable tail slope %f != 0.5", slope);
    CHECK(c1 > 1000 && c2 > 10, "insufficient tail samples: %zu %zu", c1, c2);
}

// Skewness: b>0 gives a heavy right tail in the Chambers-Mallows-Stuck
// convention (same parameterization as scipy.stats.levy_stable, which
// also shifts the median negative for a in (1, 2)). Verify by tail
// asymmetry around the median rather than the median sign.
void test_skew() {
    for(const double b: {1., -1.}) {
        pstable::cms_distribution<double> gen(1.2, b);
        const auto v = sample(gen, 400000, 7);
        std::vector<double> w(v);
        std::sort(w.begin(), w.end());
        const double med = w[w.size() / 2];
        const double qlo = w[w.size() / 1000], qhi = w[w.size() - w.size() / 1000];
        const double left = med - qlo, right = qhi - med;
        std::fprintf(stderr, "a=1.2 b=%+.1f med=%+.4f left-1%%=%.3f right-1%%=%.3f\n", b, med, left, right);
        // heavy tail must be on the side of the skew
        CHECK(b * (right - left) > 0., "skew direction wrong: b=%g left=%f right=%f", b, left, right);
        CHECK(std::max(left, right) / std::min(left, right) > 3., "tails not asymmetric enough: %f %f", left, right);
    }
}

// clipped Cauchy from this header, and the thresholded (rejection) Cauchy
// from rnla.h: bounded, symmetric, Cauchy-like in the bulk.
void test_cauchy_variants() {
    pstable::clipped_cauchy_distribution<double> clipped(500.);
    {
        const auto v = sample(clipped, 200000, 5);
        CHECK(std::all_of(v.begin(), v.end(), [](double y) {return std::abs(y) <= 500.;}), "clip violated");
        const double mean = std::accumulate(v.begin(), v.end(), 0.) / v.size();
        CHECK(std::abs(mean) < 0.5, "clipped cauchy mean %f", mean);
        // bulk density matches Cauchy near 0
        const size_t cnt0 = std::count_if(v.begin(), v.end(), [](double y) {return std::abs(y) < 0.01;});
        const double dens = double(cnt0) / v.size() / 0.02;
        CHECK(std::abs(dens - 1. / M_PI) < 0.03, "clipped density at 0: %f", dens);
    }
    rnla::thresholded_cauchy_distribution<double> thresh(500.);
    {
        const auto v = sample(thresh, 100000, 6);
        CHECK(std::all_of(v.begin(), v.end(), [](double y) {return std::abs(y) <= 500.;}), "threshold violated");
        std::vector<double> w(v);
        std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
        CHECK(std::abs(w[w.size() / 2]) < 0.05, "thresholded cauchy median %f", w[w.size() / 2]);
    }
    // argument validation
    bool threw = false;
    try { pstable::cms_distribution<double> bad(3.); } catch(const std::out_of_range &) { threw = true; }
    CHECK(threw, "cms_distribution accepted a=3");
    threw = false;
    try { pstable::cms_distribution<double> bad(1., 2.); } catch(const std::out_of_range &) { threw = true; }
    CHECK(threw, "cms_distribution accepted b=2");
    std::fprintf(stderr, "cauchy variants done\n");
}

int main() {
    test_gaussian();
    test_cauchy();
    test_stable_tail();
    test_skew();
    test_cauchy_variants();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all p-stable sampler tests passed\n");
    return 0;
}
