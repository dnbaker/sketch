#include "sketch/bmh.h"
#include <chrono>
using namespace sketch;

using namespace sketch::wmh;

#if 0
+static inline long double compute_truncexp_lambda(size_t m) {
+    return std::log1p(1.L / (m - size_t(1)));
+}
+static inline std::array<long double, 4> compute_truncexp_constants(size_t m) {
+    const auto lambda = compute_truncexp_lambda(m);
+    const auto c1 = static_cast<long double>((std::exp(lambda) - 1.L) / lambda);
+    const long double c2 = std::log(2.L / (1.L + std::exp(-lambda))) / lambda;
+    const long double c3 = (1.L - std::exp(-lambda)) / lambda;
+    return std::array<long double, 4>{lambda, c1, c2, c3};
+}
+
+template<typename FT, typename F>
+static INLINE FT expsample(uint64_t rngstate, const FT lambda, const FT c1, const FT c2, const FT c3, F &func) {
+    // Func should update the x and return the random value sampled
+    FT x = func(rngstate) * c1;
+    if(x < static_cast<FT>(1))
+        return x;
+    for(;;) {
+        if((x = func(rngstate)) < c2) break;
+        FT yhat = static_cast<FT>(0.5) * func(rngstate);
+        if(yhat > static_cast<FT>(1) - x) {
+            x = 1. - x;
+            yhat = static_cast<FT>(1) - yhat;
+        }
+        if(x <= c3 * (static_cast<FT>(1) - yhat) || (yhat * c1 <= static_cast<FT>(1) - x)) break;
+        if(yhat * c1 * lambda <= std::exp(lambda * (static_cast<FT>(1) - x)) - static_cast<FT>(1)) break;
+    }
+}
#endif
auto gett() {return std::chrono::high_resolution_clock::now();}
template<typename T>
auto timepassed(T x, T y) {return std::chrono::duration<double, std::milli>(y - x).count();}
template<typename T>
double avg(T &x) {size_t nitems = 0;long double sum = 0.; for(auto &i: x) sum += i, ++nitems; return sum / nitems;}

int main(int argc, char *argv[]) {
    using FT = double;
    static_assert(sizeof(FT) >= 8, "Must use double or larger.");
    const size_t m = (argc <= 1) ? size_t(1333337): size_t(std::strtoull(argv[1], nullptr, 10));
    const size_t nh = (argc <= 2) ? size_t(133333700): size_t(std::strtoull(argv[1], nullptr, 10));
    const auto constants = compute_truncexp_constants<double>(m);
    std::vector<uint64_t> sources(nh);
    std::mt19937_64 mt;
    for(auto &x: sources) x = mt();
    std::vector<double> hashes(nh);
    auto t1 = gett();
    for(size_t i = 0; i < nh; ++i) {
        hashes[i] = -std::log1p(std::fma(static_cast<double>(sources[i]), 0x1p-64, -1.));
    }
    auto t2 = gett();
    auto time2expsample = timepassed(t1, t2);
    t1 = gett();
    std::fprintf(stderr, "fma log1p: %gms. Average value: %g\n", time2expsample, avg(hashes));
    for(size_t i = 0; i < nh; ++i) {
        hashes[i] = -std::log(sources[i] * 0x1p-64);
    }
    t2 = gett();
    time2expsample = timepassed(t1, t2);
    std::fprintf(stderr, "normal log: %gms. Average value: %g\n", time2expsample, avg(hashes));
    t1 = gett();
    std::transform(sources.begin(), sources.end(), hashes.begin(), [](auto x) {return -std::log1p(std::fma(x, 0x1p-64, -1.));});
    t2 = gett();
    time2expsample = timepassed(t1, t2);
    std::fprintf(stderr, "std::transform fma log1p: %gms. Average value: %g\n", time2expsample, avg(hashes));
    t1 = gett();
    std::transform(sources.begin(), sources.end(), hashes.begin(), [](auto x) {return -std::log(x * 0x1p-64);});
    t2 = gett();
    time2expsample = timepassed(t1, t2);
    std::fprintf(stderr, "std::transform normal log: %gms. Average value: %g\n", time2expsample, avg(hashes));
    auto t3 = gett();
    for(size_t i = 0; i < nh; ++i) {
        hashes[i] = truncexpsample(sources[i], constants);
    }
    auto t4 = gett();
    std::fprintf(stderr, "truncated special sample: %gms. Average value: %g\n", timepassed(t3, t4), avg(hashes));
    t3 = gett();
    std::transform(sources.begin(), sources.end(), hashes.begin(), [&constants](auto x) {return truncexpsample(x, constants);});
    t4 = gett();
    std::fprintf(stderr, "std::transform truncated special sample: %gms. Average value: %g\n", timepassed(t3, t4), avg(hashes));
}
