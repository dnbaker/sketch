#include "bmh.h"
#include "common.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <numeric>
#include <unordered_map>

using namespace sketch;
using wmh::pmh1_t;
using wmh::pmh2_t;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

// Unweighted Jaccard at known overlap: PMH registers agree with probability
// J, so k*J^ ~ Binomial(m, J).
void test_unweighted_jaccard() {
    const double js[] = {0.1, 0.3, 0.5, 0.9};
    const unsigned ms[] = {256, 1024};
    const uint64_t n = 50000;
    for(const unsigned m: ms) {
        for(const double J: js) {
            // A = [0,C) u {2^40+[0,D)}; B = [0,C) u {2^41+[0,E)}; J = C/(C+D+E)
            const uint64_t C = 20000, D = uint64_t(C * (1. - J) / (2. * J) + .5);
            const uint64_t E = D;
            const double true_j = double(C) / double(C + D + E);
            std::vector<double> errs;
            for(unsigned seed = 1; seed <= 15; ++seed) {
                const uint64_t salt = uint64_t(seed) * 0x9E3779B97F4A7C15ULL; // independent hash draws per rep
                pmh2_t<> a(m), b(m);
                for(uint64_t v = 0; v < C; ++v) {a.addh(v ^ salt); b.addh(v ^ salt);}        // shared
                for(uint64_t v = 0; v < D; ++v) a.addh((v + (uint64_t(1) << 40)) ^ salt); // a-only
                for(uint64_t v = 0; v < E; ++v) b.addh((v + (uint64_t(1) << 41)) ^ salt); // b-only
                errs.push_back(a.jaccard_index(b) - true_j);
            }
            const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
            const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
            const double theory = std::sqrt(true_j * (1. - true_j) / m);
            std::fprintf(stderr, "pmh2 m=%4u J=%.1f rms=%.5f (theory %.5f) bias=%+.5f\n", m, J, rms, theory, bias);
            CHECK(rms <= 1.9 * theory + 1e-9, "pmh2 jaccard rms %.5f vs %.5f (m=%u,J=%g)", rms, theory, m, J);
            CHECK(std::abs(bias) <= 0.85 * theory + 1e-9, "pmh2 jaccard bias %.5f (m=%u,J=%g)", bias, m, J);
        }
    }
}

// Weighted Jaccard vs brute force: J_w = sum(min) / sum(max).
void test_weighted_jaccard() {
    const unsigned m = 1024;
    std::vector<double> errs;
    for(unsigned seed = 1; seed <= 10; ++seed) {
        DefaultRNGType rng(seed * 2718);
        std::unordered_map<uint64_t, double> wa, wb;
        for(unsigned t = 0; t < 300; ++t) {
            const uint64_t id = rng() % 400;
            (rng() & 1 ? wa[id] : wb[id]) += 1. + (rng() % 5);
        }
        // shared weights so J_w is nontrivial
        for(unsigned t = 0; t < 200; ++t) {
            const uint64_t id = rng() % 400;
            const double w = 1. + (rng() % 5);
            wa[id] += w;
            wb[id] += w; // consistent weights on shared elements
        }
        double smin = 0., smax = 0.;
        for(uint64_t id = 0; id < 400; ++id) {
            smin += std::min(wa[id], wb[id]);
            smax += std::max(wa[id], wb[id]);
        }
        const double true_j = smin / smax;
        pmh2_t<> a(m), b(m);
        for(const auto &kv: wa) a.update(kv.first, kv.second);
        for(const auto &kv: wb) b.update(kv.first, kv.second);
        errs.push_back(a.jaccard_index(b) - true_j);
        // report() is the exact total weight
        double wa_sum = 0.; for(const auto &kv: wa) wa_sum += kv.second;
        CHECK(std::abs(a.report() - wa_sum) < 1e-6 * wa_sum, "pmh2 report != total weight: %g vs %g", a.report(), wa_sum);
    }
    // register-equality (winner-id) estimation is exact for unweighted
    // sets; for weighted sets it is a coarse approximation of J_w (the
    // unbiased weighted estimator is a likelihood ratio, not ported), so
    // assert a plausible band rather than tight theory bounds.
    const double rms = std::sqrt(std::inner_product(errs.begin(), errs.end(), errs.begin(), 0.) / errs.size());
    const double bias = std::accumulate(errs.begin(), errs.end(), 0.) / errs.size();
    std::fprintf(stderr, "pmh2 weighted jaccard rms=%.5f bias=%+.5f\n", rms, bias);
    CHECK(rms < 0.25, "pmh2 weighted rms %.5f implausible", rms);
    CHECK(std::abs(bias) < 0.2, "pmh2 weighted bias %.5f implausible", bias);
}

void test_merge_and_pmh1() {
    // merge (register-wise min of race values) equals sketching the union
    const unsigned m = 512;
    pmh2_t<> a(m), b(m), both(m);
    DefaultRNGType rng(13);
    for(unsigned i = 0; i < 30000; ++i) {const uint64_t v = rng(); a.addh(v); both.addh(v);}
    for(unsigned i = 0; i < 30000; ++i) {const uint64_t v = rng(); b.addh(v); both.addh(v);}
    a += b;
    // merge preserves the register values exactly (the statistical
    // content); winner ids may differ on exact value ties
    unsigned veq = 0;
    for(unsigned i = 0; i < m; ++i) veq += a.data()[i] == both.data()[i];
    CHECK(veq == m, "pmh2 merged values != union values (%u/%u)", veq, m);
    CHECK(std::abs(a.report() - both.report()) < 1e-6, "pmh2 merged weight != union weight");
    // pmh1_t: jaccard sanity at J = 0.5
    pmh1_t<> p1(m), p2(m);
    const uint64_t salt = 1234567;
    for(uint64_t v = 0; v < 40000; ++v) {p1.addh(v ^ salt); p2.addh(v ^ salt);}
    for(uint64_t v = 40000; v < 80000; ++v) {p1.addh((v + (uint64_t(1) << 40)) ^ salt); p2.addh(v ^ salt);}
    const double j = p1.jaccard_index(p2);
    std::fprintf(stderr, "pmh1 jaccard (true 1/3): %.4f\n", j);
    CHECK(std::abs(j - 1. / 3.) < 0.06, "pmh1 jaccard %g off", j);
    // mismatched sizes throw
    bool threw = false;
    try { pmh2_t<> x(64), y(128); x.jaccard_index(y); } catch(const std::invalid_argument &) { threw = true; }
    CHECK(threw, "pmh2 jaccard accepted mismatched sizes");
}

int main() {
    test_unweighted_jaccard();
    test_weighted_jaccard();
    test_merge_and_pmh1();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all PMH tests passed\n");
    return 0;
}
