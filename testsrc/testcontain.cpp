#define NO_BLAZE
#include "hll.h"
#include "bbmh.h"
#include "aesctr/wy.h"

using namespace sketch::hll;
using namespace sketch::minhash;

int main(int argc, char *argv[]) {
    size_t seed = 13;
    size_t base_size = 1500000, shared_size = 200000;
    wy::WyHash<uint64_t, 8> wygen(seed);
    std::vector<uint64_t> ao(base_size), bo(base_size), so(shared_size);
    for(auto x: {&ao, &bo, &so})
        for(auto &y: *x)
            y = wygen();
    const size_t ss = 14;
    hll_t ha(ss), hb(ss), hi(ss);
    BBitMinHasher<uint64_t> ph1(ss - 1, 16), ph2(ss - 1, 16);
    for(const auto v: ao) {
        ha.addh(v);
        ph1.addh(v);
    }
    for(const auto v: bo) {
        hb.addh(v);
        ph2.addh(v);
    }
    for(const auto v: so) {
        ha.addh(v);
        hb.addh(v);
        ph1.addh(v);
        ph2.addh(v);
        hi.addh(v);
    }
    auto fh1 = ph1.finalize(), fh2 = ph2.finalize();
    assert(ha.containment_index(ha) == 1.);
    assert(fh1.containment_index(fh1) == 1.);
    assert(fh2.containment_index(fh2) == 1.);
    assert(std::abs(hi.containment_index(ha) - 1.) <= 1e-5 || !std::fprintf(stderr, "contain: %g vs expected %g\n", hi.containment_index(ha), 1.));
    assert(std::abs(hi.containment_index(hb) - 1.) < 1e-5);
    auto full_cmps_hll = ha.full_set_comparison(hb),
         full_cmps_bbmh = fh1.full_set_comparison(fh2);
    double expected_ci = double(shared_size) / (shared_size + base_size);
    double ci = ha.containment_index(hb);
    double fhci = fh1.containment_index(fh2);
    //std::fprintf(stderr, "Expected: %lf\n", expected_ci);
    //std::fprintf(stderr, "full cmps hll CI: %lf\n", full_cmps_hll[2] / (full_cmps_hll[0] + full_cmps_hll[2]));
    //std::fprintf(stderr, "full cmps bbmh CI: %lf, manual %lf\n", full_cmps_bbmh[2] / (full_cmps_bbmh[0] + full_cmps_bbmh[2]), fhci);
    assert(full_cmps_bbmh[2] / (full_cmps_bbmh[0] + full_cmps_bbmh[2]) / fhci - 1. < 1e-40);
    assert(full_cmps_hll[2] / (full_cmps_hll[0] + full_cmps_hll[2]) / ci - 1. < 1e-40);
    //std::fprintf(stderr, "CI with bbmh: %lf\n", fhci);
    //std::fprintf(stderr, "CI with hll: %lf\n", ci);
    //std::fprintf(stderr, "CI expected: %lf\n", expected_ci);
    // std::fprintf(stderr, "%% error with bbmh: %lf\n", std::abs(fhci / expected_ci - 1.) * 100);
    // std::fprintf(stderr, "%% error with hll: %lf\n", std::abs(ci / expected_ci - 1.) * 100);
    assert(std::abs(ci / expected_ci - 1.) < 0.06);
    assert(std::abs(fhci / expected_ci - 1.) < 0.06);
    // == shared_size / (shared_size + base_siae)
    for(size_t imb = base_size; --imb;ha.addh(wygen()));
    hll_t u(ha + hb);
    ha.sum(); hb.sum(); hi.sum(), u.sum();
    ci = ha.containment_index(u);
    //std::fprintf(stderr, "ci: %lf. a.contain(b)? %lf. vice versa: %lf\n", ci, ha.containment_index(hb), hb.containment_index(ha));
}
