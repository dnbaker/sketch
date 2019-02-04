#include "bbmh.h"
#include "hll.h"
#include "aesctr/aesctr.h"
using namespace sketch;
using namespace common;

int main() {
    hll::hll_t h1(12), h2(12);
    mh::BBitMinHasher<uint64_t> b1(4, 12), b2(14, 12);
    aes::AesCtr<uint64_t, 4> gen(137);
    size_t shared = 0, b1c = 0, b2c = 0;
    for(size_t i = 10000000; --i;) {
        auto v = gen();
        switch(v & 0x3uL) {
            case 0:
            case 1: h1.addh(v); h2.addh(v); b2.addh(v); b1.addh(v); ++shared;break;
            case 2: h1.addh(v); b1.addh(v); ++b1c; break;
            case 3: h2.addh(v); b2.addh(v); ++b2c; break;
        }
    }
    b1.densify();
    b2.densify();
    auto f1 = b1.finalize(), f2 = b2.finalize();
    std::fprintf(stderr, "Expected Cardinality [shared:%zu/b1:%zu/b2:%zu]\n", shared, b1c, b2c);
    std::fprintf(stderr, "h1 est %lf, h2 est: %lf\n", h1.report(), h2.report());
    std::fprintf(stderr, "Estimate Harmonicard [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(HARMONIC_MEAN), b2.cardinality_estimate(HARMONIC_MEAN));
    std::fprintf(stderr, "Estimate HLL [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(HLL_METHOD), b2.cardinality_estimate(HLL_METHOD));
    std::fprintf(stderr, "Estimate median [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(MEDIAN), b2.cardinality_estimate(MEDIAN));
    std::fprintf(stderr, "Estimate arithmetic mean [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(ARITHMETIC_MEAN), b2.cardinality_estimate(ARITHMETIC_MEAN));
}
