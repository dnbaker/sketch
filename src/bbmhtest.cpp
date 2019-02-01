#include "bbmh.h"
#include "aesctr/aesctr.h"
using namespace sketch;
using namespace common;

int main() {
    mh::BBitMinHasher<uint64_t> b1(4, 8), b2(4, 8);
    aes::AesCtr<uint64_t, 4> gen(137);
    size_t shared = 0, b1c = 0, b2c = 0;
    for(size_t i = 1000000; --i;) {
        auto v = gen();
        switch(v & 0x3uL) {
            case 0:
            case 1: b2.addh(v); b1.addh(v); ++shared;break;
            case 2: b1.addh(v); ++b1c; break;
            case 3: b2.addh(v); ++b2c; break;
        }
    }
    auto f1 = b1.finalize(), f2 = b2.finalize();
    std::fprintf(stderr, "Expected Cardinality [shared:%zu/b1:%zu/b2:%zu]\n", shared, b1c, b2c);
    std::fprintf(stderr, "Estimate Harmonicard [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(), b2.cardinality_estimate());
    std::fprintf(stderr, "Estimate Harmonicard [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(HLL_METHOD), b2.cardinality_estimate(HLL_METHOD));
}
