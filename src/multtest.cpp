#include "vec/vec.h"
#include "mult.h"
#include "hll.h"
using namespace sketch;
using namespace cws;


int main () {
    common::DefaultRNGType gen;
#if VECTOR_WIDTH <= 32 || AVX512_REDUCE_OPERATIONS_ENABLED
    CWSamples<> zomg(100, 1000);
#endif
    realccm_t<> rc(0.999, 10, 20, 8);
    nt::VecCard<uint16_t> vc(13, 10), vc2(13, 10);
    for(size_t i = 0; i < 100000; ++i)
        vc.addh(gen()), vc2.addh(gen());
    auto vc3 = vc + vc2;
    auto zomg2 = vc.report();
    auto zomg3 = vc3.report();
    wj::WeightedSketcher<hll::hll_t, cm::ccm_t> ws(cm::ccm_t(8, 16, 2), hll::hll_t(10));
    wj::WeightedSketcher<hll::hll_t, cm::ccm_t> ws2(cm::ccm_t(4, 16, 2), hll::hll_t(10));
    for(size_t i =0; i < 200; ++i) {
        ws.addh(1);
        auto v = gen(), c = gen() % 8;
        for(size_t j = 0; j < c; ++j)
            ws.addh(v), ws2.addh(v);
        v = gen();
        c = gen() % 4;
        for(size_t j = 0; j < c; ++j)
            ws2.addh(v);
        v = gen();
        for(size_t j = 0; j < c; ++j)
            ws.addh(v);
    }
    hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
    v1.sum(); v2.sum();
    std::fprintf(stderr, "v1 wji with v2 %lf\n", v1.jaccard_index(v2));
    std::fprintf(stderr, "v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
}
