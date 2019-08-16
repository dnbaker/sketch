#include "vec/vec.h"
#include "mult.h"
#include "hll.h"
#include "hk.h"
#include "ccm.h"
using namespace sketch;
using namespace cws;

using H = HeavyKeeper<16,16>;
using C = cm::ccmbase_t<>;
using S = wj::WeightedSketcher<hll::hll_t, H>;
using S2 = wj::WeightedSketcher<hll::hll_t, C>;
int main () {
    common::DefaultRNGType gen;
    int tbsz = 1 << 16;
    int ntbls = 10;
    size_t nitems = 1 << (18 - 6);
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
    {
        S ws(H(tbsz, ntbls), hll::hll_t(10));
        S ws2(H(tbsz, ntbls), hll::hll_t(10));
        for(size_t i =0; i < nitems; ++i) {
            auto v = gen(), t = gen(), c = t % 64, c2 = (t >> 6) % 64;
            for(size_t j = 0; j < c; ++j)
                ws.addh(v), ws2.addh(v);
            for(size_t j = 0; j < c2; ++j)
                ws.addh(v+1), ws2.addh(v-1);
        }
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "HeavyKeeper:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        std::fprintf(stderr, "HeavyKeeper:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
    {
        int nbits = 8;
        int l2sz = ilog2(tbsz);
        int nhashes = ntbls;
        S2 ws(C(nbits, l2sz, nhashes), hll::hll_t(10));
        S2 ws2(C(nbits, l2sz, nhashes), hll::hll_t(10));
        for(size_t i =0; i < nitems; ++i) {
            auto v = gen(), t = gen(), c = t % 64, c2 = (t >> 6) % 64;
            for(size_t j = 0; j < c; ++j)
                ws.addh(v), ws2.addh(v);
            for(size_t j = 0; j < c2; ++j)
                ws.addh(v+1), ws2.addh(v-1);
        }
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "CM:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        std::fprintf(stderr, "CM:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
}
