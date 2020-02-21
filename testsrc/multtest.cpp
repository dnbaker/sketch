#include "mult.h"
#include "hll.h"
#include "hk.h"
#include "ccm.h"
using namespace sketch;
using namespace cws;
using PairHasher = XXH3PairHasher;

using H = HeavyKeeper<7,9>;
using C = cm::ccmbase_t<>;
using S = wj::WeightedSketcher<hll::hll_t, H>;
using S2 = wj::WeightedSketcher<hll::hll_t, C>;
using S3 = wj::WeightedSketcher<hll::hll_t, wj::ExactCountingAdapter>;

int main (int argc, char *argv[]) {
    common::DefaultRNGType gen;
    int tbsz = argc == 1 ? 1 << 20: std::atoi(argv[1]);
    int ntbls = argc <= 2 ? 6: std::atoi(argv[2]);
    size_t nitems = 1 << (20 - 6);
#if !defined(NO_BLAZE) && (VECTOR_WIDTH <= 32 || AVX512_REDUCE_OPERATIONS_ENABLED)
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
        S ws(H(tbsz / 2, ntbls), hll::hll_t(10));
        S ws2(H(tbsz /2, ntbls), hll::hll_t(10));
        hll::hll_t cmp1(10), cmp2(10);
        gen.seed(0);
        for(size_t i =0; i < nitems; ++i) {
            auto v = gen(), t = gen(), c = t % 16, c2 = (t >> 6) % 16;
            for(unsigned i = 0; i < c; ++i) {
                ws.addh(v);
                ws2.addh(v);
            }
            uint64_t v1 = v;
            for(unsigned i = 0; i < c2; ++i) {
                wy::wyhash64_stateless(&v1);
                ws.addh(v1);
                wy::wyhash64_stateless(&v1);
                ws2.addh(v1);
            }
        }
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "HeavyKeeper:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.03);
        std::fprintf(stderr, "HeavyKeeper:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
    {
        S3 ws(H(tbsz / 2, ntbls), hll::hll_t(10));
        S3 ws2(H(tbsz /2, ntbls), hll::hll_t(10));
        hll::hll_t cmp1(10), cmp2(10);
        gen.seed(0);
        for(size_t i =0; i < nitems; ++i) {
            auto v = gen(), t = gen(), c = t % 16, c2 = (t >> 6) % 16;
            for(unsigned i = 0; i < c; ++i) {
                ws.addh(v);
                ws2.addh(v);
            }
            uint64_t v1 = v;
            for(unsigned i = 0; i < c2; ++i) {
                wy::wyhash64_stateless(&v1);
                ws.addh(v1);
                wy::wyhash64_stateless(&v1);
                ws2.addh(v1);
            }
        }
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "ExactCounter:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.03);
        std::fprintf(stderr, "ExactCounter:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
    {
        int nbits = 32;
        int l2sz = ilog2(tbsz);
        int nhashes = ntbls;
        S2 ws(C(nbits, l2sz, nhashes), hll::hll_t(10));
        S2 ws2(C(nbits, l2sz, nhashes), hll::hll_t(10));
        hll::hll_t cmp1(10), cmp2(10);
        size_t shared = 0, unshared = 0;
        gen.seed(0);
        for(size_t i =0; i < nitems; ++i) {
            auto v = gen(), t = gen(), c = t % 16, c2 = (t >> 6) % 16;
            shared += c;
            for(size_t j = 0; j < c; ++j) {
                auto hv = PairHasher()(v, j);
                ws.addh(v), ws2.addh(v), cmp1.add(hv), cmp2.add(hv);
            }
            unshared += c2;
            for(size_t j = 0; j < c2; ++j) {
                uint64_t v1 = v, v2;
                wy::wyhash64_stateless(&v1);
                v2 = v1;
                wy::wyhash64_stateless(&v2);
                ws.addh(v1), ws2.addh(v2);
                auto hv1 = PairHasher()(v1, j);
                auto hv2 = PairHasher()(v2, j);
                cmp1.add(hv1), cmp2.add(hv2);
            }
        }
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "WJ without HK or CM by HLL: %lf\n", cmp1.jaccard_index(cmp2));
        assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.03);
        assert(std::abs(cmp1.jaccard_index(cmp2) - 0.333333) < 0.05);
        std::fprintf(stderr, "CM:v1 wji with v2 %lf and true ji %lf\n", v1.jaccard_index(v2), double(shared) / (shared + unshared * 2));
        std::fprintf(stderr, "CM:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
}
