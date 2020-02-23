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
    int tbsz = argc == 1 ? 1 << 10: std::atoi(argv[1]);
    tbsz = roundup(tbsz);
    int ntbls = argc <= 2 ? 6: std::atoi(argv[2]);
    size_t nitems = argc <= 3 ? 1 << (20 - 3): std::atoi(argv[3]);
    unsigned ss = 12;
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
    std::vector<uint64_t> data, d1, d2;
    data.reserve(nitems * 16);
    wy::WyRand<uint64_t, 4> rng;
    for(size_t i = 0; i < nitems; ++i) {
        data.insert(data.end(), std::pow(rng() % 8, rng() % 5) / 10, rng());
        auto c = std::pow(rng() % 8, rng() % 5) / 10;
        auto v1 = rng(), v2 = rng();
        d1.insert(d1.end(), c, v1);
        d2.insert(d2.end(), c, v2);
    }
    std::shuffle(d1.begin(), d1.end(), rng);
    std::shuffle(d2.begin(), d2.end(), rng);
    std::shuffle(data.begin(), data.end(), rng);
    {
        S ws(H(tbsz / 2, ntbls), hll::hll_t(ss));
        S ws2(H(tbsz /2, ntbls), hll::hll_t(ss));
        hll::hll_t cmp1(10), cmp2(10);
        for(const auto v: data) {
            ws.addh(v);
            ws2.addh(v);
        }
        for(const auto v: d1) ws.addh(v);
        for(const auto v: d2) ws2.addh(v);
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "HeavyKeeper:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.1);
        std::fprintf(stderr, "HeavyKeeper:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
    {
        S3 ws(14, hll::hll_t(ss));
        S3 ws2(17, hll::hll_t(ss));
        hll::hll_t cmp1(10), cmp2(10);
        for(const auto v: data) {
            ws.addh(v);
            ws2.addh(v);
        }
        for(const auto v: d1) ws.addh(v);
        for(const auto v: d2) ws2.addh(v);
        hll::hll_t v1 = ws.finalize(), v2 = ws2.finalize();
        v1.sum(); v2.sum();
        std::fprintf(stderr, "ExactCounter:v1 wji with v2 %lf\n", v1.jaccard_index(v2));
        assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.03);
        std::fprintf(stderr, "ExactCounter:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
    {
        int nbits = 8;
        int l2sz = ilog2(tbsz);
        std::fprintf(stderr, "l2sz: %d. tbsz: %u\n", l2sz, tbsz);
        int nhashes = ntbls;
        S2 ws(C(nbits, l2sz, nhashes), hll::hll_t(ss));
        S2 ws2(C(nbits, l2sz, nhashes), hll::hll_t(ss));
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
        std::fprintf(stderr, "CM:v1 wji with v2 %lf and true ji %lf\n", v1.jaccard_index(v2), double(shared) / (shared + unshared * 2));
        //assert(std::abs(v1.jaccard_index(v2) - 0.333333) < 0.1);
        assert(std::abs(cmp1.jaccard_index(cmp2) - 0.333333) < 0.05);
        std::fprintf(stderr, "CM:v1.str: %s. ws1 cardinality %lf\n", v1.to_string().data(), ws.sketch_.report());
    }
}
