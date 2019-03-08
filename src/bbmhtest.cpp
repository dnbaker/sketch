#include "bbmh.h"
#include "hll.h"
using namespace sketch;
using namespace common;

int main() {
    static_assert(sizeof(schism::Schismatic<int32_t>) == sizeof(schism::Schismatic<uint32_t>), "wrong size!");
    for(size_t i = 7; i < 20; i += 3) {
        for(const auto b: {7u, 13u, 14u, 17u, 9u}) {
            std::fprintf(stderr, "b: %u. i: %zu\n", b, i);
            mh::SuperMinHash<policy::SizePow2Policy> smh(12);
            mh::SuperMinHash<policy::SizeDivPolicy> smh2(1 << 12);
            hll::hll_t h1(i), h2(i);
            mh::BBitMinHasher<uint64_t> b1(i, b), b2(i, b), b3(i, b);
            size_t dbval = 15u << (i - 3);
            mh::DivBBitMinHasher<uint64_t> db1(dbval, b), db2(dbval, b), db3(dbval, b);
            //mh::DivBBitMinHasher<uint64_t> fb(i, b);
            mh::CountingBBitMinHasher<uint64_t, uint32_t> cb1(i, b), cb2(i, b), cb3(i, b);
            DefaultRNGType gen(137);
            size_t shared = 0, b1c = 0, b2c = 0;
            for(size_t i = 5000000; --i;) {
                auto v = gen();
                switch(v & 0x3uL) {
                    case 0:
                    case 1: h1.addh(v); h2.addh(v); b2.addh(v); b1.addh(v); ++shared; b3.addh(v); db1.addh(v); db2.addh(v);/*fb.addh(v);*/ break;
                    case 2: h1.addh(v); b1.addh(v); ++b1c; b3.addh(v); cb3.addh(v); db1.addh(v); break;
                    case 3: h2.addh(v); b2.addh(v); ++b2c; cb1.addh(v); db2.addh(v); break;
                }
            }
            b1.densify();
            b2.densify();
            auto f1 = b1.finalize(), f2 = b2.finalize(), f3 = b3.finalize();
            auto fdb1 = db1.finalize();
            auto fdb2 = db2.finalize();
            //auto fdb3 = db3.finalize();
            std::fprintf(stderr, "Expected Cardinality [shared:%zu/b1:%zu/b2:%zu]\n", shared, b1c, b2c);
            std::fprintf(stderr, "h1 est %lf, h2 est: %lf\n", h1.report(), h2.report());
            std::fprintf(stderr, "Estimate Harmonicard [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(HARMONIC_MEAN), b2.cardinality_estimate(HARMONIC_MEAN));
            std::fprintf(stderr, "Estimate HLL [b1:%lf/b2:%lf/b3:%lf]\n", b1.cardinality_estimate(HLL_METHOD), b2.cardinality_estimate(HLL_METHOD), b3.cardinality_estimate(HLL_METHOD));
            std::fprintf(stderr, "Estimate arithmetic mean [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(ARITHMETIC_MEAN), b2.cardinality_estimate(ARITHMETIC_MEAN));
            std::fprintf(stderr, "Estimate (median) b1:%lf/b2:%lf]\n", b1.cardinality_estimate(MEDIAN), b2.cardinality_estimate(MEDIAN));
            std::fprintf(stderr, "Estimate geometic mean [b1:%lf/b2:%lf]\n", b1.cardinality_estimate(GEOMETRIC_MEAN), b2.cardinality_estimate(GEOMETRIC_MEAN));
            std::fprintf(stderr, "JI for f3 and f2: %lf\n", f1.jaccard_index(f2));
            std::fprintf(stderr, "JI for fdb1 and fdb2: %lf, where nmin = %zu and b = %d\n", fdb2.jaccard_index(fdb1), i, b);
            std::fprintf(stderr, "equal blocks: %zu\n", size_t(f2.equal_bblocks(f3)));
            std::fprintf(stderr, "f1, f2, and f3 cardinalities: %lf, %lf, %lf\n", f1.est_cardinality_, f2.est_cardinality_, f3.est_cardinality_);
            //auto cb13res = cb1.finalize().histogram_sums(cb3.finalize());
            //std::fprintf(stderr, "cb13res %lf, %lf\n", cb13res.weighted_jaccard_index(), cb13res.jaccard_index());
            cb1.finalize().write("ZOMG.cb");
        }
    }
}
