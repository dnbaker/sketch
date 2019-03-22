#include "bbmh.h"
#include "hll.h"
using namespace sketch;
using namespace common;
using namespace mh;

int main() {
    static_assert(sizeof(schism::Schismatic<int32_t>) == sizeof(schism::Schismatic<uint32_t>), "wrong size!");
    for(size_t i = 7; i < 20; i += 3) {
        for(const auto b: {7u, 13u, 14u, 17u, 9u}) {
            std::fprintf(stderr, "b: %u. i: %zu\n", b, i);
            SuperMinHash<policy::SizePow2Policy> smhp2(1 << i);
            SuperMinHash<policy::SizeDivPolicy>  smhdp(1 << i);
            SuperMinHash<policy::SizePow2Policy> smhp21(1 << i);
            SuperMinHash<policy::SizeDivPolicy>  smhdp1(1 << i);
            hll::hll_t h1(i), h2(i);
            BBitMinHasher<uint64_t> b1(i, b), b2(i, b), b3(i, b);
            size_t dbval = 15u << (i - 3);
            DivBBitMinHasher<uint64_t> db1(dbval, b), db2(dbval, b), db3(dbval, b);
            //DivBBitMinHasher<uint64_t> fb(i, b);
            CountingBBitMinHasher<uint64_t, uint32_t> cb1(i, b), cb2(i, b), cb3(i, b);
            DefaultRNGType gen(137);
            size_t shared = 0, b1c = 0, b2c = 0;
            for(size_t i = 5000000; --i;) {
                auto v = gen();
                switch(v & 0x3uL) {
                    case 0:
                    case 1: h1.addh(v); h2.addh(v);
                            b2.addh(v); b1.addh(v); ++shared;
                            b3.addh(v);
                            db1.addh(v); db2.addh(v);
                            smhp2.addh(v); smhp21.addh(v);
                            smhdp.addh(v); smhdp1.addh(v);
                    /*fb.addh(v);*/
                    break;
                    case 2: h1.addh(v); b1.addh(v); ++b1c; b3.addh(v); cb3.addh(v); db1.addh(v);
                            smhp2.addh(v);
                            smhdp.addh(v);
                    break;
                    case 3: h2.addh(v); b2.addh(v); ++b2c; cb1.addh(v); db2.addh(v);
                            smhdp1.addh(v);
                            smhp21.addh(v);
                    break;
                }
                //if(i % 250000 == 0) std::fprintf(stderr, "%zu iterations left\n", size_t(i));
            }
            b1.densify();
            b2.densify();
            auto f1 = b1.finalize(), f2 = b2.finalize(), f3 = b3.finalize();
            auto fdb1 = db1.finalize();
            auto fdb2 = db2.finalize();
            //std::fprintf(stderr, "About to finalize with %zu for i and %u for b\n", i, b);
            auto smh1 = smhp2.finalize(16), smh2 = smhp21.finalize(16);
            auto smhd1 = smhdp.finalize(16), smhd2 = smhdp1.finalize(16);
            std::fprintf(stderr, "with ss=%zu, smh1 and itself: %lf. 2 and 2/1 jaccard? %lf/%lf\n", size_t(1) << i, double(smh1.jaccard_index(smh1)), double(smh2.jaccard_index(smh1)), smh1.jaccard_index(smh2));
            std::fprintf(stderr, "with ss=%zu, smhd1 and itself: %lf. 2 and 2/1 jaccard? %lf/%lf\n", size_t(1) << i, double(smhd1.jaccard_index(smhd1)), double(smhd2.jaccard_index(smhd1)), smhd1.jaccard_index(smhd2));
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
            //std::fprintf(stderr, "equal blocks: %zu\n", size_t(f2.equal_bblocks(f3)));
            std::fprintf(stderr, "f1, f2, and f3 cardinalities: %lf, %lf, %lf\n", f1.est_cardinality_, f2.est_cardinality_, f3.est_cardinality_);
            auto fcb1 = cb1.finalize(), fcb2 = cb3.finalize();
            //auto cb13res = fcb1.histogram_sums(fcb2);
            //assert(sizeof(cb13res) == sizeof(uint64_t) * 4);
            //std::fprintf(stderr, "cb13res %lf, %lf\n", cb13res.weighted_jaccard_index(), cb13res.jaccard_index());
            cb1.finalize().write("ZOMG.cb");
            auto whl = b1.make_whll();
            std::fprintf(stderr, "whl card: %lf/%zu vs expected %lf/%lf/%lf\n", whl.cardinality_estimate(), whl.core_.size(), f1.est_cardinality_, h1.report(), whl.union_size(whl));
        }
    }
}
