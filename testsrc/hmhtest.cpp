#include "sketch/hmh.h"
#include <iostream>

int main() {
    const char *hm1p = "__hm1.hmh", *hm2p = "__hm2.hmh";
    size_t nelem = 1000000;
    for(const auto hms: {8, 10, 12}) {
        std::mt19937_64 rng(hms);
        for(const auto rem: {8, 16, 32, 64}) {
            double hle = 0., hme = 0.;
            double jhle = 0., jhme = 0., cjhme = 0.;
            for(size_t inum = 0; inum < 4; ++inum) {
                sketch::HyperMinHash hm(hms, rem), hm2(hms, rem), hmh4(hms, rem), hmh8(hms, rem);
                size_t hlls = hms + sketch::ilog2(rem) - 3;
                sketch::hll_t hl(hlls), hl2(hlls), hl4(hlls), hl8(hlls);
                auto seed = rng();
                for(size_t i = 0; i < nelem; ++i) {
                    hm.addh(seed + i);
                    hm2.addh(seed + i);
                    hm2.addh(~(seed + i));
                    hmh4.addh(seed + i);
                    hmh4.addh(~(seed + i));
                    hmh4.addh(sketch::hash::WangHash::hash(~(seed + i)));
                    hmh4.addh(sketch::hash::WangHash::hash(seed + i));
                    hl.addh(seed + i);
                    hl2.addh(seed + i);
                    hl2.addh(~(seed + i));
                    hl4.addh(seed + i);
                    hl4.addh(~(seed + i));
                    hl4.addh(sketch::hash::WangHash::hash(~(seed + i)));
                    hl4.addh(sketch::hash::WangHash::hash(seed + i));
                    hmh8.addh(seed + i);
                    hl8.addh(seed + i);
                    wy::WyRand<uint64_t> rng(seed + i);
                    for(size_t i = 7; i--;) {auto r = rng();hl8.addh(r); hmh8.addh(r);}
                }
                // True JI should be 50%.
                double ce = hm.cardinality_estimate();
                //std::cerr << ce << " vs " << nelem << " for %" << std::abs(ce - nelem) / nelem * 100 << " error.\n";
                //std::cerr << rem << "HLL est: " << hm.estimate_hll_portion() << '\n';
                //std::cerr << rem << "MH est: " << hm.estimate_mh_portion() << '\n';
                hme += std::abs(ce - nelem);
                hle += std::abs(hl.cardinality_estimate() - nelem);
                //double h2 = hm2.estimate_hll_portion(), mh2 = hm2.estimate_mh_portion();
                //std::cerr << rem << "2HLL est: " << h2 << '\n';
                //std::cerr << rem << "2MH est: " << mh2 << '\n';
                double ji = hm.jaccard_index(hm2);
                std::cerr << ji << '\n';
                std::cerr << "JI via HLL: " << hl.jaccard_index(hl2) << '\n';
                hm.write(hm1p);
                hm2.write(hm2p);
                assert(hm == sketch::HyperMinHash(hm1p));
                assert(hm2 == sketch::HyperMinHash(hm2p));
                double ji4 = hm.jaccard_index(hmh4);
                //std::fprintf(stderr, "JI for hm and hmh4: %g. (expected 25%% (1/4))\n", ji4);
                //std::fprintf(stderr, "JI for hll and hll4: %g. (expected 25%% (1/4))\n", hl.jaccard_index(hl4));
                jhle += std::abs(hl.jaccard_index(hl4) - .25); jhme += std::abs(ji4 - .25);
                cjhme += std::abs(hm.card_ji(hmh4) - .25);
            }
            std::fprintf(stderr, "[%d:%d] ji error hll: %g. ji error hmh: %g. HMH via card: %g\n", hms, rem, jhle, jhme, cjhme);
            std::fprintf(stderr, "[%d:%d] card error hll: %g. card error hmh: %g\n", hms, rem, hle, hme);
        }
    }
}
