#include "sketch/hmh.h"
#include <iostream>

int main() {
    const char *hm1p = "__hm1.hmh", *hm2p = "__hm2.hmh";
    size_t nelem = 10000000;
    for(const auto rem: {8, 16, 64, 32}) {
        sketch::HyperMinHash hm(12, rem), hm2(12, rem);
        size_t seed = std::mt19937_64(0)();
        for(size_t i = 0; i < nelem; ++i) {
            hm.addh(seed + i);
            hm2.addh(seed + i);
            hm2.addh(~(seed + i));
        }
        // True JI should be 50%.
        double ce = hm.cardinality_estimate();
        std::cerr << ce << " vs " << nelem << " for %" << std::abs(ce - nelem) / nelem * 100 << " error.\n";
        std::cerr << rem << "HLL est: " << hm.estimate_hll_portion() << '\n';
        std::cerr << rem << "MH est: " << hm.estimate_mh_portion() << '\n';
        double h2 = hm2.estimate_hll_portion(), mh2 = hm2.estimate_mh_portion();
        std::cerr << rem << "2HLL est: " << h2 << '\n';
        std::cerr << rem << "2MH est: " << mh2 << '\n';
        double ji = hm.jaccard_index(hm2);
        std::cerr << ji << '\n';
        hm.write(hm1p);
        hm2.write(hm2p);
        assert(hm == sketch::HyperMinHash(hm1p));
        assert(hm2 == sketch::HyperMinHash(hm2p));
    }
}
