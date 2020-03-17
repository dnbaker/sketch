#include "sketch/hll.h"
using namespace sketch;
int main() {
    std::mt19937_64 mt;
    std::vector<hll_t> hlls;
    unsigned start = 10, stop = 16;
    for(unsigned c = start; c <= stop; hlls.emplace_back(c++));
    for(const auto n: {1000000u, 10000000u, 1000u}) {
        mt.seed(n);
        for(auto i = n; i--;) {
            auto v = mt();
            for(auto &h: hlls) h.addh(v);
        }
        auto &core10 = hlls.front().core();
        for(unsigned j = 1; j < hlls.size(); ++j) {
            std::fprintf(stderr, "Comparing p = %d with p = 10 under n = %u items\n", hlls[j].p(), n);
            auto comp1 = hlls[j].compress(10);
            assert(comp1.core().size() == core10.size());
            assert(std::equal(core10.begin(), core10.end(), comp1.core().begin()));
        }
    }
}
