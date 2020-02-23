#include "sketch/pc.h"
using namespace sketch;
int main() {
    ProbabilisticCounter<uint64_t> pc, pc2;
    PCSA<uint64_t> pcsa(1000);
    wy::WyRand<uint64_t> gen;
    //std::mt19937_64 gen;
    for(size_t i = 0; i < 100000; ++i) {
        auto v = gen();
        pc.addh(v);
        pcsa.addh(v);
    }
    PCSA<uint64_t> pcsa2(1000);
    for(size_t i = 0; i < 100000; ++i) {
        auto v = gen();
        pc2.addh(v);
        pcsa2.addh(v);
    }
    std::fprintf(stderr, "100000: %g, pcsa: %g\n", pc.report(), pcsa.report());
    std::fprintf(stderr, "100000: %g, pcsa: %g\n", pc2.report(), pcsa2.report());
    pcsa2 |= pcsa;
    pc2 |= pc;
    std::fprintf(stderr, "200000 (union): %g, pcsa: %g\n", pc2.report(), pcsa2.report());
}
