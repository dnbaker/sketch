#include "sketch/pc.h"
using namespace sketch;
int main() {
    ProbabilisticCounter<uint64_t> pc;
    PCSA<uint64_t> pcsa(1000);
    std::mt19937_64 gen;
    for(size_t i = 0; i < 100000; ++i) {
        auto v = gen();
        pc.addh(v);
        pcsa.addh(v);
    }
    PCSA<uint64_t> pcsa2(1000);
    std::fprintf(stderr, "100000: %g, pcsa: %g\n", pc.report(), pcsa.report());
    pcsa2 |= pcsa;
    std::fprintf(stderr, "100000: %g, pcsa: %g\n", pc.report(), pcsa2.report());
}
