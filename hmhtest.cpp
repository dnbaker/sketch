#include "mh.h"
#include <random>

template<typename T>
void pc(const T &x, const char *s="unspecified") {
    auto it = std::begin(x);
    std::string t = std::to_string(*it);
    while(++it != x.end()) {
        t += ',';
        t += std::to_string(*it);
    }
    std::fprintf(stderr, "Container %s contains: %s\n", s, t.data());
}

using namespace sketch;
using namespace mh;

int main(int argc, char *argv[]) {
    size_t ss = argc < 2 ? 8: size_t(std::strtoull(argv[1], nullptr, 10));
    HyperMinHash<> hmh1(ss, 10), hmh2(ss, 10);
    std::mt19937_64 mt(1337);
    hmh1.addh(uint64_t(13));
    hmh1.addh(uint64_t(17));
    hmh1.addh(uint64_t(21));
    hmh1.print_all();
    hmh1.clear();
    for(size_t i = 0; i < 10000; ++i)
        hmh2.addh(mt());
    for(size_t i = 0; i < 10000; ++i)
        hmh1.addh(mt());
    for(size_t i = 0; i < 20000; ++i) {
        auto v = mt();
        hmh1.addh(v);
        hmh2.addh(v);
    }
    std::fprintf(stderr, "Cardinality estimated (should be 30000) %lf, %lf\n", hmh1.report(), hmh2.report());
    auto hmh3 = hmh1 + hmh2;
    std::fprintf(stderr, "Cardinality estimated (should be 30000) %lf, %lf. ji: %lf\n", hmh1.report(), hmh3.report(), hmh1.jaccard_index(hmh2));
    hmh1.print_all();
#if 0
    size_t olap_n = (olap_frac * nelem);
    double true_ji = double(olap_n ) / (nelem * 2 - olap_n);
    olap_frac = static_cast<double>(olap_n) / nelem;
    std::fprintf(stderr, "Inserting %zu to one\n", olap_n);
    for(size_t i = 0; i < olap_n; ++i) {
        auto v = mt();
        hmh1.addh(v);
        //hmh2.addh(v);
        //pc(rm1);
    }
    std::fprintf(stderr, "Inserting %zu to set 1\n", olap_n, nelem);
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        hmh1.addh(v);
    }
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        //hmh2.addh(v);
    }
    std::fprintf(stderr, "JI %zu items: %lf.\n", nelem, hmh1.jaccard_index(hmh2));
    std::fprintf(stderr, "JI with itself %lf.\n", hmh1.jaccard_index(hmh2));
    //std::fprintf(stderr, "JI %zu items with itself: %lf.\n", nelem, hmh1.jaccard_index(hmh1));
    std::fprintf(stderr, "Cardinality estimate for %zu items: %lf.\n", nelem, hmh1.report());
    //pc(rm1, "rm1");
    //pc(rm2, "rm2");
    //std::fprintf(stderr, "sketch is: %zu. sketch ji: %lf. True: %lf\n", is, ji, true_ji);
    //std::fprintf(stderr, "ji: %lf\n", ji);
    HyperMinHash<> hmh3(ss, 10);
    hmh3 += hmh1;
    hmh3 += hmh2;
#endif
    //std::fprintf(stderr, "hmh3: %lf\n", hmh3.report());
}
