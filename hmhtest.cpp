#include "mh.h"
#include <unordered_set>
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

double ji(const std::unordered_set<uint64_t> &a, const std::unordered_set<uint64_t> &b) {
    size_t olap = 0;
    for(const auto v: a) olap += b.find(v) != b.end();
    return double(olap) / (a.size() + b.size() - olap);
}

int main(int argc, char *argv[]) {
    size_t ss = argc < 2 ? 10: size_t(std::strtoull(argv[1], nullptr, 10));
    HyperMinHash<> hmh1(ss, 6), hmh2(ss, 6);
    FinalBBitMinHash<uint64_t> fbhmm(10, 0, 0);
    std::mt19937_64 mt(1337);
    hmh1.addh(uint64_t(13));
    hmh1.addh(uint64_t(17));
    hmh1.addh(uint64_t(21));
    //hmh1.print_all();
    hmh1.clear();
    std::unordered_set<uint64_t> s1, s2;
    for(size_t i = 0; i < 100000; ++i) {
        auto v = mt();
        s1.insert(v);
        hmh1.addh(v);
    }
    for(size_t i = 0; i < 100000; ++i) {
        auto v = mt();
        s2.insert(v);
        hmh2.addh(v);
    }
    for(size_t i = 0; i < 200000; ++i) {
        auto v = mt();
        s1.insert(v);
        s2.insert(v);
        hmh1.addh(v);
        hmh2.addh(v);
    }
    std::fprintf(stderr, "Cardinality estimated (should be 30000) %lf, %lf\n", hmh1.report(), hmh2.report());
    auto hmh3 = hmh1 + hmh2;
    auto hmh3v = hmh3.report();
    auto hmh1v = hmh1.report();
    auto hmh2v = hmh2.report();
    std::fprintf(stderr, "Cardinality estimated (should be 30000) hm1: %lf, hm2: %lf. hm3: %lf. ji: %lf. Exact JI: %lf. Manual JI: %lf\n", hmh1v, hmh2v, hmh3v, hmh1.jaccard_index(hmh2), ji(s1, s2), (hmh1v + hmh2v - hmh3v) / hmh3v);
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
