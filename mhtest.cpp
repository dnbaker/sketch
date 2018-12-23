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
    size_t nelem = argc == 1 ? 10000: size_t(std::strtoull(argv[1], nullptr, 10));
    double olap_frac = argc < 3 ? 0.1: std::atof(argv[2]);
    size_t ss = argc < 4 ? 10: size_t(std::strtoull(argv[3], nullptr, 10));
    RangeMinHash<uint64_t> rm1(1 << ss), rm2(1 << ss);
    CountingRangeMinHash<uint64_t> crhm(200);
    std::mt19937_64 mt(1337);
    size_t olap_n = (olap_frac * nelem);
    double true_ji = double(olap_n ) / (nelem * 2 - olap_n);
    olap_frac = static_cast<double>(olap_n) / nelem;
    std::fprintf(stderr, "Inserting to both\n");
    for(size_t i = 0; i < olap_n; ++i) {
        auto v = mt();
        rm1.addh(v); rm2.addh(v); crhm.addh(v);
    }
    std::fprintf(stderr, "olap_n: %zu. nelem: %zu\n", olap_n, nelem);
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        rm1.addh(v);
    }
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        rm2.addh(v);
    }
    size_t is = intersection_size(rm1, rm2);
    double ji = rm1.jaccard_index(rm2);
    std::fprintf(stderr, "sketch is: %zu. sketch ji: %lf. True: %lf\n", is, ji, true_ji);
    is = intersection_size(rm1, rm1);
    ji = rm1.jaccard_index(rm1);
    std::fprintf(stderr, "ji for a sketch and itself: %lf\n", ji);
    crhm.print();
    mt.seed(1337);
    for(size_t i = 0; i < olap_n; ++i) {
        crhm.addh(mt());
    }
    crhm.print();
}
