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
    CountingRangeMinHash<uint64_t> crhm(200), crhm2(200);
    std::mt19937_64 mt(1337);
    size_t olap_n = (olap_frac * nelem);
    double true_ji = double(olap_n ) / (nelem * 2 - olap_n);
    olap_frac = static_cast<double>(olap_n) / nelem;
    std::fprintf(stderr, "Inserting to both\n");
    crhm.addh(olap_n); crhm2.addh(olap_n >> 1);
    for(size_t i = 0; i < olap_n; ++i) {
        auto v = mt();
        rm1.addh(v); rm2.addh(v);
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
    //mt.seed(1337);
#define MAXMIN \
    std::fprintf(stderr, "Current max: %" PRIu64 "\n", std::max_element(crhm.min().begin(), crhm.min().end(), [](const auto &x, const auto &y) {return x.first < y.first;})->first); \
    std::fprintf(stderr, "Current min: %" PRIu64 "\n", std::min_element(crhm.min().begin(), crhm.min().end(), [](const auto &x, const auto &y) {return x.first < y.first;})->first);
    MAXMIN
    for(size_t i = 0; i < olap_n << 6; ++i) {
        auto v = mt();
        crhm.addh(v);
        crhm.addh(v);
        crhm.addh(v);

        crhm2.addh(v + 2);
        crhm2.addh(v + 2);

        crhm2.addh(v * 5);
        crhm.addh(v * 5);
        for(size_t i = v & 0x7;i--;) {
            crhm.add(v);
        }
    }
    MAXMIN
    auto crfin = crhm.finalize(), crfin2 = crhm2.finalize();
    std::fprintf(stderr, "jaccard/is: %zu/%lf. hist intersect: %lf\n", size_t(crfin.intersection_size(crfin2)), crfin.jaccard_index(crfin2), crfin.histogram_intersection(crfin2));
    assert(crfin.histogram_intersection(crfin2) == crhm.histogram_intersection(crhm2));
}
