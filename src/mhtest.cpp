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

template<typename T>
bool forward_sorted(T i1, T i2) {
    auto tmp = *i1++;
    int ret = -1;
    start:
    if(tmp == *i1) throw "a party";
    if(tmp < *i1) {if(ret >= 0 && ret != 1) throw std::runtime_error("ZOMG"); ret = 1;}
    else  {if(ret >= 0 && ret != 0) throw std::runtime_error("ZOMG"); ret = 0;}
    if(++i1 != i2) goto start;
    if(ret < 0) throw "up";
    return ret;
}
using namespace sketch;
using namespace common;
using namespace mh;

int main(int argc, char *argv[]) {
    KWiseHasherSet<4> zomg(100);
    std::fprintf(stderr, "hv for 133: %zu\n", size_t(zomg(133, 1)));
    size_t nelem = argc == 1 ? 1000000: size_t(std::strtoull(argv[1], nullptr, 10));
    double olap_frac = argc < 3 ? 0.1: std::atof(argv[2]);
    size_t ss = argc < 4 ? 11: size_t(std::strtoull(argv[3], nullptr, 10));
    RangeMinHash<uint64_t> rm1(1 << ss), rm2(1 << ss);
    CountingRangeMinHash<uint64_t> crhm(1 << ss), crhm2(1 << ss);
    //KthMinHash<uint64_t> kmh(30, 100);
    std::mt19937_64 mt(1337);
    size_t olap_n = (olap_frac * nelem);
    double true_ji = double(olap_n ) / (nelem * 2 - olap_n);
    olap_frac = static_cast<double>(olap_n) / nelem;
    std::fprintf(stderr, "Inserting to both\n");
    crhm.addh(olap_n); crhm2.addh(olap_n >> 1);
    std::set<uint64_t> z;
    for(size_t i = 0; i < olap_n; ++i) {
        auto v = mt();
        v = WangHash()(v);
        //kmh.addh(v);
        z.insert(v);
        rm1.add(v); rm2.add(v);
    }
    std::fprintf(stderr, "olap_n: %zu. nelem: %zu\n", olap_n, nelem);
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        z.insert(v);
        rm1.addh(v);
    }
    for(size_t i = nelem - olap_n; i--;) {
        auto v = mt();
        rm2.addh(v);
    }
    size_t is = intersection_size(rm1, rm2);
    double ji = rm1.jaccard_index(rm2);
    std::fprintf(stderr, "sketch is: %zu. sketch ji: %lf. True: %lf\n", is, ji, true_ji);
    assert(std::abs(ji - true_ji) / true_ji < 0.1);
    is = intersection_size(rm1, rm1);
    ji = rm1.jaccard_index(rm1);
    std::fprintf(stderr, "ji for a sketch and itself: %lf\n", ji);
    mt.seed(1337);
    for(size_t i = 0; i < nelem; ++i) {
        auto v = mt();
        crhm.addh(v);
        for(size_t i = 0; i < 9; ++i)
            crhm2.addh(v);
        crhm2.addh(v * v);
    }
    std::fprintf(stderr, "is/jaccard: %zu/%lf. hist intersect: %lf.\n", size_t(crhm.intersection_size(crhm2)), crhm.jaccard_index(crhm2), crhm.histogram_intersection(crhm2));
    auto f1 = crhm.finalize();
    auto f2 = crhm2.finalize();
    std::fprintf(stderr, "rmh1 b: %zu. rmh1 rb: %zu. max: %zu. min: %zu\n", size_t(*rm1.begin()), size_t(*rm2.rbegin()), size_t(rm1.max_element()), size_t(rm1.min_element()));
    assert(f1.histogram_intersection(f2) == f2.histogram_intersection(f1) && f1.histogram_intersection(f2) == crhm.histogram_intersection(crhm2));
    assert(crhm.histogram_intersection(crhm2) ==  f1.tf_idf(f2));
    std::fprintf(stderr, "tf-idf with equal weights: %lf\n", f1.tf_idf(f2));
    std::fprintf(stderr, "f1 est cardinality: %lf\n", f1.cardinality_estimate());
    std::fprintf(stderr, "f1 est cardinality: %lf\n", f1.cardinality_estimate(ARITHMETIC_MEAN));
    std::fprintf(stderr, "f2 est cardinality: %lf\n", f1.cardinality_estimate());
    std::fprintf(stderr, "f2 est cardinality: %lf\n", f1.cardinality_estimate(ARITHMETIC_MEAN));
    auto m1 = rm1.finalize(), m2 = rm2.finalize();
    std::fprintf(stderr, "m1 is %s-sorted\n", forward_sorted(m1.begin(), m1.end()) ? "forward": "reverse");
    //auto kmf = kmh.finalize();
    std::fprintf(stderr, "jaccard between finalized MH sketches: %lf, card %lf\n", m1.jaccard_index(m2), m1.cardinality_estimate());
}
