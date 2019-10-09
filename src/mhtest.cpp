#include "mh.h"
#include <random>

template<typename T>
int pc(const T &x, const char *s="unspecified") {
    auto it = std::begin(x);
    std::string t = std::to_string(*it);
    while(++it != x.end()) {
        t += ',';
        t += std::to_string(*it);
    }
    std::fprintf(stderr, "Container %s contains: %s\n", s, t.data());
    return 0;
}

template<typename T>
bool forward_sorted(T i1, T i2) {
    auto tmp = *i1++;
    int ret = -1;
    start:
    if(tmp == *i1) throw std::runtime_error("e1");
    if(tmp < *i1) {if(ret >= 0 && ret != 1) throw std::runtime_error("e2"); ret = 1;}
    else  {if(ret >= 0 && ret != 0) throw std::runtime_error("e3"); ret = 0;}
    if(++i1 != i2) goto start;
    if(ret < 0) throw std::runtime_error("e4");
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
    CountingRangeMinHash<uint64_t> crmh(1 << ss), crmh2(1 << ss);
    //KthMinHash<uint64_t> kmh(30, 100);
    std::mt19937_64 mt(1337);
    size_t olap_n = (olap_frac * nelem);
    double true_ji = double(olap_n ) / (nelem * 2 - olap_n);
    olap_frac = static_cast<double>(olap_n) / nelem;
    std::fprintf(stderr, "Inserting to both\n");
    crmh.addh(olap_n); crmh2.addh(olap_n >> 1);
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
    size_t is = intersection_size(rm1, rm2, typename RangeMinHash<uint64_t>::key_compare());
    double ji = rm1.jaccard_index(rm2);
    std::fprintf(stderr, "sketch is: %zu. sketch ji: %lf. True: %lf\n", is, ji, true_ji);
    assert(std::abs(ji - true_ji) / true_ji < 0.1);
    is = intersection_size(rm1, rm1, typename RangeMinHash<uint64_t>::key_compare());
    ji = rm1.jaccard_index(rm1);
    std::fprintf(stderr, "ji for a sketch and itself: %lf\n", ji);
    mt.seed(1337);
    for(size_t i = 0; i < nelem; ++i) {
        auto v = mt();
        crmh.addh(v);
        for(size_t i = 0; i < 9; ++i)
            crmh2.addh(v);
        crmh2.addh(v * v);
    }
    std::fprintf(stderr, "is/jaccard: %zu/%lf. hist intersect: %lf.\n", size_t(crmh.intersection_size(crmh2)), crmh.jaccard_index(crmh2), crmh.histogram_intersection(crmh2));
    auto f1 = crmh.cfinalize();
    auto f2 = crmh2.cfinalize();
    std::fprintf(stderr, "finalized\n");
    std::fprintf(stderr, "crmh sum/sumsq: %zu/%zu\n", size_t(crmh.sum()), size_t(crmh.sum_sq()));
    std::fprintf(stderr, "rmh1 b: %zu. rmh1 rb: %zu. max: %zu. min: %zu\n", size_t(*rm1.begin()), size_t(*rm2.rbegin()), size_t(rm1.max_element()), size_t(rm1.min_element()));
    assert(f1.histogram_intersection(f2) == f2.histogram_intersection(f1));
    assert(f1.histogram_intersection(f2) == crmh.histogram_intersection(crmh2));
    assert(crmh.histogram_intersection(crmh2) ==  f1.tf_idf(f2) || !std::fprintf(stderr, "v1: %f. v2: %f\n", crmh.histogram_intersection(crmh2), f1.tf_idf(f2)));
    std::fprintf(stderr, "tf-idf with equal weights: %lf\n", f1.tf_idf(f2));
    std::fprintf(stderr, "f1 est cardinality: %lf\n", f1.cardinality_estimate());
    std::fprintf(stderr, "f1 est cardinality: %lf\n", f1.cardinality_estimate(ARITHMETIC_MEAN));
    std::fprintf(stderr, "f2 est cardinality: %lf\n", f1.cardinality_estimate());
    std::fprintf(stderr, "f2 est cardinality: %lf\n", f1.cardinality_estimate(ARITHMETIC_MEAN));
    auto m1 = rm1.cfinalize(), m2 = rm2.cfinalize();
    std::fprintf(stderr, "m1 is %s-sorted\n", forward_sorted(m1.begin(), m1.end()) ? "forward": "reverse");
    //auto kmf = kmh.finalize();
    std::fprintf(stderr, "jaccard between finalized MH sketches: %lf, card %lf\n", m1.jaccard_index(m2), m1.cardinality_estimate());
    {
        // Part 2: verify merging of final sketches
        size_t n = 10000;
        RangeMinHash<uint64_t> rmh1(20), rmh2(20);
        aes::AesCtr<uint64_t> gen1(13), gen2(1337);
        for(size_t i = n; i--;) {
            rmh1.addh(gen1());
            rmh2.addh(gen2());
        }
        auto fmh1 = rmh1.cfinalize(), fmh2 = rmh2.cfinalize();
        auto u = fmh1 + fmh2;
        RangeMinHash<uint64_t> rmh3(20);
        gen1.seed(13); gen2.seed(1337);
        for(size_t i = n; i--;)
            rmh3.addh(gen1()), rmh3.addh(gen2());
        auto fmh3 = rmh3.cfinalize();
        assert(fmh3.first == u.first || (pc(fmh3, "fmh3") || pc(u, "u")));
    }

}
