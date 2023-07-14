#define NO_BLAZE
#include "mh.h"
#include "aesctr/wy.h"
using namespace sketch;
using namespace minhash;
template<typename T>
int print(const T &x, const char *s) {
    std::fprintf(stderr, "s: %s\n", s);
    for(const auto v: x) std::fprintf(stderr, ", %zu", size_t(v));
    std::fprintf(stderr, "\n");
    return 0;
}
template<typename T>
void sortfn(T &x) {std::sort(x.begin(), x.end(), typename RangeMinHash<uint64_t>::Compare());}
int main() {
    size_t nelem = 10000, ss = 16;
    RangeMinHash<uint64_t> s1(ss), s2(ss);
    CountingRangeMinHash<uint64_t> cs1(ss), cs2(ss);
    wy::WyHash<> gen(1337);
    for(size_t i = 0; i < nelem; ++i) {
        auto v = gen();
        if(v & 0x1)
            s1.addh(v), cs1.addh(v);
        else if(v & 0x2)
            s2.addh(v), cs2.addh(v);
        else
            s1.addh(v), s2.addh(v), cs1.addh(v), cs2.addh(v);
    }
    auto f1 = s1.finalize(), f2 = s2.finalize();
    RangeMinHash<uint64_t> s3 = s1;
    s3 += s2;
    auto sf3 = s3.finalize();
    std::fprintf(stderr, "us: %g. card: %g. f3 card: %g. s1 card %g. s2 card %g.\n", f1.union_size(f2), sf3.cardinality_estimate(), s3.cardinality_estimate(), s1.cardinality_estimate(), s2.cardinality_estimate());
    assert(f1.union_size(f2) == sf3.cardinality_estimate());
    auto v3 = f1 + f2;
    decltype(f1.first) v1(s1.begin(), s1.end());
    decltype(f1.first) v2(s2.begin(), s2.end());
    auto sf3v = sf3.first;
    sortfn(sf3v);
    assert(f1.first == v1);
    assert(f2.first == v2);
    assert(v3.first == sf3v || print(v3, "v3") || print(sf3v, "sf3"));
}
