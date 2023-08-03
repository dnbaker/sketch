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
    size_t nelem = 1000000, ss = 1024;
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
    VERBOSE_ONLY(std::fprintf(stderr, "cad est before adding: %g\n", s3.cardinality_estimate());)
    s3 += s2;
    VERBOSE_ONLY(std::fprintf(stderr, "cad est after adding: %g\n", s3.cardinality_estimate());)
    auto f3 = s3.finalize();
#ifndef VERBOSE_AF
    std::fprintf(stderr, "s1: %g. s2: %g. s3: %g\n", s1.cardinality_estimate(), s2.cardinality_estimate(), s3.cardinality_estimate());
    std::fprintf(stderr, "f1: %g. f2: %g. f3: %g\n", f1.cardinality_estimate(), f2.cardinality_estimate(), f3.cardinality_estimate());
    std::fprintf(stderr, "card: %g. f3 card: %g. s1 card %g. s2 card %g.\n", f3.cardinality_estimate(), s3.cardinality_estimate(), s1.cardinality_estimate(), s2.cardinality_estimate());
#endif
}
