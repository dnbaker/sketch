#include "hll.h"
#include "aesctr/aesctr.h"
#include <iostream>
using namespace sketch;
using namespace sketch::common;
using namespace hll;

#if 0
struct XorMultiplyN: public RecursiveReversibleHash<XorMultiply> {
    XorMultiply(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<XorMultiply>(n, seed1) {}
}
struct MultiplyAddN: public RecursiveReversibleHash<MultiplyAdd> {
    MultiplyAdd(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<MultiplyAdd>(n, seed1) {}
}
struct MultiplyAddXorN: public RecursiveReversibleHash<MultiplyAddXor> {
    MultiplyAddXor(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<MultiplyAddXor>(n, seed1) {}
}
#endif

int main() {
    aes::AesCtr<uint64_t> gen(137);
    std::vector<uint64_t> vec(1 << 20);
    hll::hllbase_t<XorMultiplyN<20>> h1(10);
    hll::hllbase_t<XorMultiplyN<1000>> h2(10);
    hll::hllbase_t<XorMultiplyNVec> h3(10, ERTL_MLE, ERTL_JOINT_MLE, 1, false, 10);
    //hll::hllbase_t<XorMultiplyNVec> h2(10, ERTL_MLE, ERTL_JOINT_MLE, 1, false, 1000);
    XorMultiplyN<2> xm;
    //XorMultiplyNVec xm2(1000);
    XorMultiplyN<20> xm2;
    MultiplyAddXoRotN<33, 10> xm3;
    for(auto &v: vec) {
        v = gen(), h1.addh(v), h2.addh(v), h3.addh(v);
        assert(xm.inverse(xm(v)) == v);
        assert(xm2.inverse(xm2(v)) == v);
        //assert(xm3.inverse(xm3(v)) == v);
    }
    std::fprintf(stderr, "Reported sizes (20 mixes: %zu), (1000 mixes:%zu) (10000 %zu). True: %zu\n", size_t(h1.report()), size_t(h2.report()), size_t(h3.report()), vec.size());
    std::fprintf(stderr, "[Warning: MultiplyAddXoRotN is currently failing reversibility test. This needs further attention.\n");
    VType t = Space::set1(1337);
    VType t2 = xm(t);
    t2.for_each([](auto v) {std::fprintf(stderr, "Value is %zu\n", size_t(v));});
}
