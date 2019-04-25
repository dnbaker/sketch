#include "sparse.h"
#include <map>
using namespace sketch;
using namespace hll;
using namespace sparse;

int main() {
    hll_t h1(10), h2(10);
    std::map<uint32_t, uint8_t> tmp;
    for(size_t i = 0; i < 1000; ++i) {
        auto v = h1.hash(i);
        tmp[v >> 54] = std::max(tmp[v >> 54], uint8_t(clz(v << 10) + 1));
        h1.addh(i);
    }
    for(size_t i = 500; i < 1500; ++i) {
        h2.addh(i);
    }
    h1.report();
    h2.report();
    SparseHLL<> h(h2), hh = h;
    std::fprintf(stderr, "h JI with h2: %lf\n", h.jaccard_index(h2));
    std::fprintf(stderr, "h JI with h1: %lf\n", h.jaccard_index(h1));
    auto vals = pair_query(tmp, h1);
    std::fprintf(stderr, "%lf/%lf/%lf for the trio which should be a perfect match\n", vals[0], vals[1], vals[2]);
    hll_t uh = h1 + h2;
    auto us = h2.union_size(h1);
    std::fprintf(stderr, "Size 1: %lf. us from func: %lf\n", uh.report(), us);
    assert(uh.report() == us);
    std::fprintf(stderr, "h2 JI with h1: %lf\n", h2.jaccard_index(h1));
    assert(h.jaccard_index(h2) == 1.);
    assert(h.jaccard_index(h1) == h2.jaccard_index(h1));
}
