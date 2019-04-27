#include "sparse.h"
#include <map>
using namespace sketch;
using namespace hll;
using namespace sparse;
#if !NDEBUG
#else
static_assert(false, "WOO");
#endif

int main() {
    hll_t h1(10), h2(10);
    auto rshift = 64 - h1.p(), lshift = h1.p();
    std::map<uint32_t, uint8_t> tmp;
    std::vector<uint32_t> i1, i2;
    for(size_t i = 0; i < 1000; ++i) {
        const auto v = h1.hash(i);
        auto ind = v >> rshift;
        auto val = uint8_t(clz(v << lshift) + 1);
        auto encval = SparseHLL32::encode_value(ind, val);
        assert(SparseHLL32::get_index(SparseHLL32::encode_value(v >> rshift, uint8_t(clz(v << lshift) + 1))) == (v >> rshift));
        assert(SparseHLL32::get_value(SparseHLL32::encode_value(v >> rshift, uint8_t(clz(v << lshift) + 1))) == uint8_t(clz(v << lshift) + 1));
        tmp[v >> rshift] = std::max(tmp[v >> rshift], uint8_t(clz(v << lshift) + 1));
        i1.push_back(encval);
        h1.addh(i);
    }
    for(size_t i = 500; i < 1500; ++i) {
        const auto v = h2.hash(i);
        auto ind = v >> rshift;
        auto val = uint8_t(clz(v << lshift) + 1);
        auto encval = SparseHLL32::encode_value(ind, val);
        h2.addh(i);
        i2.push_back(encval);
    }
    h1.report();
    h2.report();
    SparseHLL<> h(h2), hh = h;
    SparseHLL<> h1fromv(i1), h2fromv(i2);
    assert(h2fromv == h);
    assert(h.is_sorted());
    assert(hh.is_sorted());
    std::fprintf(stderr, "h JI with h2: %.16lf\n", h.jaccard_index(h2));
    std::fprintf(stderr, "h JI with h1: %.16lf\n", h.jaccard_index(h1));
    std::fprintf(stderr, "h2 JI with h1: %.16lf\n", h2.jaccard_index(h1));
    auto vals = pair_query(tmp, h1);
    assert(vals[0] == 0. && vals[1] == 0.);
    hll_t uh = h1 + h2;
    auto us = h2.union_size(h1);
    std::fprintf(stderr, "Size 1: %lf. us from func: %lf\n", uh.report(), us);
    assert(uh.report() == us);
    assert(h.jaccard_index(h2) == 1.);
    assert(h.jaccard_index(h1) == h2.jaccard_index(h1));
}
