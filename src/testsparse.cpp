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
    hll_t h1(16), h2(16);
    auto rshift = 64 - h1.p(), lshift = h1.p();
    std::map<uint32_t, uint8_t> tmp;
    std::vector<uint32_t> i1, i2;
    for(size_t i = 0; i < 10000; ++i) {
        const auto v = h1.hash(i);
        auto ind = v >> rshift;
        auto val = uint8_t(clz(v << lshift) + 1);
        auto encval = SparseHLL32::encode_value(ind, val);
        assert(SparseHLL32::get_index(SparseHLL32::encode_value(ind, uint8_t(val))) == (ind));
        assert(SparseHLL32::get_value(SparseHLL32::encode_value(ind, uint8_t(val))) == uint8_t(val));
        tmp[ind] = std::max(tmp[ind], uint8_t(val));
        assert(ind == SparseHLL32::get_index(encval));
        assert((encval >> 6) == ind);
        i1.push_back(encval);
        h1.addh(i);
    }
    for(size_t i = 5000; i < 15000; ++i) {
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
    std::vector<uint32_t> i1copy = i1;
    flatten(i1copy);
    auto flattened_vals = flatten_and_query(i1copy, h2);
    std::fprintf(stderr, "Flattened vals from i1: %lf/%lf/%f. ci: %lf. True ci: %lf\n", flattened_vals[0], flattened_vals[1], flattened_vals[2], flattened_vals[2] / (flattened_vals[0] + flattened_vals[2]), h1.containment_index(h2));
    i1copy = i2;
    flattened_vals = flatten_and_query(i2, h2);
    std::fprintf(stderr, "Flattened vals: %lf/%lf/%f. True: %lf\n", flattened_vals[0], flattened_vals[1], flattened_vals[2], h2.report());
    auto vals = pair_query(tmp, h1);
    auto it = tmp.begin();
    size_t ind = 0;
    i1copy = i1;
    flatten(i1copy);
    assert(tmp.size() == i1copy.size());
    assert(std::is_sorted(i1copy.begin(), i1copy.end()));
    i1copy = i1;
    flatten(i1copy);
    SparseHLL<> fil(i1copy);
    assert(fil == SparseHLL<>(h1));
    while(it != tmp.end() && ind < i1copy.size()) {
        //std::fprintf(stderr, "At ind %zu, Oval: %u. our val: %u. Expec %zu in tmp size and %zu in i1copy\n", ind, SparseHLL32::encode_value(it->first, it->second), i1copy[ind], tmp.size(), i1copy.size());
        assert(SparseHLL32::encode_value(it->first, it->second) == i1copy[ind]);
        ++it, ++ind;
    }
    std::fprintf(stderr, "iterator finished ? %s. index into i1copy finished? %s\n", it == tmp.end() ? "true": "false", ind == i1copy.size() ? "true": "NUH UH");
    std::fprintf(stderr, "Flattened vals: %lf/%lf/%fl\n", vals[0], vals[1], vals[2]);
    assert(vals[0] == 0. && vals[1] == 0.);
    hll_t uh = h1 + h2;
    auto us = h2.union_size(h1);
    std::fprintf(stderr, "Size 1: %lf. us from func: %lf\n", uh.report(), us);
    assert(uh.report() == us);
    assert(h.jaccard_index(h2) == 1.);
    assert(h.jaccard_index(h1) == h2.jaccard_index(h1));
}
