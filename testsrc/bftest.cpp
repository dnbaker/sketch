#include "bf.h"
#include <iostream>
#include "cbf.h"
#include <unordered_set>
using namespace sketch::bf;

int main(int argc, char *argv[]) {
    bf_t bf1(25, 1, 137), bf2(25, 1, 137);
    cbf_t cbf(8, 25, 1, 137);
    std::unordered_set<uint64_t> s1, s2;
    std::mt19937_64 mt;
    uint64_t val;
    size_t nels = argc > 1 ? std::atoi(argv[1]): 1000000;
    while(s1.size() < nels) s1.insert(mt());
    while(s2.size() < nels) if(s1.find((val = mt())) == std::end(s1)) s2.emplace(val);
    for(const auto &el: s1) bf1.addh(el), cbf.addh(el), bf2.addh(el & 1 ? el: 1337);
    for(const auto &el: s1) assert(bf1.may_contain(el) || !std::fprintf(stderr, "Missing element %" PRIu64"\n", el));
    size_t nfalse(0), ncfalse(0);
    for(const auto &el: s2) nfalse += bf1.may_contain(el);
    for(const auto &el: s2) ncfalse += cbf.may_contain(el);
    std::fprintf(stderr, "Error rate: %lf\n", static_cast<double>(nfalse) / nels);
    std::fprintf(stderr, "Counting error rate: %lf\n", static_cast<double>(ncfalse) / nels);
    uint64_t s1c = 0, s2c = 0, s1f = 0;
    auto bf4(bf1);
    bf4 ^= bf2;
    auto bfxor = bf1 ^ bf2;
    auto bfand = bf1 & bf2;
    for(const auto &el: s1) {
        auto count = cbf.est_count(el);
        s1c += (count > 1);
        s1f += count == 0;
    }
    for(const auto &el: s2) {
        auto count = cbf.est_count(el);
        s2c += (count > 1);
    }
    std::fprintf(stderr, "Counts above 1 for s1: %" PRIu64 ". Counts above 1 for s2: %" PRIu64 ". Counts of zero for s1: %" PRIu64 "\n", s1c, s2c, s1f);
    bf_t bfl(8, 1, 137);
    for(size_t i = 0; i < 100; ++i)
        bfl.addh(i);
    auto bfji = bfl.jaccard_index(bfl);
    std::fprintf(stderr, "jiwith self: %f\n", bfji);
    assert(bfji == 1.);
    auto srs = bfl.template to_sparse_representation<uint32_t>();
    assert(srs.size() == bfl.popcnt());
    for(const auto e: srs) {
        assert(bfl.is_set(e));
    }
    for(unsigned i = 0; i < bfl.size(); ++i) {
        if(bfl.is_set(i))
            assert(std::find(srs.begin(), srs.end(), i) != srs.end());
    }
    sparsebf_t<uint32_t> sbf(bfl);
#if 0
    sketch::common::for_each_delta_decode(srs, [&bfl](size_t x) {
        assert(bfl.is_set(x));});
#endif
}
