#include "bf.h"
#include "cbf.h"
#include <unordered_set>
using namespace sketch::bf;

int main(int argc, char *argv[]) {
    bf_t bf(25, 1, 137);
    cbf_t cbf(8, 25, 1, 137);
    std::unordered_set<uint64_t> s1, s2;
    std::mt19937_64 mt;
    uint64_t val;
    size_t nels = argc > 1 ? std::atoi(argv[1]): 1000000;
    while(s1.size() < nels) s1.insert(mt());
    while(s2.size() < nels) if(s1.find((val = mt())) == std::end(s1)) s2.emplace(val);
    for(const auto &el: s1) bf.addh(el), cbf.addh(el);
    for(const auto &el: s1) assert(bf.may_contain(el) || !std::fprintf(stderr, "Missing element %" PRIu64"\n", el));
    size_t nfalse(0), ncfalse(0);
    for(const auto &el: s2) nfalse += bf.may_contain(el);
    for(const auto &el: s2) ncfalse += cbf.may_contain(el);
    std::fprintf(stderr, "Error rate: %lf\n", static_cast<double>(nfalse) / nels);
    std::fprintf(stderr, "Counting error rate: %lf\n", static_cast<double>(ncfalse) / nels);
    uint64_t s1c = 0, s2c = 0, s1f = 0;
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
}
