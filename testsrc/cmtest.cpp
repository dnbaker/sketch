#include "mh.h"

using namespace sketch;
struct Counter
{
  struct value_type { template<typename T> value_type(const T&) { } };
  void push_back(const value_type&) { ++count; }
  size_t count = 0;
};
\
int main() {
    mh::CountingRangeMinHash<uint64_t, std::greater<uint64_t>, std::hash<uint64_t>, double> cm(16), cm2(16);
    for(size_t i = 64; i; cm.addh(i--));
    for(size_t i = 96; i > 8; cm2.addh(i--));
    auto cmf2 = cm2.cfinalize();
    auto cmf = cm.cfinalize();
    std::vector<uint64_t> lhv, rhv;
    for(const auto &v: cm) lhv.push_back(v.first);
    for(const auto &v: cm2) rhv.push_back(v.first);
    sort::default_sort(lhv);
    sort::default_sort(rhv);
    Counter c;
    std::set_intersection(lhv.begin(), lhv.end(), rhv.begin(), rhv.end(), std::back_inserter(c));
    assert(c.count == cm.intersection_size(cm2));
    assert(c.count == cmf.intersection_size(cmf2));
    for(auto &c: cmf.second)
        c = 2;
    assert(c.count == cmf.intersection_size(cmf2)); // Make sure intersection size is still the same
    std::fprintf(stderr, "%llu, %llu\n", (unsigned long long)c.count, (unsigned long long)cm.union_size(cm2));
}
