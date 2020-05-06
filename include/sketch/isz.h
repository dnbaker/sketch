#ifndef ISZ_H__
#define ISZ_H__
#include <cstdint>

namespace sketch {
namespace isz {
template<typename Container, typename Cmp=std::less<>>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    //static_assert(std::is_same<decltype(*std::begin(c1)), decltype(*std::begin(c2))>::value, "Containers must derefernce to the same type.");
    //for(const auto v: c1) std::fprintf(stderr, "element is %zu\n", size_t(v));
    assert(std::is_sorted(c2.begin(), c2.end(), cmp));
    assert(std::is_sorted(c1.begin(), c1.end(), cmp));
    auto it1 = std::begin(c1);
    auto it2 = std::begin(c2);
    const auto e1 = std::cend(c1);
    const auto e2 = std::cend(c2);
    if(it1 == e1 || it2 == e2) return 0;
    std::uint64_t ret = 0;
    FOREVER {
        if(*it1 == *it2) { // Easily predicted
            ++ret;
            if(++it1 == e1 || ++it2 == e2) break;
        } else if(cmp(*it1, *it2)) {
            if(++it1 == e1) break;
        } else {
            if(++it2 == e2) break;
        }
    }
    return ret;
}
} // common
} // sketch

#endif
