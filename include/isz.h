#ifndef ISZ_H__
#define ISZ_H__
#include <cstdint>

namespace sketch {
namespace common {
template<typename Container1, typename Container2, typename Cmp=typename Container1::key_compare>
std::uint64_t intersection_size(const Container1 &c1, const Container2 &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    static_assert(std::is_same<decltype(*std::begin(c1)), decltype(*std::begin(c2))>::value, "Containers must derefernce to the same type.");
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
