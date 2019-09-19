#ifndef ISZ_H__
#define ISZ_H__
#include <cstdint>

namespace sketch {
namespace common {
template<typename Container, typename Cmp=typename Container::key_compare>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
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
