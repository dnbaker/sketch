#ifndef ISZ_H__
#define ISZ_H__
#include <cstdint>

namespace sketch {
namespace common {
template<typename Container, typename Cmp=typename Container::key_compare>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    std::uint64_t ret = 0;
    auto it1 = c1.begin();
    auto it2 = c2.begin();
    const auto e1 = c1.end();
    const auto e2 = c2.end();
    if(it1 != e1 && it2 != e2) {
        for(;;) {
            if(cmp(*it1, *it2)) {
                if(++it1 == e1) break;
            } else if(cmp(*it2, *it1)) {
                if(++it2 == e2) break;
            } else {
                ++ret;
                if(++it1 == e1 || ++it2 == e2) break;
            }
        }
    }
    return ret;
}
} // common
} // sketch

#endif
