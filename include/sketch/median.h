#ifndef SK_MEDIAN_H
#define SK_MEDIAN_H
#include <algorithm>
#include "macros.h"

namespace sketch {
inline namespace med {

template<typename T>
INLINE constexpr T median3(T x, T y, T z) {
    using std::max;
    using std::min;
    return max(min(z, x), min(max(z, x), y));
}

template<typename C, typename T=typename C::value_type>
INLINE constexpr T median3(const C &c) {
    return median3(c[0], c[1], c[2]);
}

template<typename T>
INLINE constexpr T median5(T x, T y, T z, T a, T b) {
    using std::max;
    using std::min;
    return median3(max(min(a, y), min(z, x)),
                   min(max(a, y), max(z, x)),
                   b);
}
template<typename C, typename T=typename C::value_type>
INLINE constexpr T median5(const C &c) {
    return median5(c[0], c[1], c[2], c[3], c[4]);
}

namespace detail {
template<class Iter, class Compare>
inline void insertion_sort(Iter begin, Iter end, Compare comp) {
    using T = typename std::iterator_traits<Iter>::value_type;

    for (Iter cur = begin + 1; cur < end; ++cur) {
        Iter sift = cur;
        Iter sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already positioned correctly.
        if (comp(*sift, *sift_1)) {
            T tmp = std::move(*sift);

            do { *sift-- = std::move(*sift_1); }
            while (sift != begin && comp(tmp, *--sift_1));

            *sift = std::move(tmp);
        }
    }
}
template<class Iter>
inline void insertion_sort(Iter begin, Iter end) {
    insertion_sort(begin, end, std::less<std::decay_t<decltype(*begin)>>());
}
} // detail
template<typename T>
INLINE T median(T *v, size_t n) {
    static_assert(std::is_arithmetic<T>::value, "must be arithmetic");
    switch(n) {
        case 1: return v[0];
        case 2: return (v[0] + v[1]) / 2;
        case 3: return median3(v[0], v[1], v[2]);
        case 5: return median5(v[0], v[1], v[2], v[3], v[4]);
    }
    if(n < 50)
        detail::insertion_sort(v, v + n);
    else
#ifdef PDQSORT_H
        pdqsort(v, v + n);
#else
        std::sort(v, v + n);
#endif
    T ret;
    if(n&1) ret = v[n / 2];
    else    ret = (v[n / 2] + v[(n - 1) / 2]) / 2;
    return ret;
}

} //inline namespace med
} // sketch
#endif
