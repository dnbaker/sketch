#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "unistd.h"

#include "kthread.h"
#include "libpopcnt/libpopcnt.h"
#include "compact_vector/include/compact_vector.hpp"

#ifndef _VEC_H__
#  define NO_SLEEF
#  define NO_BLAZE
#  include "./vec/vec.h" // Import vec.h, but disable blaze and sleef.
#endif

#if __AES__
#include "aesctr/aesctr.h"
#endif

#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif

#include "sseutil.h"
#include "div.h"
#include "hash.h"
#include "policy.h"
#include "exception.h"
#include "integral.h"


#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#if __CUDACC__ || __GNUC__ || __clang__
#  define SK_RESTRICT __restrict__
#elif _MSC_VER
#  define SK_RESTRICT __restrict
#else
#  define SK_RESTRICT
#endif

#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH 1
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if defined(NDEBUG)
#  if NDEBUG == 0
#    undef NDEBUG
#  endif
#endif

#ifndef FOREVER
#  define FOREVER for(;;)
#endif
#ifndef ASSERT_INT_T
#  define ASSERT_INT_T(T) typename=typename ::std::enable_if<::std::is_integral<(T)>::value>::type
#endif

#if __has_cpp_attribute(no_unique_address)
#  define NO_ADDRESS [[no_unique_address]]
#else
#  define NO_ADDRESS
#endif

#ifndef CONST_IF
#  if __cplusplus >= __cpp_if_constexpr
#    define CONST_IF(...) if constexpr(__VA_ARGS__)
#  else
#    define CONST_IF(...) if(__VA_ARGS__)
#  endif
#endif


#ifndef DBSKETCH_WRITE_STRING_MACROS
#define DBSKETCH_WRITE_STRING_MACROS \
    ssize_t write(const std::string &path, int compression=6) const {return write(path.data(), compression);}\
    ssize_t write(const char *path, int compression=6) const {\
        std::string mode = compression ? std::string("wb") + std::to_string(compression): std::string("wT");\
        gzFile fp = gzopen(path, mode.data());\
        if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at ") + path);\
        auto ret = write(fp);\
        gzclose(fp);\
        return ret;\
    }

#define DBSKETCH_READ_STRING_MACROS \
    ssize_t read(const std::string &path) {return read(path.data());}\
    ssize_t read(const char *path) {\
        gzFile fp = gzopen(path, "rb");\
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);\
        ssize_t ret = read(fp);\
        gzclose(fp);\
        return ret;\
    }
#endif

namespace sketch {
inline namespace common {
using namespace hash;
using namespace integral;

#if __AES__
using DefaultRNGType = aes::AesCtr<uint64_t, 2>;
#else
using DefaultRNGType = std::mt19937_64;
#endif

template<typename BloomType>
inline double jaccard_index(const BloomType &h1, const BloomType &h2) {
    return h1.jaccard_index(h2);
}
template<typename BloomType>
inline double jaccard_index(BloomType &h1, BloomType &h2) {
    return h1.jaccard_index(h2);
}

using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::int64_t;
using std::int32_t;
using std::int16_t;
using std::int8_t;
using std::size_t;
using Space = vec::SIMDTypes<uint64_t>;

static constexpr auto AllocatorAlignment = sse::Alignment::
#if HAS_AVX_512
AVX512
#elif __AVX2__
AVX
#elif __SSE2__
SSE
#else
Normal
#pragma message("Note: no SIMD available, using scalar values")
#endif
     ; // TODO: extend for POWER9 ISA


template<typename ValueType>
using Allocator = sse::AlignedAllocator<ValueType, AllocatorAlignment>;
#ifdef NOT_THREADSAFE
using DefaultCompactVectorType = ::compact::vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;

template<size_t NBITS>
class DefaultStaticCompactVectorType: public ::compact::vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>> {
public:
    DefaultStaticCompactVectorType(size_t nb, size_t nelem): ::compact::vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>>(nelem) {}
};
#else
using DefaultCompactVectorType = ::compact::ts_vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;

template<size_t NBITS>
class DefaultStaticCompactVectorType: public ::compact::ts_vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>> {
public:
    DefaultStaticCompactVectorType(size_t nb, size_t nelem): ::compact::ts_vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>>(nelem) {}
};
#endif


struct identity {
    template<typename T>
    constexpr decltype(auto) operator()(T &&t) const {
        return std::forward<T>(t);
    }
};

template<typename I1, typename I2, typename Func=identity>
double geomean(I1 beg, I2 end, const Func &func=Func()) {
   return std::exp(std::accumulate(beg, end, 0., [&func](auto x, auto y) {return x + std::log(func(y));}) / std::distance(beg, end));
}
template<typename I1, typename I2, typename Func=identity>
double geomean_invprod(I1 beg, I2 end, double num=1., const Func &func=Func()) {
   return std::exp(std::log(num) - std::accumulate(beg, end, 0., [&func](auto x, auto y) {return x + std::log(func(y));}) / std::distance(beg, end));
}
template<typename I1, typename I2, typename Func=identity>
double arithmean(I1 beg, I2 end, const Func &func=Func()) {
   return std::accumulate(beg, end, 0., [&func](auto x, auto y) {return x + func(y);}) / std::distance(beg, end);
}
template<typename I1, typename I2, typename Func=identity>
double arithmean_invsim(I1 beg, I2 end, double num=1., const Func &func=Func()) {
   return std::accumulate(beg, end, 0., [&func,num](auto x, auto y) {return x + num / func(y);}) / std::distance(beg, end);
}


namespace lut {
static const uint8_t nhashesper64bitword [] {
/*
# Auto-generated using:
print("    0xFFu, %s" % ", ".join(str(64 // x) for x in range(1, 63)))
    0xFFu, 64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

*/
    0xFFu, 64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
} // namespace lut



template<typename T> class TD;


namespace sort {
// insertion_sort from https://github.com/orlp/pdqsort
// Slightly modified stylistically.
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
#ifndef SORT_ALGORITHM
template<typename... Args>
void default_sort(Args &&... args) {std::sort(std::forward<Args>(args)...);}
#else
template<typename... Args>
void default_sort(Args &&... args) {SORT_ALGORITHM(std::forward<Args>(args)...);}
#endif

} // namespace sort


struct DoNothing {
    template<typename... Args>void operator()(const Args &&...  args)const{}
    template<typename T>void operator()(const T &x)const{}
};

namespace detail {
// Overloads for setting memory to 0 for either compact vectors
// or std::vectors
template<typename T, typename AllocatorType=typename T::allocator>
static inline void zero_memory(std::vector<T, AllocatorType> &v, size_t newsz) {
    std::memset(v.data(), 0, v.size() * sizeof(v[0]));
    v.resize(newsz);
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::ts_vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}

template<typename T, size_t BUFFER_SIZE=256>
struct tmpbuffer {
    T *const ptr_;
    const size_t n_;
    T buf_[BUFFER_SIZE];
    tmpbuffer(size_t n):
        ptr_(n <= BUFFER_SIZE ? buf_: static_cast<T *>(malloc(n * sizeof(T)))),
        n_(n)
    {}
    T *get() {
        return ptr_;
    }
    const T *get() const {
        return ptr_;
    }
    ~tmpbuffer() {if(n_ > BUFFER_SIZE) std::free(ptr_);}
    auto &operator[](size_t i) {return ptr_[i];}
    const auto &operator[](size_t i) const {return ptr_[i];}
    auto begin() {return ptr_;}
    auto begin() const {return ptr_;}
    auto end() {return ptr_ + n_;}
    auto end() const {return ptr_ + n_;}
};

} // namespace detail

template<typename T>
static constexpr int range_check(unsigned nbits, T val) {
    CONST_IF(std::is_floating_point<T>::value) {
        return 0; // Yeah, we're fine.
    }
    CONST_IF(std::is_signed<T>::value) {
        const int64_t v = val;
        return v < -int64_t(1ull << (nbits - 1)) ? -1
                                                 : v > int64_t((1ull << (nbits - 1)) - 1);
    } else {
        return val >= (1ull << nbits);
    }
}

enum MHCardinalityMode: uint8_t {
    HARMONIC_MEAN,
    GEOMETRIC_MEAN,
    ARITHMETIC_MEAN,
    MEDIAN,
    HLL_METHOD, // Should perform worse than harmonic
};

template<typename T, typename Alloc>
auto &delta_encode(std::vector<T, Alloc> &x) {
    assert(std::is_sorted(x.begin(), x.end()));
    for(auto it = x.begin(), e = x.end(); it != e; ++it) {
        *it = *(it + 1) - *it;
    }
    x.pop_back();
    return x;
}

template<typename T, typename Alloc>
std::vector<T, Alloc> delta_encode(const std::vector<T, Alloc> &x) {
    std::vector<T, Alloc> ret(x);
    delta_encode(ret);
    return ret;
}

#ifdef __CUDACC__
#endif

} // namespace common
} // namespace sketch
