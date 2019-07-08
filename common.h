#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "sseutil.h"
#include "math.h"
#include "unistd.h"
#include "x86intrin.h"
#include "kthread.h"
#include  "div.h"
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif
#include "libpopcnt/libpopcnt.h"
#include "compact_vector/include/compact_vector.hpp"
#ifndef _VEC_H__
#  define NO_SLEEF
#  define NO_BLAZE
#  include "vec.h" // Import vec.h, but disable blaze and sleef.
#endif
#if __AES__
#include "aesctr/aesctr.h"
#endif

#include "hash.h"

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__ || __AVX512__)
#endif

#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
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
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);\
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
namespace common {
using namespace hash;
class NotImplementedError: public std::runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): std::runtime_error(std::forward<Args>(args)...) {}

    NotImplementedError(): std::runtime_error("NotImplemented.") {}
};

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
using std::size_t;
using Space = vec::SIMDTypes<uint64_t>;
#if HAS_AVX_512
static_assert(sizeof(Space::VType) == sizeof(__m512i), "Must be the right size");
#endif

template<typename ValueType>
#if HAS_AVX_512
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX512>;
#elif __AVX2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX>;
#elif __SSE2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::SSE>;
#else
using Allocator = std::allocator<ValueType, sse::Alignment::Normal>;
#endif

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

template<typename T>
static INLINE T roundup(T x) noexcept {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    if(sizeof(x) > 1)
        x |= x >> 8;
    if(sizeof(x) > 2)
        x |= x >> 16;
    if(sizeof(x) > 4)
        x |= x >> 32;
    return ++x;
}

#if __GNUC__ || __clang__
constexpr INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}
constexpr INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
constexpr INLINE unsigned clz(unsigned x) {
    return __builtin_clz(x);
}
constexpr INLINE unsigned ctz(unsigned long long x) {
    return __builtin_ctzll(x);
}
constexpr INLINE unsigned ctz(unsigned long x) {
    return __builtin_ctzl(x);
}
constexpr INLINE unsigned ctz(unsigned x) {
    return __builtin_ctz(x);
}
constexpr INLINE unsigned ffs(unsigned long long x) {
    return __builtin_ffsll(x);
}
constexpr INLINE unsigned ffs(unsigned long x) {
    return __builtin_ffsl(x);
}
constexpr INLINE unsigned ffs(unsigned x) {
    return __builtin_ffs(x);
}
#else
#pragma message("Using manual clz instead of gcc/clang __builtin_*")
#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
    }} while(0)

constexpr INLINE int clz_manual( uint32_t x )
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}

// Overload
constexpr INLINE int clz_manual( uint64_t x )
{
  int n(0);
  if ((x & 0xFFFFFFFF00000000ull) == 0) {n  = 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (64 - 4));
  return n;
}

// clz wrappers. Apparently, __builtin_clzll is undefined for values of 0.
// However, by modifying our code to set a 1-bit at the end of the shifted
// region, we can guarantee that this does not happen for our use case.

#define clz(x) clz_manual(x)
// https://en.wikipedia.org/wiki/Find_first_set#CLZ
// Modified for constexpr, added 64-bit overload.
#endif /* #if not __GNUC__ or __clang__ */

static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz failed.");
static_assert(clz(0x0000000000000003ull) == 62, "64-bit clz failed.");
static_assert(clz(0x0000013333000003ull) == 23, "64-bit clz failed.");


template<typename T>
static constexpr INLINE unsigned ilog2(T x) noexcept {
    return sizeof(T) * CHAR_BIT - clz(x)  - 1;
}

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


template<typename T>
static constexpr inline bool is_pow2(T val) {
    return val && (val & (val - 1)) == 0;
}

template<typename T>
class TD;

INLINE auto popcount(uint64_t val) noexcept {
#ifdef AVOID_ASM_POPCNT
// From cqf https://github.com/splatlab/cqf/
    asm("popcnt %[val], %[val]"
            : [val] "+r" (val)
            :
            : "cc");
    return val;
#else
    // According to GodBolt, gcc7.3 fails to inline this function call even at -Ofast.
    //
    //
    return __builtin_popcountll(val);
#endif
}
inline unsigned popcount(__m64 val) noexcept {return popcount(*reinterpret_cast<uint64_t *>(&val));}

template<typename T>
static INLINE uint64_t vatpos(const T v, size_t ind) {
    return reinterpret_cast<const uint64_t *>(&v)[ind];
}

template<typename T>
static INLINE uint64_t sum_of_u64s(const T val) {
    uint64_t sum = vatpos(val, 0);
    for(size_t i = 1; i < sizeof(T) / sizeof(uint64_t); ++i)
        sum += vatpos(val, i);
    return sum;
}
#if defined(__AVX512F__) || defined(__KNCNI__)
#  if (__clang__ &&__clang_major__ >= 4) || (__GNUC__ && __GNUC__ >= 7)
#define AVX512_REDUCE_OPERATIONS_ENABLED 1
template<>
INLINE uint64_t sum_of_u64s<__m512i>(const __m512i val) {
    return _mm512_reduce_add_epi64(val);
}
#  endif
#endif
template<typename T>
INLINE auto popcnt_fn(T val);
template<>
INLINE auto popcnt_fn(typename vec::SIMDTypes<uint64_t>::Type val) {

#define VAL_AS_ARR(ind) reinterpret_cast<const uint64_t *>(&val)[ind]
#if HAS_AVX_512
#  if __AVX512VPOPCNTDQ__
#    define FUNCTION_CALL ::_mm512_popcnt_epi64(val)
#  else
#    define FUNCTION_CALL popcnt512(val)
#  endif
#elif __AVX2__
// This is supposed to be the fastest option according to the README at https://github.com/kimwalisch/libpopcnt
#define FUNCTION_CALL popcnt256(val)
#elif __SSE2__
#define FUNCTION_CALL _mm_set_epi64x(popcount(_mm_cvtsi128_si64(val)), popcount(_mm_cvtsi128_si64(_mm_unpackhi_epi64(val, val))))
#else
#  error("Need SSE2. TODO: make this work for non-SIMD architectures")
#endif
    return FUNCTION_CALL;
#undef FUNCTION_CALL
#undef VAL_AS_ARR
}
template<>
INLINE auto popcnt_fn(typename vec::SIMDTypes<uint64_t>::VType val) {
    return popcnt_fn(val.simd_);
}

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
    std::fprintf(stderr, "New size of container: %zu\n", newsz);
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::ts_vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}

template<typename T>
struct alloca_wrap {
    T *ptr_;
    alloca_wrap(size_t n): ptr_
#if defined(AVOID_ALLOCA)
        (static_cast<T *>(std::malloc(n * sizeof(T))))
#else
        (static_cast<T *>(__builtin_alloca(n * sizeof(T))))
#endif
    {}
    T *get() {
        return ptr_;
    }
    ~alloca_wrap() {
#if defined(AVOID_ALLOCA)
        std::free(ptr_);
#endif
    }
};

} // namespace detail

namespace policy {

template<typename T>
struct SizePow2Policy {
    T mask_;
    SizePow2Policy(size_t n): mask_((1ull << nelem2arg(n)) - 1) {
    }
    static size_t nelem2arg(size_t nelem) {
        return ilog2(roundup(nelem));
    }
    size_t nelem() const {return size_t(mask_) + 1;}
    static size_t arg2vecsize(size_t arg) {return size_t(1) << nelem2arg(arg);}
    T mod(T rv) const {
        return rv & mask_;
    }
};

template<typename T>
struct SizeDivPolicy {
    schism::Schismatic<T> div_;
    static size_t nelem2arg(size_t nelem) {
        return nelem;
    }
    size_t nelem() const {return div_.d_;}
    static size_t arg2vecsize(size_t arg) {return arg;}
    T mod(T rv) const {return div_.mod(rv);}
    SizeDivPolicy(T div): div_(div) {}
};

} // policy

enum MHCardinalityMode: uint8_t {
    HARMONIC_MEAN,
    GEOMETRIC_MEAN,
    ARITHMETIC_MEAN,
    MEDIAN,
    HLL_METHOD, // Should perform worse than harmonic
};

} // namespace common
} // namespace sketch
