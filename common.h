#pragma once

#include <algorithm>
#include <array>
#include <atomic>
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

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)
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
#  define ASSERT_INT_T(T) typename=::std::enable_if_t<::std::is_integral_v<(T)>>
#endif

#if __has_cpp_attribute(no_unique_address)
#  define NO_ADDRESS [[no_unique_address]]
#else
#  define NO_ADDRESS
#endif

namespace sketch {
namespace common {
using namespace std::literals;

using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::size_t;
using Space = vec::SIMDTypes<uint64_t>;
using Type  = typename vec::SIMDTypes<uint64_t>::Type;
using VType = typename vec::SIMDTypes<uint64_t>::VType;

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
using DefaultCompactVectorType = ::compact::ts_vector<uint32_t, 0, uint32_t, Allocator<uint32_t>>;
#else
using DefaultCompactVectorType = ::compact::vector<uint32_t, 0, uint32_t, Allocator<uint32_t>>;
#endif

// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It a bijection within [0,1<<64)
// and can be inverted with irving_inv_hash.
struct WangHash {
    INLINE auto operator()(uint64_t key) const {
          key = (~key) + (key << 21); // key = (key << 21) - key - 1;
          key = key ^ (key >> 24);
          key = (key + (key << 3)) + (key << 8); // key * 265
          key = key ^ (key >> 14);
          key = (key + (key << 2)) + (key << 4); // key * 21
          key = key ^ (key >> 28);
          key = key + (key << 31);
          return key;
    }
    INLINE Type operator()(Type element) const {
        VType key = Space::add(Space::slli(element, 21), ~element); // key = (~key) + (key << 21);
        key = Space::srli(key.simd_, 24) ^ key.simd_; //key ^ (key >> 24)
        key = Space::add(Space::add(Space::slli(key.simd_, 3), Space::slli(key.simd_, 8)), key.simd_); // (key + (key << 3)) + (key << 8);
        key = key.simd_ ^ Space::srli(key.simd_, 14);  // key ^ (key >> 14);
        key = Space::add(Space::add(Space::slli(key.simd_, 2), Space::slli(key.simd_, 4)), key.simd_); // (key + (key << 2)) + (key << 4); // key * 21
        key = key.simd_ ^ Space::srli(key.simd_, 28); // key ^ (key >> 28);
        key = Space::add(Space::slli(key.simd_, 31), key.simd_);    // key + (key << 31);
        return key.simd_;
    }
};

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

struct MurFinHash {
    INLINE uint64_t operator()(uint64_t key) const {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccd;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53;
        key ^= key >> 33;
        return key;
    }
    INLINE Type operator()(Type key) const {
        return this->operator()(*(reinterpret_cast<VType *>(&key)));
    }
    INLINE Type operator()(VType key) const {
#if (HAS_AVX_512)
        static const Type mul1 = Space::set1(0xff51afd7ed558ccd);
        static const Type mul2 = Space::set1(0xc4ceb9fe1a85ec53);
#endif

#if !NDEBUG
        auto save = key.arr_[0];
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xff51afd7ed558ccd;});
#  else
        key = Space::mullo(key.simd_, mul1); // h *= 0xff51afd7ed558ccd;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xc4ceb9fe1a85ec53;});
#  else
        key = Space::mullo(key.simd_, mul2); // h *= 0xc4ceb9fe1a85ec53;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        assert(this->operator()(save) == key.arr_[0]);
        return key.simd_;
    }
};
static INLINE uint64_t finalize(uint64_t key) {
    return MurFinHash()(key);
}


template<typename T>
static constexpr inline bool is_pow2(T val) {
    return val && (val & (val - 1)) == 0;
}

template<typename T>
class TD;

inline unsigned popcount(uint64_t val) noexcept {
#ifndef NO_USE_CQF_ASM
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

template<typename T>
INLINE auto popcnt_fn(T val);
template<>
INLINE auto popcnt_fn(Type val) {

#define VAL_AS_ARR(ind) reinterpret_cast<const uint64_t *>(&val)[ind]
#if HAS_AVX_512
#define FUNCTION_CALL popcnt_avx256((Type *)&val, 2)
#elif __AVX2__
// This is supposed to be the fastest option according to the README at https://github.com/kimwalisch/libpopcnt
#define FUNCTION_CALL popcount(VAL_AS_ARR(0)) + popcount(VAL_AS_ARR(1)) + popcount(VAL_AS_ARR(2)) + popcount(VAL_AS_ARR(3))
#elif __SSE2__
#define FUNCTION_CALL popcount(((const uint64_t *)&val)[0]) + popcount(((const uint64_t *)&val)[1])
#else
#  error("Need SSE2. TODO: make this work for non-SIMD architectures")
#endif
    return FUNCTION_CALL;
#undef FUNCTION_CALL
#undef VAL_AS_ARR
}
template<>
INLINE auto popcnt_fn(VType val) {
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
} // namespace sort

} // namespace common
} // namespace sketch
