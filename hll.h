#ifndef HLL_H_
#define HLL_H_
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cinttypes>
#include <algorithm>
#include <string>
#include <vector>
#include "logutil.h"
#include "sseutil.h"
#define XSTR(x) STR(x)
#define STR(x) #x

#ifndef INLINE
#  if __GNUC__ || __clang__
#  define INLINE __attribute__((always_inline)) inline
#  else
#  define INLINE inline
#  endif
#endif

#include "x86intrin.h"
#define HAS_AVX_512 _FEATURE_AVX512F

namespace hll {

// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It has a 1-1 mapping from any one 64-bit integer to another
// and can be inverted with irving_inv_hash.
INLINE uint64_t wang_hash(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

static INLINE uint64_t roundup64(std::size_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return ++x;
}


// clz wrappers. Apparently, __builtin_clzll is undefined for values of 0.
// For our hash function, there is only 1 64-bit integer value which causes this problem.
// I'd expect that this is acceptable. And on Haswell+, this value is the correct value.
#if __GNUC__ || __clang__
#ifndef NAVOID_CLZ_UNDEF
constexpr INLINE unsigned clz(unsigned long long x) {
    return x ? __builtin_clzll(x) : sizeof(x) * CHAR_BIT;
}

constexpr INLINE unsigned clz(unsigned long x) {
    return x ? __builtin_clzl(x) : sizeof(x) * CHAR_BIT;
}
#else
constexpr INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}

constexpr INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
#endif
#else
// https://en.wikipedia.org/wiki/Find_first_set#CLZ
// Modified for constexpr, added 64-bit overload.
#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
        default:                        x += 0; break;\
    }} while(0)

constexpr INLINE int clz( uint32_t x )
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}
// Overload
constexpr INLINE int clz( uint64_t x )
{
  int n(0);
  if ((x & 0xFFFFFFFF00000000ull) == 0) {n  = 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (64 - 4));
  return n;
}
#endif

static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz hand-rolled failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz hand-rolled failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz hand-rolled failed.");
static_assert(clz(0x0000000000000000ull) == 64, "64-bit clz hand-rolled failed.");


constexpr double make_alpha(std::size_t m) {
    switch(m) {
        case 16: return .673;
        case 32: return .697;
        case 64: return .709;
        default: return 0.7213 / (1 + 1.079/m);
    }
}

class hll_t {
// HyperLogLog implementation.
// To make it general, the actual point of entry is a 64-bit integer hash function.
// Therefore, you have to perform a hash function to convert various types into a suitable query.
// We could also cut our memory requirements by switching to only using 6 bits per element,
// (up to 64 leading zeros), though the gains would be relatively small
// given how memory-efficient this structure is.

// Attributes
    std::size_t np_;
    const std::size_t m_;
    double alpha_;
    double relative_error_;
#if HAS_AVX_512
#if !NDEBUG
#pragma message("Building with avx512")
#endif
    std::vector<std::uint8_t, sse::AlignedAllocator<std::uint8_t, sse::Alignment::AVX512>> core_;
#elif __AVX2__
#if !NDEBUG
#pragma message("Building with avx2")
#endif
    std::vector<std::uint8_t, sse::AlignedAllocator<std::uint8_t, sse::Alignment::AVX>> core_;
#elif __SSE2__
#if !NDEBUG
#pragma message("Building with sse2")
#endif
    std::vector<std::uint8_t, sse::AlignedAllocator<std::uint8_t, sse::Alignment::SSE>> core_;
#else
    std::vector<std::uint8_t> core_;
#endif
    double sum_;
    int is_calculated_;

public:
    // Constructor
    hll_t(std::size_t np=20): np_(np), m_(1uL << np), alpha_(make_alpha(m_)),
                         relative_error_(1.03896 / std::sqrt(m_)),
                         core_(m_, 0),
                         sum_(0.), is_calculated_(0) {}

    // Call sum to recalculate if you have changed contents.
    void sum();

    // Returns cardinality estimate. Sums if not calculated yet.
    double creport() const; // creport doesn't calculate
    double report();

    // Returns error estimate
    double est_err();
    double cest_err() const;

    INLINE void add(uint64_t hashval) {
        const uint32_t index(hashval >> (64ull - np_));
        const uint32_t lzt(clz(hashval << np_) + 1);
        if(core_[index] < lzt) core_[index] = lzt;
    }

    INLINE void addh(uint64_t element) {add(wang_hash(element));}

    std::string desc_string() const {
        return std::string("Size: ") + std::to_string(np_) + ". Number of bits: " + std::to_string(m_)
               + ". Relative error: " + std::to_string(relative_error_)
               + ". Is calculated? " + (is_calculated_ ? "true.": "false.") + " sum: " + std::to_string(sum_)
               + ".";
    }

    std::string to_string() const {
        return is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                              : desc_string();
    }

    // Reset.
    void clear();
    hll_t(const hll_t&) = default;
    hll_t(hll_t&&) = default;
    hll_t& operator=(const hll_t&) = default;
    hll_t& operator=(hll_t&&) = default;

    hll_t const &operator+=(const hll_t &other);
    hll_t const &operator&=(const hll_t &other);

    // Clears, allows reuse with different np.
    void resize(std::size_t new_size);
    // Getter for is_calculated_
    bool is_ready() const {
        return is_calculated_;
    }
    std::size_t get_np() const {return np_;}
};

// Returns the size of a symmetric set difference.
double operator^(hll_t &first, hll_t &other);
// Returns the size of the set intersection
double operator&(hll_t &first, hll_t &other);
// Returns a HyperLogLog union
hll_t operator+(const hll_t &one, const hll_t &other);



} // namespace hll

#endif // #ifndef HLL_H_
