#ifndef SKETCH_INTEGRAL_H
#define SKETCH_INTEGRAL_H
#include "./macros.h"
#include <cstring>
#include "sketch/intrinsics.h"
#include <cstdint>
#include <climits>
#include <limits>
#include <cinttypes>
#include "libpopcnt/libpopcnt.h"


#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  elif __CUDACC__
#    define INLINE __forceinline__ inline
#  else
#    define INLINE inline
#  endif
#endif

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__ || __AVX512__)
#endif

#if defined(__AVX512VPOPCNTDQ__)
#  define popcnt512(x) ::_mm512_popcnt_epi64(x)
#endif
namespace sketch {
inline namespace integral {
#if defined(__GNUC__) || defined(__clang__)
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(signed long long x) noexcept {
    return
#ifdef __CUDA_ARCH__
    __clzll(x);
#else
    __builtin_clzll(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(signed long x) noexcept {
    return
#ifdef __CUDA_ARCH__
    __clzll(x);
#else
    __builtin_clzl(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(signed x) noexcept {
    return
#ifdef __CUDA_ARCH__
    __clz(x);
#else
    __builtin_clz(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(signed long long x) noexcept {
#ifdef __CUDA_ARCH__
    return __clzll(__brevll(x));
#else
    return __builtin_ctzl(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(signed long x) noexcept {
#ifdef __CUDA_ARCH__
    return __clzll(__brevll(x));
#else
    return __builtin_ctzl(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(signed x) noexcept {
#ifdef __CUDA_ARCH__
    return __clz(__brev(x));
#else
    return __builtin_ctz(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(signed long long x) noexcept {
#ifdef __CUDA_ARCH__
    return __ffsll(x);
#else
    return __builtin_ffsll(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(signed long x) noexcept {
#ifdef __CUDA_ARCH__
    return __ffsll(x);
#else
    return __builtin_ffsl(x);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(signed x) noexcept {
#ifdef __CUDA_ARCH__
    return __ffs(x);
#else
    return __builtin_ffs(x);
#endif
}


CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(unsigned long long x) noexcept {
    return clz(static_cast<signed long long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(unsigned long x) noexcept {
    return clz(static_cast<signed long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned clz(unsigned x) noexcept {
    return clz(static_cast<signed>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(unsigned long long x) noexcept {
    return ctz(static_cast<signed long long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(unsigned long x) noexcept {
    return ctz(static_cast<signed long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ctz(unsigned x) noexcept {
    return ctz(static_cast<signed>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(unsigned long long x) noexcept {
    return ffs(static_cast<signed long long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(unsigned long x) noexcept {
    return ffs(static_cast<signed long>(x));
}
CUDA_ONLY(__host__ __device__) INLINE HOST_ONLY(constexpr) unsigned ffs(unsigned x) noexcept {
    return ffs(static_cast<signed>(x));
}
#endif
#if !defined(__clang__) && !defined(__GNUC__) && !defined(__CUDACC__)
#pragma message("Using manual clz instead of gcc/clang __builtin_*")
#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
    }} while(0)

INLINE constexpr int clz_manual( uint32_t x ) noexcept
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}

// Overload
INLINE constexpr int clz_manual( uint64_t x ) noexcept
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

#ifndef __CUDACC__
static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz failed.");
static_assert(clz(0x0000000000000003ull) == 62, "64-bit clz failed.");
static_assert(clz(0x0000013333000003ull) == 23, "64-bit clz failed.");
#endif

template<typename T>
static constexpr INLINE unsigned ilog2(T x) noexcept {
    return sizeof(T) * CHAR_BIT - clz(x) - 1;
}
template<typename T>
static constexpr INLINE T roundup(T x) noexcept {
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
template<typename T>
static constexpr INLINE bool is_pow2(T val) noexcept {
    return val && (val & (val - 1)) == 0;
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(unsigned long long val) noexcept {
#ifdef __CUDA_ARCH__
    return __popcll(val);
#else
    return __builtin_popcountll(val);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(unsigned long val) noexcept {
#ifdef __CUDA_ARCH__
    return __popcll(val);
#else
    return __builtin_popcountl(val);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(signed long long val) noexcept {
#ifdef __CUDA_ARCH__
    return __popcll(val);
#else
    return __builtin_popcountll(val);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(signed long val) noexcept {
#ifdef __CUDA_ARCH__
    return __popcll(val);
#else
    return __builtin_popcountl(val);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(unsigned val) noexcept {
#ifdef __CUDA_ARCH__
    return __popc(val);
#else
    return __builtin_popcount(val);
#endif
}
CUDA_ONLY(__host__ __device__) INLINE auto popcount(int val) noexcept {
    return popcount(unsigned(val));
}

#if defined(__MMX__) || defined(__SSE2__) || defined(__SSE4_1__) || defined(__SSE4_2__)
INLINE unsigned popcount(__m64 arg) noexcept {
    uint64_t val;
    std::memcpy(&val, &arg, sizeof(val));
    return popcount(val);
}
#elif !defined(__CUDA_ARCH__)
template<typename T>
INLINE unsigned popcount(const T x) {
    size_t i = 0;
    unsigned ret = 0;
    while(i + 1 <= sizeof(x) / 8) {
        ret += popcount(reinterpret_cast<const uint64_t *>(&x)[i++]);
    }
    while(i + 1 <= sizeof(x) / 4) {
        ret += popcount(reinterpret_cast<const uint32_t *>(&x)[i++]);
    }
    while(i + 1 <= sizeof(x)) {
        ret += popcount(uint32_t(reinterpret_cast<const uint8_t *>(&x)[i++]));
    }
    return ret;
}
#endif

template<typename T>
static INLINE uint64_t vatpos(const T v, size_t ind) noexcept {
    return reinterpret_cast<const uint64_t *>(&v)[ind];
}

template<typename T>
INLINE uint64_t sum_of_u64s(const T val) noexcept {
    if(sizeof(val) < sizeof(uint64_t)) {
        if(sizeof(val) == 8) return *(uint64_t *)&val;
        if(sizeof(val) == 4) return *((uint32_t *)&val);
    }
    uint64_t sum = vatpos(val, 0);
    for(size_t i = 1; i < sizeof(T) / sizeof(uint64_t); ++i)
        sum += vatpos(val, i);
    return sum;
}

#if __AVX2__
template<>
INLINE uint64_t sum_of_u64s<__m256i>(const __m256i x) noexcept {
    __m256i y = (__m256i)(_mm256_permute2f128_pd((__m256d)x, (__m256d)x, 1));
    __m256i m1 = _mm256_add_epi64(x, y);
    __m256i m2 = (__m256i)_mm256_permute_pd((__m256d)m1, 5);
    return _mm256_extract_epi64(_mm256_add_epi64(m1, m2), 0);
}
#endif
#if __SSE2__
template<>
INLINE uint64_t sum_of_u64s<__m128i>(const __m128i val) noexcept {
    return _mm_extract_epi64(val, 0) + _mm_extract_epi64(val, 1);
}
#endif

#ifndef AVX512_REDUCE_OPERATIONS_ENABLED
#  if (defined(__AVX512F__) || defined(__KNCNI__)) && ((defined(__clang__) && __clang_major__ >= 4) \
                                                    || (defined(__GNUC__) && __GNUC__ >= 7))
#    define AVX512_REDUCE_OPERATIONS_ENABLED 1
#  else
#    define AVX512_REDUCE_OPERATIONS_ENABLED 0
#  endif
#endif

#if AVX512_REDUCE_OPERATIONS_ENABLED
template<>
INLINE uint64_t sum_of_u64s<__m512i>(const __m512i val) noexcept {
    return _mm512_reduce_add_epi64(val);
}
#endif

#ifdef __AVX512F__
INLINE __m512i popcnt_fn(__m512i val) noexcept {
#if __AVX512VPOPCNTDQ__ && __AVX512VL__
    return ::_mm512_popcnt_epi64(val);
#else
    return popcnt512(val);
#endif
}
#endif

#ifdef __AVX__
INLINE __m256i popcnt_fn(__m256i val) noexcept {
#if __AVX512VPOPCNTDQ__ && __AVX512VL__
    return ::_mm256_popcnt_epi64(val);
#else
    return popcnt256(val);
#endif
}
#endif

#ifdef __SSE2__
INLINE __m128i popcnt_fn(__m128i val) noexcept {
    return _mm_set_epi64x(popcount(_mm_cvtsi128_si64(val)), popcount(_mm_cvtsi128_si64(_mm_unpackhi_epi64(val, val))));
}
#endif

#ifdef _VEC_H__
INLINE auto popcnt_fn(typename vec::SIMDTypes<uint64_t>::VType val) noexcept {
    return popcnt_fn(val.simd_);
}
#endif
template<typename T, typename T2>
INLINE auto roundupdiv(T x, T2 div) noexcept {
    return ((x + div - 1) / div) * div;
}

static constexpr inline uint64_t bitmask(size_t n) {
    uint64_t x = uint64_t(1) << (n - 1);
    x ^= x - 1;
    return x;
}


} // integral
} // sketch

#endif
