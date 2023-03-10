#ifndef _VEC_H__
#define _VEC_H__
#define NOSVML
#if __has_include("sleef.h")
#    include "sleef.h"
#else
#    define NO_SLEEF
#endif
#include "sketch/intrinsics.h"
#include <cmath>
#include <iterator>
#include <type_traits>
#include <cstdint>
#include <array>
#ifndef NO_BLAZE
#include "blaze/Math.h"
#endif

#ifdef __aarch64__

#ifndef VEC_DISABLED__
#define VEC_DISABLED__
#endif

#endif

#ifndef VEC_FALLTHROUGH
#  if __cplusplus >= __has_cpp_attribute(fallthrough)
#    define VEC_FALLTHROUGH [[fallthrough]];
#  elif defined(__GNUC__) || defined(__clang__)
#    define VEC_FALLTHROUGH __attribute__((fallthrough));
#  else
#    define VEC_FALLTHROUGH ;
#  endif
#endif

#ifndef CONST_IF
#  if defined(__cpp_if_constexpr) && __cplusplus >= __cpp_if_constexpr
#    define CONST_IF(...) if constexpr(__VA_ARGS__)
#  else
#    define CONST_IF(...) if(__VA_ARGS__)
#  endif
#endif


#ifndef IS_BLAZE
#  define IS_BLAZE(x) (::blaze::IsVector<x>::value || ::blaze::IsMatrix<x>::value)
#endif
#ifndef IS_CONTIGUOUS_BLAZE
#  define IS_CONTIGUOUS_BLAZE(x) (bool(::blaze::TransposeFlag<x>::value))
#endif
#ifndef IS_COMPRESSED_BLAZE
#  define IS_COMPRESSED_BLAZE(x) (::blaze::IsSparseVector<x>::value || ::blaze::IsSparseMatrix<x>::value)
#endif
#ifndef IS_CONTIGUOUS_UNCOMPRESSED_BLAZE
#  define IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(x) (IS_BLAZE(x) && !IS_COMPRESSED_BLAZE(x) && IS_CONTIGUOUS_BLAZE(x))
#endif
#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)
#endif

#ifndef VECTOR_WIDTH
#  if HAS_AVX_512
#    define VECTOR_WIDTH 64u
#  elif __AVX2__
#    define VECTOR_WIDTH 32u
#  elif __SSE2__
#    define VECTOR_WIDTH 16u
#  else
#    define VECTOR_WIDTH 8u
#  endif
#endif

#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif




namespace vec {

#ifndef VEC_DISABLED__
using std::uint64_t;
#ifndef NO_SLEEF
namespace scalar {
    using namespace std;
    __attribute__((const)) Sleef_double2 sincos(double x) {
        return Sleef_double2{sin(x), cos(x)};
    }
    __attribute__((const)) Sleef_float2 sincos(float x) {
        return Sleef_float2{sin(x), cos(x)};
    }
    template<typename T> auto sqrt_u35(T val) {return sqrt(val);}
    template<typename T> auto sqrt_u05(T val) {return sqrt(val);}
    template<typename T> auto frfrexp(T val) {return Sleef_frfrexp(val);}
}
#endif // #ifndef NO_SLEEF

// Modified from Agner Fog's vectorclass library
#if !(defined(__AVX512DQ__) && defined(__AVX512VL__))
#  if defined ( __SSE4_1__ )
INLINE __m128i _mm_mullo_epi64(__m128i a, __m128i b) {
	//
    // instruction does not exist. Split into 32-bit multiplies
    __m128i bswap   = _mm_shuffle_epi32(b,0xB1);           // b0H,b0L,b1H,b1L (swap H<->L)
    __m128i prodlh  = _mm_mullo_epi32(a,bswap);            // a0Lb0H,a0Hb0L,a1Lb1H,a1Hb1L, 32 bit L*H products
    __m128i zero    = _mm_setzero_si128();                 // 0
    __m128i prodlh2 = _mm_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m128i prodlh3 = _mm_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m128i prodll  = _mm_mul_epu32(a,b);                  // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m128i prod    = _mm_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
}
INLINE __m128i _mm_mullo_epi64x(__m128i a, uint64_t b)
{
    return _mm_mullo_epi64(a, _mm_set1_epi64x(b));
}
#  endif // SSE4.1
#  if __AVX2__
INLINE __m256i _mm256_mullo_epi64 (__m256i a, __m256i b) {
    __m256i bswap   = _mm256_shuffle_epi32(b,0xB1);           // swap H<->L
    __m256i prodlh  = _mm256_mullo_epi32(a,bswap);            // 32 bit L*H products
    __m256i zero    = _mm256_setzero_si256();                 // 0
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m256i prodll  = _mm256_mul_epu32(a,b);                  // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m256i prod    = _mm256_add_epi64(prodll,prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
}
INLINE __m256i _mm256_mullo_epi64x(__m256i a, uint64_t b)
{
    return _mm256_mullo_epi64(a, _mm256_set1_epi64x(b));
}
  #endif // __AVX2__
#else
using ::_mm_mullo_epi64;
using ::_mm256_mullo_epi64;
INLINE __m128i _mm_mullo_epi64x(__m128i a, uint64_t b)
{
    return _mm_mullo_epi64(a, _mm_set1_epi64x(b));
}
INLINE __m256i _mm256_mullo_epi64x(__m256i a, uint64_t b)
{
    return _mm256_mullo_epi64(a, _mm256_set1_epi64x(b));
}
#endif // Don't have AVX512{DQ,VL}

template<typename ValueType>
struct SIMDTypes;

#define OP(op, suf, sz) _mm##sz##_##op##_##suf
#define decop(op, suf, sz) static constexpr decltype(&OP(op, suf, sz)) op = &OP(op, suf, sz);

/* Use or separately because it's a keyword.*/

#if __AVX512F__
#define DECMAX64
#else
    #define DECMAX64 \
    static INLINE typename SIMDTypes<ValueType>::Type max(typename SIMDTypes<ValueType>::Type ret, typename SIMDTypes<ValueType>::Type rhs) { \
        for(unsigned i = 0; i < sizeof(Type) / sizeof(uint64_t); ++i) \
            ((uint64_t *)&ret)[i] = std::max(((uint64_t *)&ret)[i], ((uint64_t *)&rhs)[i]); \
        return ret; \
    } \
    static INLINE typename SIMDTypes<ValueType>::Type min(typename SIMDTypes<ValueType>::Type ret, typename SIMDTypes<ValueType>::Type rhs) { \
        for(unsigned i = 0; i < sizeof(Type) / sizeof(uint64_t); ++i) \
            ((uint64_t *)&ret)[i] = std::min(((uint64_t *)&ret)[i], ((uint64_t *)&rhs)[i]); \
        return ret; \
    }
#endif

#define declare_avx512_cmpeq_mask(sz) \
    static constexpr decltype(&_mm512_cmpeq_epi##sz##_mask) cmpeq_mask = &_mm512_cmpeq_epi##sz##_mask; \
    static constexpr decltype(&_mm512_cmpneq_epi##sz##_mask) cmpneq_mask = &_mm512_cmpneq_epi##sz##_mask;

#define declare_all(suf, sz) \
   decop(loadu, suf, sz) \
   decop(storeu, suf, sz) \
   decop(load, suf, sz) \
   decop(store, suf, sz) \
   static constexpr decltype(&OP(or, suf, sz)) or_fn = &OP(or, suf, sz);\
   static constexpr decltype(&OP(and, suf, sz)) and_fn = &OP(and, suf, sz);\
   decop(add, suf, sz) \
   decop(sub, suf, sz) \
   decop(mul, suf, sz) \
   decop(set1, suf, sz) \
   /*decop(setr, suf, sz) */\
   decop(set, suf, sz) \
   decop(mask_and, suf, sz) \
   decop(maskz_and, suf, sz) \
   decop(maskz_andnot, suf, sz) \
   decop(mask_andnot, suf, sz) \
   decop(andnot, suf, sz) \
   /*decop(blendv, suf, sz) */

#define declare_int_ls(suf, sz) \
    decop(loadu, si##sz, sz) \
    decop(load, si##sz, sz) \
    decop(storeu, si##sz, sz) \
    decop(store, si##sz, sz)

#define declare_int_ls128(suf, sz) \
    decop(loadu, si128, sz) \
    decop(load, si128, sz) \
    decop(storeu, si128, sz) \
    decop(store, si128, sz)

#define declare_int_epi32(sz) \
    decop(slli, epi32, sz) \
    decop(srli, epi32, sz) \
    decop(add, epi32, sz) \
    decop(sub, epi32, sz) \
    decop(min, epu32, sz) \
    decop(max, epu32, sz) \
    decop(mullo, epi32, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(mullo, epi32, sz)) mul = &OP(mullo, epi32, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi32, sz) \
    decop(cmpeq, epi32, sz)

#define declare_int_epi32_512(sz) \
    decop(slli, epi32, sz) \
    decop(srli, epi32, sz) \
    decop(add, epi32, sz) \
    decop(sub, epi32, sz) \
    decop(min, epu32, sz) \
    decop(max, epu32, sz) \
    decop(mullo, epi32, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi32, sz) \
    declare_avx512_cmpeq_mask(32)

#define declare_int_epi32_128(sz) \
    decop(slli, epi32, sz) \
    decop(srli, epi32, sz) \
    decop(add, epi32, sz) \
    decop(sub, epi32, sz) \
    decop(min, epu32, sz) \
    decop(max, epu32, sz) \
    static constexpr decltype(&_mm_mullo_epi32) mul = &_mm_mullo_epi32;\
    static constexpr decltype(&_mm_mullo_epi32) mullo = &_mm_mullo_epi32;\
    static constexpr decltype(&OP(xor, si128, sz)) xor_fn = &OP(xor, si128, sz);\
    static constexpr decltype(&OP(and, si128, sz)) and_fn = &OP(and, si128, sz);\
    static constexpr decltype(&OP(or, si128, sz))  or_fn = &OP(or, si128, sz);\
    decop(set1, epi32, sz) \
    decop(cmpeq, epi32, sz)

#define declare_int_epi8(sz) \
    decop(add, epi8, sz) \
    decop(sub, epi8, sz) \
    decop(min, epu8, sz) \
    decop(max, epu8, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi8, sz) \
    decop(cmpeq, epi8, sz) \

#define declare_int_epi8_512(sz) \
    decop(add, epi8, sz) \
    decop(sub, epi8, sz) \
    decop(min, epu8, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi8, sz) \
    declare_avx512_cmpeq_mask(8)

#define declare_int_epi8_128(sz) \
    decop(add, epi8, sz) \
    decop(sub, epi8, sz) \
    decop(min, epu8, sz) \
    decop(max, epu8, sz) \
    static constexpr decltype(&OP(xor, si128, sz)) xor_fn = &OP(xor, si128, sz);\
    static constexpr decltype(&OP(and, si128, sz)) and_fn = &OP(and, si128, sz);\
    static constexpr decltype(&OP(or, si128, sz))  or_fn = &OP(or, si128, sz);\
    decop(set1, epi8, sz) \
    decop(cmpeq, epi8, sz)

#define declare_int_epi16(sz) \
    decop(slli, epi16, sz) \
    decop(srli, epi16, sz) \
    decop(add, epi16, sz) \
    decop(sub, epi16, sz) \
    decop(min, epu16, sz) \
    decop(max, epu16, sz) \
    decop(mullo, epi16, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(mullo, epi16, sz)) mul = &OP(mullo, epi16, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi16, sz) \
    decop(cmpeq, epi16, sz)

#define declare_int_epi16_512(sz) \
    decop(slli, epi16, sz) \
    decop(srli, epi16, sz) \
    decop(add, epi16, sz) \
    decop(sub, epi16, sz) \
    decop(mullo, epi16, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi16, sz) \
    declare_avx512_cmpeq_mask(16)

#define declare_int_epi16_128(sz) \
    decop(slli, epi16, sz) \
    decop(srli, epi16, sz) \
    decop(add, epi16, sz) \
    decop(min, epu16, sz) \
    decop(max, epu16, sz) \
    decop(sub, epi16, sz) \
    static constexpr decltype(&_mm_mullo_epi16) mul = &_mm_mullo_epi16;\
    static constexpr decltype(&_mm_mullo_epi16) mullo = &_mm_mullo_epi16;\
    static constexpr decltype(&OP(xor, si128, sz)) xor_fn = &OP(xor, si128, sz);\
    static constexpr decltype(&OP(and, si128, sz)) and_fn = &OP(and, si128, sz);\
    static constexpr decltype(&OP(or, si128, sz))  or_fn = &OP(or, si128, sz);\
    decop(set1, epi16, sz) \
    decop(cmpeq, epi16, sz) \


#define declare_int_epi64(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    decop(mullo, epi64, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(mullo, epi64, sz)) mul = &OP(mullo, epi64, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi64x, sz) \
    decop(cmpeq, epi64, sz) \


#define declare_int_epi64_512(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    decop(mullo, epi64, sz) \
    decop(min, epu64, sz) \
    decop(max, epu64, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi64, sz) \
    declare_avx512_cmpeq_mask(64)

#define declare_int_epi64_128(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    static constexpr decltype(&_mm_mullo_epi64) mul = &_mm_mullo_epi64;\
    static constexpr decltype(&_mm_mullo_epi64) mullo = &_mm_mullo_epi64;\
    static constexpr decltype(&OP(xor, si128, sz)) xor_fn = &OP(xor, si128, sz);\
    static constexpr decltype(&OP(and, si128, sz)) and_fn = &OP(and, si128, sz);\
    static constexpr decltype(&OP(or, si128, sz))  or_fn = &OP(or, si128, sz);\
    decop(set1, epi64x, sz) \
    decop(cmpeq, epi64, sz)

#define declare_all_int(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi64(sz)

#define declare_all_int128(suf, sz) \
    declare_int_ls128(suf, sz) \
    declare_int_epi64_128(sz)

#define declare_all_int512(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi64_512(sz) \

#define declare_all_int_32(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi32(sz)

#define declare_all_int128_32(suf, sz) \
    declare_int_ls128(suf, sz) \
    declare_int_epi32_128(sz)

#define declare_all_int512_32(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi32_512(sz)

#define declare_all_int_16(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi16(sz)

#define declare_all_int128_16(suf, sz) \
    declare_int_ls128(suf, sz) \
    declare_int_epi16_128(sz)

#define declare_all_int512_16(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi16_512(sz)

#define declare_all_int_8(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi8(sz)

#define declare_all_int128_8(suf, sz) \
    declare_int_ls128(suf, sz) \
    declare_int_epi8_128(sz)

#define declare_all_int512_8(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi8_512(sz)


#ifndef NO_SLEEF

#define SLEEF_OP(op, suf, prec, set) Sleef_##op##suf##_##prec##set
#define dec_sleefop_prec(op, suf, prec, instructset) \
    static constexpr decltype(&SLEEF_OP(op, suf, prec, instructset)) op##_##prec = \
    &SLEEF_OP(op, suf, prec, instructset); \
    struct apply_##op##_##prec {\
        template<typename... T>\
        auto operator()(T &&...args) const {return op##_##prec(std::forward<T...>(args)...);} \
        template<typename OT>\
        OT scalar(OT val) const {return scalar::op(val);} \
    };

#define dec_sleefop_noprec(op, suf, instructset) \
    static constexpr decltype(&SLEEF_OP(op, suf,, instructset)) op = \
    &SLEEF_OP(op, suf,, instructset); \
    struct apply_##op##_##prec {\
        template<typename... T>\
        auto operator()(T &&...args) const {return op(std::forward<T...>(args)...);} \
        template<typename OT>\
        OT scalar(OT val) const {return scalar::op(val);} \
    };


#define dec_all_precs(op, suf, instructset) \
    dec_sleefop_prec(op, suf, u35, instructset) \
    dec_sleefop_prec(op, suf, u10, instructset)

#define dec_all_precs_u05(op, suf, instructset) \
    dec_sleefop_prec(op, suf, u35, instructset) \
    dec_sleefop_prec(op, suf, u05, instructset)


#define dec_double_sz(type) using TypeDouble = Sleef_##type##_2;


#define declare_all_sleef_special(suf, set) \
   dec_all_precs(sin, suf, set) \
   dec_all_precs(cos, suf, set) \
   dec_all_precs(asin, suf, set) \
   dec_all_precs(acos, suf, set) \
   dec_all_precs(atan, suf, set) \
   dec_all_precs(cbrt, suf, set) \
   dec_all_precs(sincos, suf, set) \
   dec_all_precs(atan2, suf, set) \
   dec_sleefop_prec(log, suf, u10, set) \
   dec_sleefop_prec(log1p, suf, u10, set) \
   dec_sleefop_prec(pow, suf, u10, set) \
   dec_sleefop_prec(expm1, suf, u10, set) \
   dec_sleefop_prec(exp, suf, u10, set) \
   dec_sleefop_prec(exp2, suf, u10, set) \
   dec_sleefop_prec(lgamma, suf, u10, set) \
   dec_sleefop_prec(tgamma, suf, u10, set) \
   dec_sleefop_prec(erf, suf, u10, set) \
   dec_sleefop_prec(erfc, suf, u15, set) \
   dec_sleefop_prec(sinh, suf, u10, set) \
   dec_sleefop_prec(cosh, suf, u10, set) \
   dec_sleefop_prec(asinh, suf, u10, set) \
   dec_sleefop_prec(acosh, suf, u10, set) \
   dec_sleefop_prec(tanh, suf, u10, set) \
   dec_sleefop_prec(atanh, suf, u10, set) \
   dec_all_precs_u05(sqrt, suf, set) \
   dec_sleefop_noprec(floor, suf, set) \
   dec_sleefop_noprec(ceil, suf, set) \
   dec_sleefop_noprec(round, suf, set) \
   dec_sleefop_noprec(nextafter, suf, set) \
   dec_sleefop_noprec(fmod, suf, set) \
   dec_sleefop_noprec(frfrexp, suf, set) \
   dec_sleefop_noprec(remainder, suf, set) \
   dec_sleefop_noprec(trunc, suf, set)

#endif // #ifndef NO_SLEEF

template<typename SType>
union UType {
    using ValueType = typename SType::ValueType;
    using Type      = typename SType::Type;
    static constexpr size_t COUNT = SType::COUNT;
    std::array<ValueType, COUNT> arr_;
    Type                        simd_;
    UType() {}
    constexpr UType(Type val): simd_(val) {}
    constexpr UType(ValueType val): simd_(SType::set1(val)) {}
    operator Type() const {return simd_;}
    UType &operator=(Type val) {
        simd_ = val;
        return *this;
    }
    UType &operator=(ValueType val) {
        simd_ = SType::set1(val);
        return *this;
    }
    UType &operator+=(Type val) {
        simd_ += val;
        return *this;
    }
    UType &operator+=(ValueType val) {
        simd_ += SType::set1(val);
        return *this;
    }
    UType &operator-=(Type val) {
        simd_ -= val;
        return *this;
    }
    UType &operator-=(ValueType val) {
        simd_ -= SType::set1(val);
        return *this;
    }
    template<size_t nleft, size_t done>
    struct unroller {
        UType &ref_;
        template<typename Functor>
        constexpr void for_each(const Functor &func) {
            func(ref_.arr_[COUNT - nleft]);
            unroller<nleft - 1, done + 1> ur(ref_);
            ur.for_each(func);
        }
        template<typename Functor, typename AccumType>
        decltype(auto) accumulate(const Functor &func, AccumType val=AccumType()) {
            val = func(ref_.arr_[COUNT - nleft], val);
            unroller<nleft - 1, done + 1> ur(ref_);
            return ur.accumulate(func, val);
        }
        template<typename Functor, typename AccumType>
        decltype(auto) accumulate(const Functor &func, AccumType &val) {
            val = func(ref_.arr_[COUNT - nleft], val);
            unroller<nleft - 1, done + 1> ur(ref_);
            return ur.accumulate(func, val);
        }
        template<typename Functor, typename AccumType>
        decltype(auto) accumulate(const Functor &func, AccumType val=AccumType()) const {
            val = func(ref_.arr_[COUNT - nleft], val);
            unroller<nleft - 1, done + 1> ur(ref_);
            return ur.accumulate(func, val);
        }
        template<typename Functor, typename AccumType>
        decltype(auto) accumulate(const Functor &func, AccumType &val) const {
            val = func(ref_.arr_[COUNT - nleft], val);
            unroller<nleft - 1, done + 1> ur(ref_);
            return ur.accumulate(func, val);
        }
        unroller(UType &ref): ref_(ref) {}
    };
    template<size_t done>
    struct unroller<0, done> {
        UType &ref_;
        template<typename Functor> constexpr void for_each(const Functor &) {}
        unroller(UType &ref): ref_(ref){}
    };
    template<size_t nleft, size_t done>
    struct const_unroller {
        const UType &ref_;
        template<typename Functor>
        constexpr void for_each(const Functor &func) {
            func(ref_.arr_[COUNT - nleft]);
            const_unroller<nleft - 1, done + 1> ur(ref_);
            ur.for_each(func);
        }
        const_unroller(const UType &ref): ref_(ref) {}
    };
    template<size_t done>
    struct const_unroller<0, done> {
        template<typename Functor> constexpr void for_each(const Functor &) {}
        const_unroller(const UType &) {}
    };
    template<typename Functor>
    constexpr void for_each(const Functor &func) {
        unroller<COUNT, 0> ur(*this);
        ur.for_each(func);
    }
    template<typename Functor>
    constexpr void for_each(const Functor &func) const {
        const_unroller<COUNT, 0> ur(*this);
        ur.for_each(func);
    }
    auto sum() const {
        ValueType ret = arr_[0];
        for(uint8_t i = 1; i < COUNT; ++i)
            ret += arr_[i];
        return ret;
    }
};

template<>
struct SIMDTypes<uint64_t> {
    using ValueType = uint64_t;
#if HAS_AVX_512
    using Type = __m512i;
    declare_all_int512(epi64, 512)
#elif __AVX2__
    using Type = __m256i;
    declare_all_int(epi64, 256)
#elif __SSE2__
    using Type = __m128i;
    declare_all_int128(epi64,)
#else
    using Type = uint64_t;
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
    DECMAX64
};
template<> struct SIMDTypes<int64_t>: public SIMDTypes<uint64_t> {};

template<>
struct SIMDTypes<uint32_t> {
    using ValueType = uint32_t;
#if HAS_AVX_512
    using Type = __m512i;
    declare_all_int512_32(epi32, 512)
#elif __AVX2__
    using Type = __m256i;
    declare_all_int_32(epi32, 256)
#elif __SSE2__
    using Type = __m128i;
    declare_all_int128_32(epi32,)
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};
template<> struct SIMDTypes<int32_t>: public SIMDTypes<uint32_t> {};

template<>
struct SIMDTypes<uint16_t> {
    using ValueType = uint16_t;
#if HAS_AVX_512
    using Type = __m512i;
    declare_all_int512_16(epi16, 512)
#ifndef __AVX512BW__
// Max
    static INLINE __m512i _mm512_max_epu16_nobw(__m512i lhs, __m512i rhs) {
        const __m512i upper_mask = _mm512_set_epi16(0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu);
        const __m512i lower_mask = _mm512_set_epi16(0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0);
        return _mm512_max_epu32(lhs & upper_mask, rhs & upper_mask) | _mm512_max_epu32(lhs & lower_mask, rhs & lower_mask);
    }
    static INLINE __m512i max(__m512i l, __m512i r) {return _mm512_max_epu16_nobw(l, r);}
// Min
    static INLINE __m512i _mm512_min_epu16_nobw(__m512i lhs, __m512i rhs) {
        const __m512i upper_mask = _mm512_set_epi16(0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu);
        const __m512i lower_mask = _mm512_set_epi16(0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0, 0xFFFFu, 0);
        return _mm512_min_epu32(lhs & upper_mask, rhs & upper_mask) | _mm512_min_epu32(lhs & lower_mask, rhs & lower_mask);
    }
    static INLINE __m512i min(__m512i l, __m512i r) {return _mm512_min_epu16_nobw(l, r);}
#else
    static INLINE __m512i min(__m512i lhs, __m512i rhs) {
        return _mm512_min_epu16(lhs, rhs);
    }
    static INLINE __m512i max(__m512i lhs, __m512i rhs) {
        return _mm512_max_epu16(lhs, rhs);
    }
#endif
#elif __AVX2__
    using Type = __m256i;
    declare_all_int_16(epi16, 256)
#elif __SSE2__
    using Type = __m128i;
    declare_all_int128_16(epi16,)
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};
template<> struct SIMDTypes<int16_t>: public SIMDTypes<uint16_t> {};
template<> struct SIMDTypes<uint8_t> {
    using ValueType = uint8_t;
#if HAS_AVX_512
    using Type = __m512i;
    declare_all_int512_8(epi8, 512)
    static INLINE auto srli(__m512i a, int imm8) {
        return _mm512_and_si512(_mm512_set1_epi8(0xFF >> imm8), _mm512_srli_epi32(a, imm8));
    }
    static INLINE auto slli(__m512i a, int imm8) {
        return _mm512_and_si512(_mm512_set1_epi8(0xFF << imm8), _mm512_slli_epi32(a, imm8));
    }
#ifdef __AVX512BW__
    static INLINE auto max(__m512i lhs, __m512i rhs) {return _mm512_max_epu8(lhs, rhs);}
#else
    static INLINE auto max(__m512i lhs, __m512i rhs) {
        __m512i mask1 = _mm512_set_epi8(255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0);
        __m512i mask2 = _mm512_set_epi8(0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0);
        __m512i mask3 = _mm512_set_epi8(0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0);
        __m512i mask4 = _mm512_set_epi8(0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,255);
        return _mm512_max_epu32(lhs & mask1, rhs & mask1) | _mm512_max_epu32(lhs & mask2, rhs & mask2) | _mm512_max_epu32(lhs & mask3, rhs & mask3) | _mm512_max_epu32(lhs & mask4, rhs & mask4);
    }
#endif
#elif __AVX2__
    using Type = __m256i;
    declare_all_int_8(epi8, 256)
    static INLINE auto srli(__m256i a, int imm8) {
        return _mm256_and_si256(_mm256_set1_epi8(0xFF >> imm8), _mm256_srli_epi32(a, imm8));
    }
    static INLINE auto slli(__m256i a, int imm8) {
        return _mm256_and_si256(_mm256_set1_epi8(0xFF << imm8), _mm256_slli_epi32(a, imm8));
    }
#elif __SSE2__
    using Type = __m128i;
    declare_all_int128_8(epi8,)
    static INLINE auto srli(__m128i a, int imm8) {
        return _mm_and_si128(_mm_set1_epi8(0xFF >> imm8), _mm_srli_epi32(a, imm8));
    }
    static INLINE auto slli(__m128i a, int imm8) {
        return _mm_and_si128(_mm_set1_epi8(0xFF << imm8), _mm_slli_epi32(a, imm8));
    }
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};
template<> struct SIMDTypes<int8_t>: public SIMDTypes<uint8_t> {};

template<>
struct SIMDTypes<float>{
    using ValueType = float;
#if HAS_AVX_512
    using Type = __m512;
    declare_all(ps, 512)
#ifndef NO_SLEEF
    dec_double_sz(__m512)
    declare_all_sleef_special(f16, avx512f)
#endif // #ifndef NO_SLEEF
#elif __AVX2__
    using Type = __m256;
    declare_all(ps, 256)
#ifndef NO_SLEEF
    dec_double_sz(__m256)
    declare_all_sleef_special(f8, avx2)
#endif // #ifndef NO_SLEEF
#elif __SSE2__
    using Type = __m128;
    declare_all(ps, )
#ifndef NO_SLEEF
    dec_double_sz(__m128)
#  ifdef __SSE4_1__
    declare_all_sleef_special(f4, sse4)
#  else
    declare_all_sleef_special(f4, sse2)
#  endif
#endif // #ifndef NO_SLEEF
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};

template<>
struct SIMDTypes<double>{
    using ValueType = double;
#if HAS_AVX_512
    using Type = __m512d;
    declare_all(pd, 512)
#ifndef NO_SLEEF
    dec_double_sz(__m512d)
    declare_all_sleef_special(d8, avx512f)
    dec_sleefop_noprec(ldexp, d8, avx512f)
#endif // #ifndef NO_SLEEF
#elif __AVX2__
    using Type = __m256d;
    declare_all(pd, 256)
#ifndef NO_SLEEF
    dec_double_sz(__m256d)
    declare_all_sleef_special(d4, avx2)
    dec_sleefop_noprec(ldexp, d4, avx2)
#endif // #ifndef NO_SLEEF
#elif __SSE2__
    using Type = __m128d;
    declare_all(pd, )
#ifndef NO_SLEEF
    dec_double_sz(__m128d)
#  ifdef __SSE4_1__
    declare_all_sleef_special(d2, sse4)
    dec_sleefop_noprec(ldexp, d2, sse4)
#  else
    declare_all_sleef_special(d2, sse2)
    dec_sleefop_noprec(ldexp, d2, sse2)
#  endif
#endif // #ifndef NO_SLEEF
#else
#  error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};


template<typename FloatType>
void blockmul(FloatType *pos, size_t nelem, FloatType prod) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType factor(SIMDTypes<FloatType>::set1(prod));
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr)) {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(factor, Space::loadu(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        } else {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(factor, Space::load(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        }
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos++ *= prod;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] *= prod; // Could be vectorized.
#endif
}

#define BLOCKOP(op, scalar) \
template<typename Container>\
void block##op(Container &con, double val) {\
    if(&con[1] - &con[0] == 1)\
        block##op(&con[0], con.size(), static_cast<std::decay_t<decltype(*std::begin(con))>>(val));\
    else\
        for(auto &el: con) scalar;\
}

template<typename FloatType>
void blockadd(FloatType *pos, size_t nelem, FloatType val) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType inc(SIMDTypes<FloatType>::set1(val));
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr))
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::add(inc, Space::loadu(reinterpret_cast<FloatType *>(ptr)))), ++ptr;
        else
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::add(inc, Space::load(reinterpret_cast<FloatType *>(ptr)))), ++ptr;
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos++ += val;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] += val; // Could be vectorized.
#endif
}

BLOCKOP(mul, el *= val)
BLOCKOP(add, el += val)


#ifndef DO_DUFF
#define DO_DUFF(len, ITER) \
    do { \
        if(len) {\
            std::uint64_t loop = (len + 7) >> 3;\
            switch(len & 7) {\
                case 0: do {\
                    ITER; VEC_FALLTHROUGH\
                    case 7: ITER; VEC_FALLTHROUGH case 6: ITER; VEC_FALLTHROUGH case 5: ITER; VEC_FALLTHROUGH\
                    case 4: ITER; VEC_FALLTHROUGH case 3: ITER; VEC_FALLTHROUGH case 2: ITER; VEC_FALLTHROUGH case 1: ITER;\
                } while (--loop);\
            }\
        }\
    } while(0)
#endif

template<typename FloatType>
void vecmul(FloatType *to, const FloatType *from, size_t nelem) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType *ptr(reinterpret_cast<SIMDType *>(to));
        const SIMDType *fromptr(reinterpret_cast<const SIMDType *>(from));
        FloatType *end(to + nelem);
        if(!(Space::aligned(ptr) && Space::aligned(fromptr)))
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(Space::loadu(reinterpret_cast<const FloatType *>(fromptr)), Space::loadu(reinterpret_cast<FloatType *>(ptr)))),
                ++ptr, ++fromptr;
        else
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(Space::load(reinterpret_cast<const FloatType *>(fromptr)), Space::load(reinterpret_cast<FloatType *>(ptr)))),
                ++ptr, ++fromptr;
        to = reinterpret_cast<FloatType *>(ptr), from = reinterpret_cast<const FloatType *>(fromptr);
        while(to < end) *to++ *= *from++;
#else
        DO_DUFF(nelem, *to++ *= *from++);
#endif
}

template<typename FloatType, typename Functor>
void block_apply(FloatType *pos, size_t nelem, const Functor &func=Functor{}) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using Space = SIMDTypes<FloatType>;
        using SIMDType = typename Space::Type;
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr)) {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    func(Space::loadu(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        } else {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    func(Space::load(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        }
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos  = func.scalar(*pos), ++pos;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) to[i] *= func.scalar(to[i]); // Could be vectorized.
#endif
}

template<typename Container, typename Functor>
void block_apply(Container &con, const Functor &func=Functor{}) {
#ifndef NO_BLAZE
    CONST_IF(IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(Container)) {
        const size_t nelem(con.size());
        block_apply(&(*std::begin(con)), nelem, func);
    } else {
#endif
        if(&con[1] - &con[0] == 1) {
            const size_t nelem(con.size());
            block_apply(&(*std::begin(con)), nelem, func);
        } else for(auto &el: con) el = func.scalar(el);
#ifndef NO_BLAZE
    }
#endif
}

template<typename T, typename SizeType=std::size_t>
void memblockset(void *dest, T val, SizeType nbytes) {
    using S = SIMDTypes<uint64_t>;
    using SType = typename S::Type;
    SType sv;
    {
        T *s(reinterpret_cast<T *>(&sv)), *s2(reinterpret_cast<T *>(((reinterpret_cast<char *>(&sv)) + sizeof(sv))));
        while(s < s2) *s++ = val;
    }
#define DO_LOOP_(op) \
    for(SType *s = static_cast<SType *>(dest), *e = reinterpret_cast<SType *>(static_cast<char *>(dest) + nbytes); s < e; S::op(s++, sv))
    if(S::aligned(dest))
        DO_LOOP_(store);
    else
        DO_LOOP_(storeu);
#undef DO_LOOP_
}

#endif /*ifndef __aarch64__ */

} // namespace vec

#ifndef NO_SLEEF
#undef OP
#undef SLEEF_OP
#undef dec_sleefop_prec
#undef dec_all_precs
#undef dec_all_precs_u05
#undef declare_all_sleef_special
#undef dec_double_sz
#endif // #ifndef NO_SLEEF

#undef declare_all
#undef decop

#endif // #ifndef _VEC_H__
