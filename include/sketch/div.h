#ifndef SKETCH_DIV_H__
#define SKETCH_DIV_H__

#include <cstdint>
#include <cassert>
#include <utility>
#include <limits>
#include <algorithm>

#include "sketch/intrinsics.h"

#undef INLINE
#if __GNUC__ || __clang__
#  define INLINE __attribute__((always_inline)) inline
#elif __CUDACC__
#  define INLINE __forceinline__ inline
#else
#  define INLINE inline
#endif

#ifndef CONST_IF
#if defined(__cpp_if_constexpr) && __cplusplus >= __cpp_if_constexpr
#define CONST_IF(...) if constexpr(__VA_ARGS__)
#else
#define CONST_IF(...) if(__VA_ARGS__)
#endif
#endif

// Extrapolated from 32-it method at https://github.com/lemire/fastmod and its accompanying paper
// Method for 64-bit integers developed and available at https://github.com/dnbaker/fastmod

namespace schism {
using std::uint32_t;
using std::uint64_t;


static inline __uint128_t computeM_u64(uint64_t d) {
  return (__uint128_t(-1) / d) + 1;
}

static inline uint64_t mul128_u64(__uint128_t lowbits, uint64_t d) {
  __uint128_t bottom_half = (lowbits & UINT64_C(0xFFFFFFFFFFFFFFFF)) * d; // Won't overflow
  bottom_half >>= 64;  // Only need the top 64 bits, as we'll shift the lower half away;
  __uint128_t top_half = (lowbits >> 64) * d;
  __uint128_t both_halves = bottom_half + top_half; // Both halves are already shifted down by 64
  return (both_halves >>= 64); // Get top half of both_halves
}
static inline uint64_t fastdiv_u64(uint64_t a, __uint128_t M) {
  return mul128_u64(M, a);
}

static inline uint64_t fastmod_u64(uint64_t a, __uint128_t M, uint64_t d) {
  __uint128_t lowbits = M * a;
  return mul128_u64(lowbits, d);
}
static inline uint64_t computeM_u32(uint32_t d) {
  return UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1;
}
static inline uint64_t mul128_u32(uint64_t lowbits, uint32_t d) {
  return ((__uint128_t)lowbits * d) >> 64;
}
static inline uint32_t fastmod_u32(uint32_t a, uint64_t M, uint32_t d) {
  uint64_t lowbits = M * a;
  return (uint32_t)(mul128_u32(lowbits, d));
}
#ifdef __AVX2__
INLINE __m128i cvtepi64_epi32_avx(__m256i v)
{
   __m256 vf = _mm256_castsi256_ps( v );      // free
   __m128 hi = _mm256_extractf128_ps(vf, 1);  // vextractf128
   __m128 lo = _mm256_castps256_ps128( vf );  // also free
   // take the bottom 32 bits of each 64-bit chunk in lo and hi
   __m128 packed = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));  // shufps
   return _mm_castps_si128(packed);  // if you want
}
static inline __m256i mul64_haswell (__m256i a, __m256i b);
static INLINE __m256i mul64_constant32(__m256i lhs, const uint32_t d) {
    const __m256i prod = _mm256_mul_epu32(lhs, _mm256_set1_epi32(d));               // a0Lb0L,a1Lb1L, 64 bit unsigned products
#ifndef NDEBUG
    __m256i actual_right = mul64_haswell(lhs, _mm256_set1_epi64x(d));
    assert(std::equal((uint8_t *)&actual_right, (uint8_t *)&actual_right + 32, (uint8_t *)&prod));
#endif
#if 0
    __m256i bswap   = _mm256_shuffle_epi32(_mm256_set1_epi64x(d),0xB1);        // swap H<->L
    __m256i prodlh  = _mm256_mullo_epi32(lhs,bswap);         // 32 bit L*H products
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh,_mm256_setzero_si256());      // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);  // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m256i prodll  = _mm256_mul_epu32(lhs,_mm256_set1_epi32(d));               // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m256i prod    = _mm256_add_epi64(prodll,prodlh3);    // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
#endif
    return  prod;
}


INLINE __m256i _mm256_mulhi_epi64_manual(const __m256i lhs, const uint32_t d) {
    // Now, we want a __m256i which has 4 64-bit integers
    __m256i lo_lhs_lo_upcasted = _mm256_cvtepu32_epi64(cvtepi64_epi32_avx(lhs));
    __m256i lo_lhs_hi_upcasted = _mm256_cvtepu32_epi64(cvtepi64_epi32_avx(_mm256_srli_epi64(lhs, 32)));
    __m256i lo_a_x_b_mid = mul64_constant32(lo_lhs_hi_upcasted, d);
    __m256i lo_a_x_b_lo = mul64_constant32(lo_lhs_lo_upcasted, d);
    __m256i lo_carry_bit = _mm256_srli_epi64(_mm256_add_epi64(_mm256_cvtepu32_epi64(cvtepi64_epi32_avx(lo_a_x_b_mid)), _mm256_srli_epi64(lo_a_x_b_lo, 32)), 32);
    return _mm256_add_epi64(_mm256_srli_epi64(lo_a_x_b_mid, 32), lo_carry_bit);
}
static INLINE __m128i mul128_u32_si256(__m256i lowbits, uint32_t d) {
 return cvtepi64_epi32_avx(_mm256_mulhi_epi64_manual(lowbits, d));
}
INLINE __m256i pack64to32(__m256i a, __m256i b)
{
    // grab the 32-bit low halves of 64-bit elements into one vector
   __m256 combined = _mm256_shuffle_ps(_mm256_castsi256_ps(a),
                                       _mm256_castsi256_ps(b), _MM_SHUFFLE(2,0,2,0));
    // {b3,b2, a3,a2 | b1,b0, a1,a0}  from high to low

    // re-arrange pairs of 32-bit elements with vpermpd (or vpermq if you want)
    __m256d ordered = _mm256_permute4x64_pd(_mm256_castps_pd(combined), _MM_SHUFFLE(3,1,2,0));
    return _mm256_castpd_si256(ordered);
}


INLINE __m256i mul128_u32_si256(__m256i lhs1, __m256i lhs2, const uint32_t d) {
    __m256i packedcvtboth_lo = pack64to32(lhs1, lhs2);
    __m256i packedcvtboth_hi = pack64to32(_mm256_srli_epi64(lhs1, 32), _mm256_srli_epi64(lhs2, 32));
    // Now, we want a __m256i which has 4 64-bit integers
    __m256i lo_lhs_lo_upcasted = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(packedcvtboth_lo));
    __m256i lo_lhs_hi_upcasted = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(packedcvtboth_hi));
    __m256i hi_lhs_lo_upcasted = _mm256_cvtepu32_epi64(_mm256_extractf128_si256(packedcvtboth_lo, 1));
    __m256i hi_lhs_hi_upcasted = _mm256_cvtepu32_epi64(_mm256_extractf128_si256(packedcvtboth_hi, 1));
    __m256i lo_a_x_b_mid = mul64_constant32(lo_lhs_hi_upcasted, d);
    __m256i lo_a_x_b_lo = mul64_constant32(lo_lhs_lo_upcasted, d);
    __m256i lo_carry_bit = _mm256_srli_epi64(_mm256_add_epi64(_mm256_cvtepu32_epi64(cvtepi64_epi32_avx(lo_a_x_b_mid)), _mm256_srli_epi64(lo_a_x_b_lo, 32)), 32);
    __m256i hi_a_x_b_mid = mul64_constant32(hi_lhs_hi_upcasted, d);
    __m256i hi_a_x_b_lo = mul64_constant32(hi_lhs_lo_upcasted, d);
    __m256i hi_carry_bit = _mm256_srli_epi64(_mm256_add_epi64(_mm256_cvtepu32_epi64(cvtepi64_epi32_avx(hi_a_x_b_mid)), _mm256_srli_epi64(hi_a_x_b_lo, 32)), 32);
    __m256i lhr =  _mm256_add_epi64(_mm256_srli_epi64(lo_a_x_b_mid, 32), lo_carry_bit);
    __m256i rhr =  _mm256_add_epi64(_mm256_srli_epi64(lo_a_x_b_mid, 32), hi_carry_bit);
    return pack64to32(lhr, rhr);
}
static inline __m256i mul64_haswell (__m256i a, __m256i b) {
    // From vectorclass V2
    __m256i bswap   = _mm256_shuffle_epi32(b,0xB1);        // swap H<->L
    __m256i prodlh  = _mm256_mullo_epi32(a,bswap);         // 32 bit L*H products
    __m256i zero    = _mm256_setzero_si256();              // 0
    __m256i prodlh2 = _mm256_hadd_epi32(prodlh,zero);      // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
    __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2,0x73);  // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
    __m256i prodll  = _mm256_mul_epu32(a,b);               // a0Lb0L,a1Lb1L, 64 bit unsigned products
    __m256i prod    = _mm256_add_epi64(prodll,prodlh3);    // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
    return  prod;
}

static inline __m256i fastmod_u32(__m256i a, uint64_t M, uint32_t d) {
  __m128i loex = _mm256_castsi256_si128(a), hiex = _mm256_extractf128_si256(a, 1);
  assert(std::equal((uint32_t *)&loex, (uint32_t *)&loex + 4, (uint32_t *)&a));
  __m256i loa = _mm256_cvtepu32_epi64(loex);
  assert(std::equal((uint64_t *)&loa, (uint64_t *)&loa + 4, (uint32_t *)&loex));
  assert(std::equal((uint64_t *)&loa, (uint64_t *)&loa + 4, (uint32_t *)&a));
  __m256i hia = _mm256_cvtepu32_epi64(hiex);
  auto lo_lo = mul64_haswell(loa, _mm256_set1_epi64x(M));
  auto hi_lo = mul64_haswell(hia, _mm256_set1_epi64x(M));
  auto lo_mullo = mul128_u32_si256(lo_lo, d);
  auto hi_mullo = mul128_u32_si256(hi_lo, d);
  return _mm256_setr_m128i(lo_mullo, hi_mullo);
}
#if 0
static inline std::pair<__m256i> fastmod_u32(__m256i a, __m256i a2, uint64_t M, uint32_t d) {
  __m128i loex = _mm256_castsi256_si128(a), hiex = _mm256_extractf128_si256(a, 1);
  __m256i loa = _mm256_cvtepu32_epi64(loex);
  __m256i hia = _mm256_cvtepu32_epi64(hiex);
  auto lo_lo = mul64_haswell(loa, _mm256_set1_epi64x(M));
  auto hi_lo = mul64_haswell(hia, _mm256_set1_epi64x(M));
  auto lo_mullo = mul128_u32_si256(lo_lo, d);
  auto hi_mullo = mul128_u32_si256(hi_lo, d);
  __m128i loex2 = _mm256_castsi256_si128(a), hiex2 = _mm256_extractf128_si256(a, 1);
  __m256i loa2 = _mm256_cvtepu32_epi64(loex);
  __m256i hia2 = _mm256_cvtepu32_epi64(hiex);
  auto lo_lo2 = mul64_haswell(loa2, _mm256_set1_epi64x(M));
  auto hi_lo2 = mul64_haswell(hia2, _mm256_set1_epi64x(M));
  auto lo_mullo2 = mul128_u32_si256(lo_lo, d);
  auto hi_mullo2 = mul128_u32_si256(hi_lo, d);
}
#endif
#endif

// fastmod computes (a / d) given precomputed M for d>1
static inline uint32_t fastdiv_u32(uint32_t a, uint64_t M) {
  return (uint32_t)(mul128_u32(M, a));
}

template<typename T> struct div_t {
    T quot;
    T rem;
    operator std::pair<T, T> &() {
        return *reinterpret_cast<std::pair<T, T> *>(this);
    }
    auto &first() {
        return this->quot;
    }
    auto &first() const {
        return this->quot;
    }
    auto &second() {
        return this->rem;
    }
    auto &second() const {
        return this->rem;
    }
    std::pair<T, T> to_pair() const {return std::make_pair(quot, rem);}
    operator const std::pair<T, T> &() const {
        return *const_cast<std::pair<T, T> *>(this);
    }
};


template<typename T, bool shortcircuit=false>
struct Schismatic;
template<bool shortcircuit> struct Schismatic<uint64_t, shortcircuit> {
private:
    uint64_t d_;
    __uint128_t M_;
    uint64_t m32_;
    uint64_t &m32() {
        return m32_;
    }
    // We swap location here so that m32 can be 64-bit aligned.
public:
    const auto &d() const {return d_;}
    const uint64_t &m32() const {assert(shortcircuit); return m32_;}
    using DivType = div_t<uint64_t>;
    Schismatic(uint64_t d): d_(d), M_(computeM_u64(d)) {
        CONST_IF(shortcircuit) {
            m32_ = computeM_u32(d);
        } else {
            m32_ = 0;
        }
    }
    INLINE bool test_limits(uint64_t v) const {
        assert(shortcircuit);
        static constexpr uint64_t threshold = std::numeric_limits<uint32_t>::max();
        return d_ <= threshold && v <= threshold;
    }
    INLINE uint64_t div(uint64_t v) const {
        if(shortcircuit) {
            return test_limits(v) ? uint64_t(fastdiv_u32(v, m32_)): fastdiv_u64(v, m32_);
        }
        return fastdiv_u64(v, M_);
    }
    INLINE uint64_t mod(uint64_t v) const {
        if(shortcircuit)
            return test_limits(v) ? uint64_t(fastmod_u32(v, m32_, d_)): fastmod_u64(v, m32_, d_);
        return fastmod_u64(v, M_, d_);
    }
    INLINE div_t<uint64_t> divmod(uint64_t v) const {
        auto d = div(v);
        return div_t<uint64_t> {d, v - d_ * d};
    }
};
template<> struct Schismatic<uint32_t> {
    uint32_t d_;
    uint64_t M_;
    Schismatic(uint32_t d): d_(d), M_(computeM_u32(d)) {}
    auto d() const {return d_;}
    INLINE uint32_t div(uint32_t v) const {return fastdiv_u32(v, M_);}
    INLINE uint32_t mod(uint32_t v) const {return fastmod_u32(v, M_, d_);}
#ifdef __AVX2__
    INLINE auto mod(__m256i v) const {return fastmod_u32(v, M_, d_);}
    //INLINE auto mod(__m256i v, __m256i v2) const {return fastmod_u32(v, v2, M_, d_);}
#endif
    INLINE div_t<uint32_t> divmod(uint32_t v) const {
        auto tmpd = div(v);
        return div_t<uint32_t> {tmpd, v - d_ * tmpd};
    }
};

} // namespace schism

#endif /* SKETCH_DIV_H__  */
