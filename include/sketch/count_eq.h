#ifndef SKETCH_COUNT_EQ_H__
#define SKETCH_COUNT_EQ_H__
#include <x86intrin.h>
#include <sys/mman.h>
#include "sketch/common.h"

namespace sketch {namespace eq {

static inline size_t count_eq_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_nibbles(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_words(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_longs(const uint64_t *SK_RESTRICT lhs, const uint64_t *SK_RESTRICT rhs, size_t n);

static INLINE unsigned int _mm256_movemask_epi16(__m256i x) {
    return _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1)));
}
static INLINE __m256i _mm256_cmpgt_epi8_unsigned(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi8(a, b) ^ _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
}
static INLINE __m256i _mm256_cmpgt_epi16_unsigned(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi16(a, b) ^ _mm256_cmpeq_epi16(_mm256_max_epu16(a, b), a);
}

template<typename T>
static inline size_t count_eq(const T *SK_RESTRICT lhs, const T *SK_RESTRICT rhs, size_t n) {
    size_t ret = 0;
    for(size_t i = 0; i < n; ++i) ret += lhs[i] == rhs[i];
    return ret;
}
template<> inline size_t count_eq<uint16_t>(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n) {
   return count_eq_shorts(lhs, rhs, n);
}
template<> inline size_t count_eq<uint8_t>(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
   return count_eq_bytes(lhs, rhs, n);
}
template<> inline size_t count_eq<uint32_t>(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n) {
   return count_eq_words(lhs, rhs, n);
}
template<> inline size_t count_eq<uint64_t>(const uint64_t *SK_RESTRICT lhs, const uint64_t *SK_RESTRICT rhs, size_t n) {
   return count_eq_longs(lhs, rhs, n);
}

static inline size_t count_eq_shorts(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
#if __AVX512BW__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        uint64_t v1, v2, v3, v4;
        v1 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i));
        v2 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 1), _mm512_loadu_si512((__m512i *)rhs + i + 1));
        ret += popcount((uint64_t(v1) << 32) | v2);
        v3 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 2), _mm512_loadu_si512((__m512i *)rhs + i + 2));
        v4 = _mm512_cmpeq_epu16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 3), _mm512_loadu_si512((__m512i *)rhs + i + 3));
        ret += popcount((uint64_t(v3) << 32) | v4);
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm512_cmpeq_epu16_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i)));
    for(size_t i = nsimd * sizeof(__m512) / sizeof(uint16_t); i < n; ++i)
        ret += lhs[i] == rhs[i];
#elif __AVX512F__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        __m512i lhv0 = _mm512_loadu_si512((__m512i *)lhs + i + 0),
                rhv0 = _mm512_loadu_si512((__m512i *)lhs + i + 0);
        __m512i lhv1 = _mm512_loadu_si512((__m512i *)lhs + i + 1),
                rhv1 = _mm512_loadu_si512((__m512i *)lhs + i + 1);
        __m512i lhv2 = _mm512_loadu_si512((__m512i *)lhs + i + 2),
                rhv2 = _mm512_loadu_si512((__m512i *)lhs + i + 2);
        __m512i lhv3 = _mm512_loadu_si512((__m512i *)lhs + i + 3),
                rhv3 = _mm512_loadu_si512((__m512i *)lhs + i + 3);
        uint64_t eq_hi0 = _mm512_cmpeq_epi32_mask(lhv0 & _mm512_set1_epi32(0x0000FFFFu), rhv0 & _mm512_set1_epi32(0x0000FFFFu));
        uint64_t eq_lo0 = _mm512_cmpeq_epi32_mask(lhv0 & _mm512_set1_epi32(0xFFFF0000u), rhv0 & _mm512_set1_epi32(0xFFFF0000u));
        uint64_t eq_hi1 = _mm512_cmpeq_epi32_mask(lhv1 & _mm512_set1_epi32(0x0000FFFFu), rhv1 & _mm512_set1_epi32(0x0000FFFFu));
        uint64_t eq_lo1 = _mm512_cmpeq_epi32_mask(lhv1 & _mm512_set1_epi32(0xFFFF0000u), rhv1 & _mm512_set1_epi32(0xFFFF0000u));
        ret += popcount((eq_hi0 << 48) | (eq_hi1 << 32) | (eq_lo0 << 16) | eq_lo1);
        uint64_t eq_hi2 = _mm512_cmpeq_epi32_mask(lhv2 & _mm512_set1_epi32(0x0000FFFFu), rhv2 & _mm512_set1_epi32(0x0000FFFFu));
        uint64_t eq_lo2 = _mm512_cmpeq_epi32_mask(lhv2 & _mm512_set1_epi32(0xFFFF0000u), rhv2 & _mm512_set1_epi32(0xFFFF0000u));
        uint64_t eq_hi3 = _mm512_cmpeq_epi32_mask(lhv3 & _mm512_set1_epi32(0x0000FFFFu), rhv3 & _mm512_set1_epi32(0x0000FFFFu));
        uint64_t eq_lo3 = _mm512_cmpeq_epi32_mask(lhv3 & _mm512_set1_epi32(0xFFFF0000u), rhv3 & _mm512_set1_epi32(0xFFFF0000u));
        ret += popcount((eq_hi2 << 48) | (eq_hi3 << 32) | (eq_lo2 << 16) | eq_lo3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        __m512i lhv0 = _mm512_loadu_si512((__m512i *)lhs + i + 0),
                rhv0 = _mm512_loadu_si512((__m512i *)lhs + i + 0);
        uint64_t eq_hi0 = _mm512_cmpeq_epi32_mask(lhv0 & _mm512_set1_epi32(0x0000FFFFu), rhv0 & _mm512_set1_epi32(0x0000FFFFu));
        ret += popcount((eq_hi0 << 16) | _mm512_cmpeq_epi32_mask(lhv0 & _mm512_set1_epi32(0xFFFF0000u), rhv0 & _mm512_set1_epi32(0xFFFF0000u)));
    }
    for(size_t i = nsimd * sizeof(__m512) / sizeof(uint16_t); i < n; ++i)
        ret += lhs[i] == rhs[i];
#elif __AVX2__
    const size_t nsimd = (n / (sizeof(__m256) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    // Each vector register has at most 16 1-bits, so we can pack the bitmasks into 4 uint64_t popcounts.
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i), _mm256_loadu_si256((__m256i*)rhs + i));
        auto bitmask = _mm256_movemask_epi16(eq_reg);
        auto eq_reg2 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 1), _mm256_loadu_si256((__m256i*)rhs + i + 1));
        auto bitmask2 = _mm256_movemask_epi16(eq_reg2);
        auto eq_reg3 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 2), _mm256_loadu_si256((__m256i*)rhs + i + 2));
        auto bitmask3 = _mm256_movemask_epi16(eq_reg3);
        auto eq_reg4 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 3), _mm256_loadu_si256((__m256i*)rhs + i + 3));
        auto bitmask4 = _mm256_movemask_epi16(eq_reg4);
        ret += popcount((uint64_t(bitmask) << 48) | (uint64_t(bitmask2) << 32) | (uint64_t(bitmask3) << 16) | bitmask4);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i), _mm256_loadu_si256((__m256i*)rhs + i));
        auto bitmask = _mm256_movemask_epi16(eq_reg);
        ret += popcount(bitmask);
    }
    for(size_t i = nsimd * sizeof(__m256) / sizeof(uint16_t); i < n; ++i)
        ret += lhs[i] == rhs[i];
#else
    for(size_t i = 0; i < n; ++i) {
        ret += lhs[i] == rhs[i];
    }
#endif
    return ret;
}

static inline size_t count_eq_longs(const uint64_t *SK_RESTRICT lhs, const uint64_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
    for(size_t i = 0; i < n; ++i) ret += lhs[i] == rhs[i];
    return ret;
}

static inline size_t count_eq_words(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
    for(size_t i = 0; i < n; ++i) ret += lhs[i] == rhs[i];
    return ret;
}
static inline size_t count_eq_nibbles(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, const size_t nelem) {
    const size_t n = nelem >> 1;
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
#if __AVX512BW__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(char)));
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i), rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        auto lomask = _mm512_set1_epi8(0xF), himask = _mm512_set1_epi8(0x0F);
        ret += popcount(_mm512_cmpeq_epi8_mask(lhv & lomask, rhv & lomask));
        ret += popcount(_mm512_cmpeq_epi8_mask(lhv & himask, rhv & himask));
    }
    for(size_t i = nsimd * sizeof(__m512) / sizeof(char); i < n; ++i) {
        ret += (lhs[i] & 0xF) == (rhs[i] & 0xF);
        ret += (lhs[i] & 0x0F) == (rhs[i] & 0x0F);
    }
#elif __AVX2__
    const size_t nsimd = (n / (sizeof(__m256) / sizeof(char)));
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_si256((__m256i *)lhs + i), rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        auto lomask = _mm256_set1_epi8(0xF), himask = _mm256_set1_epi8(0x0F);
        uint64_t mm1 = uint64_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lhv & lomask, rhv & lomask))) << 32;
        mm1 |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(lhv & himask, rhv & himask));
        ret += popcount(mm1);
    }
    for(size_t i = nsimd * sizeof(__m256) / sizeof(char); i < n; ++i) {
        ret += (lhs[i] & 0xF) == (rhs[i] & 0xF);
        ret += (lhs[i] & 0x0F) == (rhs[i] & 0x0F);
    }
#else
    for(size_t i = 0; i < n; ++i) {
        ret += (lhs[i] & 0xF) == (rhs[i] & 0xF);
        ret += (lhs[i] & 0x0F) == (rhs[i] & 0x0F);
    }
#endif
    if(nelem & 1)
        ret += (lhs[nelem - 1] & 0xF) == (rhs[nelem - 1] & 0xF);
    return ret;
}

static inline size_t count_eq_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
#if __AVX512BW__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(char)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((__m512i *)lhs + i + 1), _mm512_loadu_si512((__m512i *)rhs + i + 1)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((__m512i *)lhs + i + 2), _mm512_loadu_si512((__m512i *)rhs + i + 2)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((__m512i *)lhs + i + 3), _mm512_loadu_si512((__m512i *)rhs + i + 3)));
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i)));
    for(size_t i = nsimd * sizeof(__m512) / sizeof(char); i < n; ++i)
        ret += lhs[i] == rhs[i];
#elif __AVX512F__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(char)));
    const size_t nsimd4 = (nsimd / 4) * 4;
#define POPC_CMP(lhv, rhv) popcount(\
                  (uint64_t(_mm512_cmpeq_epi32_mask(_mm512_slli_epi32(lhv, 24), _mm512_slli_epi32(rhv, 24))) << 48) |\
                  (uint64_t(_mm512_cmpeq_epi32_mask(lhv & _mm512_set1_epi32(0x0000FF00u), rhv & _mm512_set1_epi32(0x0000FF00u))) << 32) |\
                  (uint64_t(_mm512_cmpeq_epi32_mask(lhv & _mm512_set1_epi32(0x00FF0000u), rhv & _mm512_set1_epi32(0x00FF0000u))) << 16) |\
                  _mm512_cmpeq_epi32_mask(_mm512_srli_epi32(lhv, 24), _mm512_srli_epi32(rhv, 24)))
    for(size_t i = 0; i < nsimd4; i += 4) {
        const __m512i lhv = _mm512_loadu_si512((__m512i *)lhs + i), rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        const __m512i lhv1 = _mm512_loadu_si512((__m512i *)lhs + i + 1), rhv1 = _mm512_loadu_si512((__m512i *)rhs + i + 1);
        const __m512i lhv2 = _mm512_loadu_si512((__m512i *)lhs + i + 2), rhv2 = _mm512_loadu_si512((__m512i *)rhs + i + 2);
        const __m512i lhv3 = _mm512_loadu_si512((__m512i *)lhs + i + 3), rhv3 = _mm512_loadu_si512((__m512i *)rhs + i + 3);
        ret += POPC_CMP(lhv, rhv);
        ret += POPC_CMP(lhv1, rhv1);
        ret += POPC_CMP(lhv2, rhv2);
        ret += POPC_CMP(lhv3, rhv3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        const __m512i lhv = _mm512_loadu_si512((__m512i *)lhs + i), rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        ret += POPC_CMP(lhv, rhv);
    }
#undef POPC_CMP
    for(size_t i = nsimd * sizeof(__m512) / sizeof(char); i < n; ++i) ret += lhs[i] == rhs[i];
#elif __AVX2__
    const size_t nsimd = (n / (sizeof(__m256) / sizeof(char)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        const uint64_t v0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)lhs + i), _mm256_loadu_si256((__m256i *)rhs + i)));
        ret += popcount((v0 << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)lhs + i + 1), _mm256_loadu_si256((__m256i *)rhs + i + 1))));
        const uint64_t v2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)lhs + i + 2), _mm256_loadu_si256((__m256i *)rhs + i + 2)));
        ret += popcount((v2 << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)lhs + i + 3), _mm256_loadu_si256((__m256i *)rhs + i + 3))));
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256((__m256i *)lhs + i), _mm256_loadu_si256((__m256i *)rhs + i))));
    for(size_t i = nsimd * sizeof(__m256) / sizeof(char); i < n; ++i)
        ret += lhs[i] == rhs[i];
#else
    for(size_t i = 0; i < n; ++i) {
        ret += lhs[i] == rhs[i];
    }
#endif
    return ret;
}

static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n);
template<typename T>
static inline std::pair<uint64_t, uint64_t> count_gtlt(const T *SK_RESTRICT lhs, const T *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
    return std::make_pair(lhgt, rhgt);
}
template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    return count_gtlt_bytes(lhs, rhs, n);
}
template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n) {
    return count_gtlt_shorts(lhs, rhs, n);
}
static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512BW__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        uint64_t v0 = _mm512_cmpgt_epu8_mask(lhv, rhv);
        uint64_t v1 = _mm512_cmpgt_epu8_mask(rhv, lhv);
        lhgt += popcount(v0);
        rhgt += popcount(v1);
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX512F__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        auto lhsd = _mm512_srli_epi32(lhv, 24), rhsd =  _mm512_srli_epi32(rhv, 24);
        auto lhsu = _mm512_slli_epi32(lhv, 24), rhsu = _mm512_slli_epi32(rhv, 24);
        auto ulmask = _mm512_set1_epi32(0x00FF0000u), llmask = _mm512_set1_epi32(0x0000FF00u);
        lhgt += popcount((uint64_t(_mm512_cmpgt_epi32_mask(lhsd, rhsd)) << 48) |
                       (uint64_t(_mm512_cmpgt_epi32_mask(lhv & ulmask, rhv & ulmask)) << 32) |
                       (uint64_t(_mm512_cmpgt_epi32_mask(lhv & llmask, rhv & llmask)) << 16) |
                       _mm512_cmpgt_epi32_mask(lhsu, rhsu));
        rhgt += popcount((uint64_t(_mm512_cmpgt_epi32_mask(rhsd, lhsd)) << 48) |
                       (uint64_t(_mm512_cmpgt_epi32_mask(rhv & ulmask, lhv & ulmask)) << 32) |
                       (uint64_t(_mm512_cmpgt_epi32_mask(rhv & llmask, lhv & llmask)) << 16) |
                       _mm512_cmpgt_epi32_mask(rhsu, lhsu));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
    const size_t nper = sizeof(__m256);
    const size_t nsimd = n / nper;
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhv, lhv)));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#else
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}
static inline std::pair<uint64_t, uint64_t> count_gtlt_nibbles(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t nelem) {
    uint64_t lhgt = 0, rhgt = 0;
    const size_t n = nelem >> 1;
#if __AVX512BW__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    auto lomask = _mm512_set1_epi8(0xFu);
    auto himask = _mm512_set1_epi8(static_cast<unsigned char>(0xF0u));
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        uint64_t v0lo = _mm512_cmpgt_epu8_mask(lhv & lomask, rhv & lomask);
        uint64_t v0hi = _mm512_cmpgt_epu8_mask(lhv & himask, rhv & himask);
        uint64_t v1lo = _mm512_cmpgt_epu8_mask(rhv & lomask, lhv & lomask);
        uint64_t v1hi = _mm512_cmpgt_epu8_mask(rhv & himask, lhv & himask);
        lhgt += popcount(v0lo);
        lhgt += popcount(v0hi);
        rhgt += popcount(v1lo);
        rhgt += popcount(v1hi);
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        const auto lhl = lhs[i] & 0xFu, rhl = rhs[i] & 0xFu,
                   lhh  = lhs[i] & 0xF0u, rhh = rhs[i] & 0xF0u;
        lhgt += lhl > rhl; lhgt += lhh > rhh;
        rhgt += rhl > lhl; rhgt += rhh > lhh;
    }
#elif __AVX2__
    const size_t nper = sizeof(__m256);
    const size_t nsimd = n / nper;
    auto lomask = _mm256_set1_epi8(0xFu);
    auto himask = _mm256_set1_epi8(static_cast<unsigned char>(0xF0u));
#ifdef __GNUC__
    #pragma GCC unroll 4
#endif
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        auto lhl = lhv & lomask, rhl = rhv & lomask,
             lhh = lhv & himask, rhh = rhv & himask;
        lhgt += popcount((uint64_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhl, rhl))) << 32)
                    | _mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhh, rhh)));
        rhgt += popcount((uint64_t(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhl, lhl))) << 32)
                    | _mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhh, lhh)));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += (lhs[i] & 0xFu)  > (rhs[i] & 0xFu);
        lhgt += (lhs[i] & 0xF0u) > (rhs[i] & 0xF0u);
        rhgt += (rhs[i] & 0xFu)  > (lhs[i] & 0xFu);
        rhgt += (rhs[i] & 0xF0u) > (lhs[i] & 0xF0u);
    }
#else
    for(size_t i = 0; i < n; ++i) {
        lhgt += (lhs[i] & 0xFu)  > (rhs[i] & 0xFu);
        lhgt += (lhs[i] & 0xF0u) > (rhs[i] & 0xF0u);
        rhgt += (rhs[i] & 0xFu)  > (lhs[i] & 0xFu);
        rhgt += (rhs[i] & 0xF0u) > (lhs[i] & 0xF0u);
    }
#endif
    if(nelem & 1) {
        lhgt += (lhs[nelem >> 1] & 0xFu) > (rhs[nelem >> 1] & 0xFu);
        lhgt += (rhs[nelem >> 1] & 0xFu) > (lhs[nelem >> 1] & 0xFu);
    }
    return std::make_pair(lhgt, rhgt);
}


static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512BW__
    const size_t nper = sizeof(__m512) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lhv0 = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv0 = _mm512_loadu_si512((__m512i *)rhs + i);
        uint64_t lv0 = _mm512_cmpgt_epu16_mask(lhv0, rhv0);
        uint64_t rv0 = _mm512_cmpgt_epu16_mask(rhv0, lhv0);
        auto lhv1 = _mm512_loadu_si512((__m512i *)lhs + (i + 1));
        auto rhv1 = _mm512_loadu_si512((__m512i *)rhs + (i + 1));
        lv0 = (lv0 << 32) | _mm512_cmpgt_epu16_mask(lhv1, rhv1);
        rv0 = (rv0 << 32) | _mm512_cmpgt_epu16_mask(rhv1, lhv1);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
        auto lhv2 = _mm512_loadu_si512((__m512i *)lhs + (i + 2));
        auto rhv2 = _mm512_loadu_si512((__m512i *)rhs + (i + 2));
        lv0 = _mm512_cmpgt_epu16_mask(lhv2, rhv2);
        rv0 = _mm512_cmpgt_epu16_mask(rhv2, lhv2);
        auto lhv3 = _mm512_loadu_si512((__m512i *)lhs + (i + 3));
        auto rhv3 = _mm512_loadu_si512((__m512i *)rhs + (i + 3));
        lv0 = (lv0 << 32) | _mm512_cmpgt_epu16_mask(lhv3, rhv3);
        rv0 = (rv0 << 32) | _mm512_cmpgt_epu16_mask(rhv3, lhv3);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        uint64_t v0 = _mm512_cmpgt_epu16_mask(lhv, rhv);
        uint64_t v1 = _mm512_cmpgt_epu16_mask(rhv, lhv);
        lhgt += popcount(v0);
        rhgt += popcount(v1);
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX512F__
    const size_t nper = sizeof(__m512) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        __m512i lhv, rhv, lhsu, rhsu;
        lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        lhsu = _mm512_slli_epi32(lhv, 16);
        rhsu = _mm512_slli_epi32(rhv, 16);
        lhv = _mm512_srli_epi32(lhv, 16);
        rhv = _mm512_srli_epi32(rhv, 16);
        lhgt += popcount((_mm512_cmpgt_epi32_mask(lhsu, rhsu) << 16) | _mm512_cmpgt_epi32_mask(lhv, rhv));
        rhgt += popcount((_mm512_cmpgt_epi32_mask(rhsu, lhsu) << 16) | _mm512_cmpgt_epi32_mask(rhv, lhv));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
    const size_t nper = sizeof(__m256) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
#ifndef NDEBUG
    size_t lhgtn = 0, rhgtn = 0;
    size_t lhgtni = 0, rhgtni = 0;
    for(size_t i = 0; i < n; ++i)
        lhgtn += lhs[i] > rhs[i], rhgtn += rhs[i] > lhs[i];
    for(size_t i = 0; i < nsimd * nper; ++i)
        lhgtni += lhs[i] > rhs[i], rhgtni += rhs[i] > lhs[i];
    size_t lhman =0, rhman = 0;
#endif
    assert(lhs != rhs);
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        assert(std::equal((uint16_t *)((__m256i *)lhs + i), (uint16_t *)((__m256i *)lhs + i) + 16, (uint16_t *)&lhv));
        assert(std::equal((uint16_t *)((__m256i *)rhs + i), (uint16_t *)((__m256i *)rhs + i) + 16, (uint16_t *)&rhv));
#ifndef NDEBUG
        for(size_t j = 0; j < nper; ++j) {
            lhman += lhs[i * nper + j] > rhs[i * nper + j];
            rhman += rhs[i * nper + j] > lhs[i * nper + j];
        }
#endif
        lhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(rhv, lhv)));
        assert(lhgt == lhman || !std::fprintf(stderr, "lhgt %zu, lhman %zu\n", size_t(lhgt), lhman));
        assert(rhgt == rhman || !std::fprintf(stderr, "rhgt %zu, rhman %zu\n", size_t(rhgt), rhman));
    }
    for(size_t i = nper * nsimd; i < n; ++i) {
        const auto lhv = lhs[i], rhv = rhs[i];
        lhgt += (lhv > rhv); rhgt += (rhv > lhv);
    }
#else
    SK_UNROLL_4
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}

}} // sketch::eq

#endif
