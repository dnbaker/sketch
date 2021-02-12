#ifndef SKETCH_COUNT_EQ_H__
#define SKETCH_COUNT_EQ_H__
#include <x86intrin.h>
#include <sys/mman.h>
#include "sketch/common.h"

namespace sketch {namespace eq {

static inline size_t count_eq_shorts(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_nibbles(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_words(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_longs(const uint64_t *SK_RESTRICT lhs, const uint64_t *SK_RESTRICT rhs, size_t n);

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
#if SEPARATE_POPC
        auto v1 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i));
        auto v2 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i));
        ret += popcount((uint64_t(v1) << 32) | v2);
        auto v3 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 2), _mm512_loadu_si512((__m512i *)rhs + i + 2));
        auto v4 = _mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 3), _mm512_loadu_si512((__m512i *)rhs + i + 3));
        ret += popcount((uint64_t(v3) << 32) | v4);
#else
        ret += popcount(_mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 0), _mm512_loadu_si512((__m512i *)rhs + i + 0)));
        ret += popcount(_mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 1), _mm512_loadu_si512((__m512i *)rhs + i + 1)));
        ret += popcount(_mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 2), _mm512_loadu_si512((__m512i *)rhs + i + 2)));
        ret += popcount(_mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i + 3), _mm512_loadu_si512((__m512i *)rhs + i + 3)));
#endif
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm512_cmpeq_epi16_mask(_mm512_loadu_si512((__m512i *)lhs + i), _mm512_loadu_si512((__m512i *)rhs + i)));
    for(size_t i = nsimd * sizeof(__m512) / sizeof(uint16_t); i < n; ++i)
        ret += lhs[i] == rhs[i];
#elif __AVX2__
    const size_t nsimd = (n / (sizeof(__m256) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    // Each vector register has at most 16 1-bits, so we can pack the bitmasks into 4 uint64_t popcounts.
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i), _mm256_loadu_si256((__m256i*)rhs + i));
        auto bitmask = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(eq_reg), _mm256_extracti128_si256(eq_reg, 1)));
        auto eq_reg2 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 1), _mm256_loadu_si256((__m256i*)rhs + i + 1));
        auto bitmask2 = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(eq_reg2), _mm256_extracti128_si256(eq_reg2, 1)));
        auto eq_reg3 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 2), _mm256_loadu_si256((__m256i*)rhs + i + 2));
        auto bitmask3 = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(eq_reg3), _mm256_extracti128_si256(eq_reg3, 1)));
        auto eq_reg4 = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i + 3), _mm256_loadu_si256((__m256i*)rhs + i + 3));
        auto bitmask4 = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(eq_reg4), _mm256_extracti128_si256(eq_reg4, 1)));
        ret += popcount((uint64_t(bitmask) << 48) | (uint64_t(bitmask2) << 32) | (uint64_t(bitmask3) << 16) | bitmask4);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_loadu_si256((__m256i*)lhs + i), _mm256_loadu_si256((__m256i*)rhs + i));
        auto bitmask = _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(eq_reg), _mm256_extracti128_si256(eq_reg, 1)));
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

}} // sketch::eq

#endif
