#ifndef SKETCH_COUNT_EQ_H__
#define SKETCH_COUNT_EQ_H__
#include <sys/mman.h>
#include "sketch/intrinsics.h"
#include "sketch/common.h"

namespace sketch {namespace eq {

static inline size_t count_eq_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_nibbles(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_words(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_longs(const uint64_t *SK_RESTRICT lhs, const uint64_t *SK_RESTRICT rhs, size_t n);

static inline size_t count_eq_bytes_aligned(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline size_t count_eq_shorts_aligned(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n);

static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n);

static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes_aligned(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n);
static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts_aligned(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n);

static inline std::pair<uint64_t, uint64_t> count_gtlt_words(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n);

static inline std::pair<uint64_t, uint64_t> count_gtlt_words_aligned(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, size_t n);

#ifdef __AVX2__
static INLINE unsigned int _mm256_movemask_epi16(__m256i x) {
    return _mm_movemask_epi8(_mm_packs_epi16(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1)));
}
static INLINE __m256i _mm256_cmpgt_epi8_unsigned(__m256i a, __m256i b) {
    return _mm256_andnot_si256(_mm256_cmpeq_epi8(a, b), _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a));
}
static INLINE __m256i _mm256_cmpgt_epi16_unsigned(__m256i a, __m256i b) {
    return _mm256_andnot_si256(_mm256_cmpeq_epi16(a, b), _mm256_cmpeq_epi16(_mm256_max_epu16(a, b), a));
}
#endif

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
#if __AVX512F__
    if(reinterpret_cast<uint64_t>(lhs) % 64 == 0 && reinterpret_cast<uint64_t>(rhs) % 64 == 0) return count_eq_shorts_aligned(lhs, rhs, n);
#elif __AVX2__
    if(reinterpret_cast<uint64_t>(lhs) % 32 == 0 && reinterpret_cast<uint64_t>(rhs) % 32 == 0) return count_eq_shorts_aligned(lhs, rhs, n);
#endif
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
                rhv0 = _mm512_loadu_si512((__m512i *)rhs + i + 0);
        __m512i lhv1 = _mm512_loadu_si512((__m512i *)lhs + i + 1),
                rhv1 = _mm512_loadu_si512((__m512i *)rhs + i + 1);
        __m512i lhv2 = _mm512_loadu_si512((__m512i *)lhs + i + 2),
                rhv2 = _mm512_loadu_si512((__m512i *)rhs + i + 2);
        __m512i lhv3 = _mm512_loadu_si512((__m512i *)lhs + i + 3),
                rhv3 = _mm512_loadu_si512((__m512i *)rhs + i + 3);
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
                rhv0 = _mm512_loadu_si512((__m512i *)rhs + i + 0);
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
static inline size_t count_eq_shorts_aligned(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
#if __AVX512BW__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        uint64_t v1, v2, v3, v4;
        v1 = _mm512_cmpeq_epu16_mask(_mm512_load_si512((__m512i *)lhs + i), _mm512_load_si512((__m512i *)rhs + i));
        v2 = _mm512_cmpeq_epu16_mask(_mm512_load_si512((__m512i *)lhs + i + 1), _mm512_load_si512((__m512i *)rhs + i + 1));
        ret += popcount((uint64_t(v1) << 32) | v2);
        v3 = _mm512_cmpeq_epu16_mask(_mm512_load_si512((__m512i *)lhs + i + 2), _mm512_load_si512((__m512i *)rhs + i + 2));
        v4 = _mm512_cmpeq_epu16_mask(_mm512_load_si512((__m512i *)lhs + i + 3), _mm512_load_si512((__m512i *)rhs + i + 3));
        ret += popcount((uint64_t(v3) << 32) | v4);
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm512_cmpeq_epu16_mask(_mm512_load_si512((__m512i *)lhs + i), _mm512_load_si512((__m512i *)rhs + i)));
    for(size_t i = nsimd * sizeof(__m512) / sizeof(uint16_t); i < n; ++i)
        ret += lhs[i] == rhs[i];
#elif __AVX512F__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(uint16_t)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        __m512i lhv0 = _mm512_load_si512((__m512i *)lhs + i + 0),
                rhv0 = _mm512_load_si512((__m512i *)rhs + i + 0);
        __m512i lhv1 = _mm512_load_si512((__m512i *)lhs + i + 1),
                rhv1 = _mm512_load_si512((__m512i *)rhs + i + 1);
        __m512i lhv2 = _mm512_load_si512((__m512i *)lhs + i + 2),
                rhv2 = _mm512_load_si512((__m512i *)rhs + i + 2);
        __m512i lhv3 = _mm512_load_si512((__m512i *)lhs + i + 3),
                rhv3 = _mm512_load_si512((__m512i *)rhs + i + 3);
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
        __m512i lhv0 = _mm512_load_si512((__m512i *)lhs + i + 0),
                rhv0 = _mm512_load_si512((__m512i *)rhs + i + 0);
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
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_load_si256((__m256i*)lhs + i), _mm256_load_si256((__m256i*)rhs + i));
        auto bitmask = _mm256_movemask_epi16(eq_reg);
        auto eq_reg2 = _mm256_cmpeq_epi16(_mm256_load_si256((__m256i*)lhs + i + 1), _mm256_load_si256((__m256i*)rhs + i + 1));
        auto bitmask2 = _mm256_movemask_epi16(eq_reg2);
        auto eq_reg3 = _mm256_cmpeq_epi16(_mm256_load_si256((__m256i*)lhs + i + 2), _mm256_load_si256((__m256i*)rhs + i + 2));
        auto bitmask3 = _mm256_movemask_epi16(eq_reg3);
        auto eq_reg4 = _mm256_cmpeq_epi16(_mm256_load_si256((__m256i*)lhs + i + 3), _mm256_load_si256((__m256i*)rhs + i + 3));
        auto bitmask4 = _mm256_movemask_epi16(eq_reg4);
        ret += popcount((uint64_t(bitmask) << 48) | (uint64_t(bitmask2) << 32) | (uint64_t(bitmask3) << 16) | bitmask4);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto eq_reg = _mm256_cmpeq_epi16(_mm256_load_si256((__m256i*)lhs + i), _mm256_load_si256((__m256i*)rhs + i));
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
        ret += (lhs[nelem / 2] & 0xF) == (rhs[nelem / 2] & 0xF);
    return ret;
}

static inline size_t count_eq_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
#if __AVX512F__
    if(reinterpret_cast<uint64_t>(lhs) % 64 == 0 && reinterpret_cast<uint64_t>(rhs) % 64 == 0) return count_eq_bytes_aligned(lhs, rhs, n);
#elif __AVX2__
    if(reinterpret_cast<uint64_t>(lhs) % 32 == 0 && reinterpret_cast<uint64_t>(rhs) % 32 == 0) return count_eq_bytes_aligned(lhs, rhs, n);
#endif
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
static inline size_t count_eq_bytes_aligned(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    advise_mem(lhs, rhs, n);
    size_t ret = 0;
#if __AVX512BW__
    const size_t nsimd = (n / (sizeof(__m512) / sizeof(char)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512((__m512i *)lhs + i), _mm512_load_si512((__m512i *)rhs + i)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512((__m512i *)lhs + i + 1), _mm512_load_si512((__m512i *)rhs + i + 1)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512((__m512i *)lhs + i + 2), _mm512_load_si512((__m512i *)rhs + i + 2)));
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512((__m512i *)lhs + i + 3), _mm512_load_si512((__m512i *)rhs + i + 3)));
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512((__m512i *)lhs + i), _mm512_load_si512((__m512i *)rhs + i)));
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
        const __m512i lhv = _mm512_load_si512((__m512i *)lhs + i), rhv = _mm512_load_si512((__m512i *)rhs + i);
        const __m512i lhv1 = _mm512_load_si512((__m512i *)lhs + i + 1), rhv1 = _mm512_load_si512((__m512i *)rhs + i + 1);
        const __m512i lhv2 = _mm512_load_si512((__m512i *)lhs + i + 2), rhv2 = _mm512_load_si512((__m512i *)rhs + i + 2);
        const __m512i lhv3 = _mm512_load_si512((__m512i *)lhs + i + 3), rhv3 = _mm512_load_si512((__m512i *)rhs + i + 3);
        ret += POPC_CMP(lhv, rhv);
        ret += POPC_CMP(lhv1, rhv1);
        ret += POPC_CMP(lhv2, rhv2);
        ret += POPC_CMP(lhv3, rhv3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        const __m512i lhv = _mm512_load_si512((__m512i *)lhs + i), rhv = _mm512_load_si512((__m512i *)rhs + i);
        ret += POPC_CMP(lhv, rhv);
    }
#undef POPC_CMP
    for(size_t i = nsimd * sizeof(__m512) / sizeof(char); i < n; ++i) ret += lhs[i] == rhs[i];
#elif __AVX2__
    const size_t nsimd = (n / (sizeof(__m256) / sizeof(char)));
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        const uint64_t v0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((__m256i *)lhs + i), _mm256_load_si256((__m256i *)rhs + i)));
        ret += popcount((v0 << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((__m256i *)lhs + i + 1), _mm256_load_si256((__m256i *)rhs + i + 1))));
        const uint64_t v2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((__m256i *)lhs + i + 2), _mm256_load_si256((__m256i *)rhs + i + 2)));
        ret += popcount((v2 << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((__m256i *)lhs + i + 3), _mm256_load_si256((__m256i *)rhs + i + 3))));
    }
    for(size_t i = nsimd4; i < nsimd; ++i)
        ret += popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_load_si256((__m256i *)lhs + i), _mm256_load_si256((__m256i *)rhs + i))));
    for(size_t i = nsimd * sizeof(__m256) / sizeof(char); i < n; ++i)
        ret += lhs[i] == rhs[i];
#else
    for(size_t i = 0; i < n; ++i) {
        ret += lhs[i] == rhs[i];
    }
#endif
    return ret;
}

template<typename T>
static inline std::pair<uint64_t, uint64_t> count_gtlt(const T *SK_RESTRICT lhs, const T *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
    return std::make_pair(lhgt, rhgt);
}
#if __AVX512F__  || __AVX2__
template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const double *SK_RESTRICT lhs, const double *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512F__
    const size_t nper = 8;
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lh0 = _mm512_loadu_pd(lhs + i * nper), rh0 = _mm512_loadu_pd(rhs + i * nper);
        auto lh1 = _mm512_loadu_pd(lhs + (i + 1) * nper), rh1 = _mm512_loadu_pd(rhs + (i + 1) * nper);
        auto lh2 = _mm512_loadu_pd(lhs + (i + 2) * nper), rh2 = _mm512_loadu_pd(rhs + (i + 2) * nper);
        auto lh3 = _mm512_loadu_pd(lhs + (i + 3) * nper), rh3 = _mm512_loadu_pd(rhs + (i + 3) * nper);
        auto cmp0 = _mm512_cmp_pd_mask(lh0, rh0, _CMP_GT_OQ);
        auto cmp1 = _mm512_cmp_pd_mask(lh1, rh1, _CMP_GT_OQ);
        auto cmp2 = _mm512_cmp_pd_mask(lh2, rh2, _CMP_GT_OQ);
        auto cmp3 = _mm512_cmp_pd_mask(lh3, rh3, _CMP_GT_OQ);
        lhgt += popcount((cmp0 << 24) | (cmp1 << 16) | (cmp2 << 8) | cmp3);
        auto rcmp0 = _mm512_cmp_pd_mask(rh0, lh0, _CMP_GT_OQ);
        auto rcmp1 = _mm512_cmp_pd_mask(rh1, lh1, _CMP_GT_OQ);
        auto rcmp2 = _mm512_cmp_pd_mask(rh2, lh2, _CMP_GT_OQ);
        auto rcmp3 = _mm512_cmp_pd_mask(rh3, lh3, _CMP_GT_OQ);
        rhgt += popcount((rcmp0 << 24) | (rcmp1 << 16) | (rcmp2 << 8) | rcmp3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_pd(lhs + i * nper), rhv = _mm512_loadu_pd(rhs + i * nper);
        lhgt += popcount(_mm512_cmp_pd_mask(lhv, rhv, _CMP_GT_OQ));
        rhgt += popcount(_mm512_cmp_pd_mask(rhv, lhv, _CMP_GT_OQ));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
    const size_t nsimd = n / 4;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lh0 = _mm256_loadu_pd(lhs + i * 4), rh0 = _mm256_loadu_pd(rhs + i * 4);
        auto lh1 = _mm256_loadu_pd(lhs + (i + 1) * 4), rh1 = _mm256_loadu_pd(rhs + (i + 1) * 4);
        auto lh2 = _mm256_loadu_pd(lhs + (i + 2) * 4), rh2 = _mm256_loadu_pd(rhs + (i + 2) * 4);
        auto lh3 = _mm256_loadu_pd(lhs + (i + 3) * 4), rh3 = _mm256_loadu_pd(rhs + (i + 3) * 4);
        auto cmp0 = _mm256_movemask_pd(_mm256_cmp_pd(lh0, rh0, _CMP_GT_OQ));
        auto cmp1 = _mm256_movemask_pd(_mm256_cmp_pd(lh1, rh1, _CMP_GT_OQ));
        auto cmp2 = _mm256_movemask_pd(_mm256_cmp_pd(lh2, rh2, _CMP_GT_OQ));
        auto cmp3 = _mm256_movemask_pd(_mm256_cmp_pd(lh3, rh3, _CMP_GT_OQ));
        lhgt += popcount((cmp0 << 12) | (cmp1 << 8) | (cmp2 << 4) | cmp3);
        auto rcmp0 = _mm256_movemask_pd(_mm256_cmp_pd(rh0, lh0, _CMP_GT_OQ));
        auto rcmp1 = _mm256_movemask_pd(_mm256_cmp_pd(rh1, lh1, _CMP_GT_OQ));
        auto rcmp2 = _mm256_movemask_pd(_mm256_cmp_pd(rh2, lh2, _CMP_GT_OQ));
        auto rcmp3 = _mm256_movemask_pd(_mm256_cmp_pd(rh3, lh3, _CMP_GT_OQ));
        rhgt += popcount((rcmp0 << 12) | (rcmp1 << 8) | (rcmp2 << 4) | rcmp3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_pd(lhs + i * 4), rhv = _mm256_loadu_pd(rhs + i * 4);
        lhgt += popcount(_mm256_movemask_pd(_mm256_cmp_pd(lhv, rhv, _CMP_GT_OQ)));
        rhgt += popcount(_mm256_movemask_pd(_mm256_cmp_pd(rhv, lhv, _CMP_GT_OQ)));
    }
    for(size_t i = nsimd * 4; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}
#endif
#if __AVX512F__  || __AVX2__
template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const float *SK_RESTRICT lhs, const float *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512F__
    static constexpr size_t nper = 16;
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lh0 = _mm512_loadu_ps(lhs + i * nper), rh0 = _mm512_loadu_ps(rhs + i * nper);
        auto lh1 = _mm512_loadu_ps(lhs + (i + 1) * nper), rh1 = _mm512_loadu_ps(rhs + (i + 1) * nper);
        auto lh2 = _mm512_loadu_ps(lhs + (i + 2) * nper), rh2 = _mm512_loadu_ps(rhs + (i + 2) * nper);
        auto lh3 = _mm512_loadu_ps(lhs + (i + 3) * nper), rh3 = _mm512_loadu_ps(rhs + (i + 3) * nper);
        uint64_t cmp0 = _mm512_cmp_ps_mask(lh0, rh0, _CMP_GT_OQ);
        uint64_t cmp1 = _mm512_cmp_ps_mask(lh1, rh1, _CMP_GT_OQ);
        uint64_t cmp2 = _mm512_cmp_ps_mask(lh2, rh2, _CMP_GT_OQ);
        uint64_t cmp3 = _mm512_cmp_ps_mask(lh3, rh3, _CMP_GT_OQ);
        lhgt += popcount((cmp0 << 48) | (cmp1 << 32) | (cmp2 << 16) | cmp3);
        uint64_t rcmp0 = _mm512_cmp_ps_mask(rh0, lh0, _CMP_GT_OQ);
        uint64_t rcmp1 = _mm512_cmp_ps_mask(rh1, lh1, _CMP_GT_OQ);
        uint64_t rcmp2 = _mm512_cmp_ps_mask(rh2, lh2, _CMP_GT_OQ);
        uint64_t rcmp3 = _mm512_cmp_ps_mask(rh3, lh3, _CMP_GT_OQ);
        rhgt += popcount((rcmp0 << 48) | (rcmp1 << 32) | (rcmp2 << 16) | rcmp3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_ps(lhs + i * nper), rhv = _mm512_loadu_ps(rhs + i * nper);
        lhgt += popcount(_mm512_cmp_ps_mask(lhv, rhv, _CMP_GT_OQ));
        rhgt += popcount(_mm512_cmp_ps_mask(rhv, lhv, _CMP_GT_OQ));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
    const size_t nsimd = n / 8;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lh0 = _mm256_loadu_ps(lhs + i * 8), rh0 = _mm256_loadu_ps(rhs + i * 8);
        auto lh1 = _mm256_loadu_ps(lhs + (i + 1) * 8), rh1 = _mm256_loadu_ps(rhs + (i + 1) * 8);
        auto lh2 = _mm256_loadu_ps(lhs + (i + 2) * 8), rh2 = _mm256_loadu_ps(rhs + (i + 2) * 8);
        auto lh3 = _mm256_loadu_ps(lhs + (i + 3) * 8), rh3 = _mm256_loadu_ps(rhs + (i + 3) * 8);
        auto cmp0 = _mm256_movemask_ps(_mm256_cmp_ps(lh0, rh0, _CMP_GT_OQ));
        auto cmp1 = _mm256_movemask_ps(_mm256_cmp_ps(lh1, rh1, _CMP_GT_OQ));
        auto cmp2 = _mm256_movemask_ps(_mm256_cmp_ps(lh2, rh2, _CMP_GT_OQ));
        auto cmp3 = _mm256_movemask_ps(_mm256_cmp_ps(lh3, rh3, _CMP_GT_OQ));
        lhgt += popcount((cmp0 << 24) | (cmp1 << 16) | (cmp2 << 8) | cmp3);
        auto rcmp0 = _mm256_movemask_ps(_mm256_cmp_ps(rh0, lh0, _CMP_GT_OQ));
        auto rcmp1 = _mm256_movemask_ps(_mm256_cmp_ps(rh1, lh1, _CMP_GT_OQ));
        auto rcmp2 = _mm256_movemask_ps(_mm256_cmp_ps(rh2, lh2, _CMP_GT_OQ));
        auto rcmp3 = _mm256_movemask_ps(_mm256_cmp_ps(rh3, lh3, _CMP_GT_OQ));
        rhgt += popcount((rcmp0 << 24) | (rcmp1 << 16) | (rcmp2 << 8) | rcmp3);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_ps(lhs + i * 4), rhv = _mm256_loadu_ps(rhs + i * 4);
        lhgt += popcount(_mm256_movemask_ps(_mm256_cmp_ps(lhv, rhv, _CMP_GT_OQ)));
        rhgt += popcount(_mm256_movemask_ps(_mm256_cmp_ps(rhv, lhv, _CMP_GT_OQ)));
    }
    for(size_t i = nsimd * 4; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}
#endif

template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    return count_gtlt_bytes(lhs, rhs, n);
}
template<> inline std::pair<uint64_t, uint64_t> count_gtlt(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, size_t n) {
    return count_gtlt_shorts(lhs, rhs, n);
}
static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
#if __AVX512F__
    if(reinterpret_cast<uint64_t>(lhs) % 64 == 0 && reinterpret_cast<uint64_t>(rhs) % 64 == 0) return count_gtlt_bytes_aligned(lhs, rhs, n);
#elif __AVX2__
    if(reinterpret_cast<uint64_t>(lhs) % 32 == 0 && reinterpret_cast<uint64_t>(rhs) % 32 == 0) return count_gtlt_bytes_aligned(lhs, rhs, n);
#endif
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512BW__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    size_t i = 0;
    for(; i < nsimd4; i += 4) {
        const auto lhv = _mm512_loadu_si512((__m512i *)lhs + i), lh1 = _mm512_loadu_si512((__m512i *)lhs + i + 1), lh2 = _mm512_loadu_si512((__m512i *)lhs + i + 2), lh3 = _mm512_loadu_si512((__m512i *)lhs + i + 3);
        const auto rhv = _mm512_loadu_si512((__m512i *)rhs + i), rh1 = _mm512_loadu_si512((__m512i *)rhs + i + 1), rh2 = _mm512_loadu_si512((__m512i *)rhs + i + 2), rh3 = _mm512_loadu_si512((__m512i *)rhs + i + 3);
        lhgt += popcount(_mm512_cmpgt_epu8_mask(lhv, rhv)) + popcount(_mm512_cmpgt_epu8_mask(lh1, rh1)) + popcount(_mm512_cmpgt_epu8_mask(lh2, rh2)) + popcount(_mm512_cmpgt_epu8_mask(lh3, rh3));
        rhgt += popcount(_mm512_cmpgt_epu8_mask(rhv, lhv)) + popcount(_mm512_cmpgt_epu8_mask(rh1, lh1)) + popcount(_mm512_cmpgt_epu8_mask(rh2, lh2)) + popcount(_mm512_cmpgt_epu8_mask(rh3, lh3));
    }
    for(; i < nsimd; ++i) {
        const auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        const auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        lhgt += popcount(_mm512_cmpgt_epu8_mask(lhv, rhv));
        rhgt += popcount(_mm512_cmpgt_epu8_mask(rhv, lhv));
    }
    for(i = nsimd * nper; i < n; ++i) {
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
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_loadu_si256((__m256i *)lhs + i), rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhv, lhv)));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i]; rhgt += rhs[i] > lhs[i];
    }
#else
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}

static inline std::pair<uint64_t, uint64_t> count_gtlt_bytes_aligned(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512BW__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    size_t i = 0;
    for(; i < nsimd4; i += 4) {
        const auto lhv = _mm512_load_si512((__m512i *)lhs + i), lh1 = _mm512_load_si512((__m512i *)lhs + i + 1), lh2 = _mm512_load_si512((__m512i *)lhs + i + 2), lh3 = _mm512_load_si512((__m512i *)lhs + i + 3);
        const auto rhv = _mm512_load_si512((__m512i *)rhs + i), rh1 = _mm512_load_si512((__m512i *)rhs + i + 1), rh2 = _mm512_load_si512((__m512i *)rhs + i + 2), rh3 = _mm512_load_si512((__m512i *)rhs + i + 3);
        lhgt += popcount(_mm512_cmpgt_epu8_mask(lhv, rhv)) + popcount(_mm512_cmpgt_epu8_mask(lh1, rh1)) + popcount(_mm512_cmpgt_epu8_mask(lh2, rh2)) + popcount(_mm512_cmpgt_epu8_mask(lh3, rh3));
        rhgt += popcount(_mm512_cmpgt_epu8_mask(rhv, lhv)) + popcount(_mm512_cmpgt_epu8_mask(rh1, lh1)) + popcount(_mm512_cmpgt_epu8_mask(rh2, lh2)) + popcount(_mm512_cmpgt_epu8_mask(rh3, lh3));
    }
    for(; i < nsimd; ++i) {
        const auto lhv = _mm512_load_si512((__m512i *)lhs + i);
        const auto rhv = _mm512_load_si512((__m512i *)rhs + i);
        lhgt += popcount(_mm512_cmpgt_epu8_mask(lhv, rhv));
        rhgt += popcount(_mm512_cmpgt_epu8_mask(rhv, lhv));
    }
    for(i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX512F__
    const size_t nper = sizeof(__m512);
    const size_t nsimd = n / nper;
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm512_load_si512((__m512i *)lhs + i);
        auto rhv = _mm512_load_si512((__m512i *)rhs + i);
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
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_load_si256((__m256i *)lhs + i), rhv = _mm256_load_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhv, lhv)));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i]; rhgt += rhs[i] > lhs[i];
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
        auto lhlo = lhv & lomask, lhhi = lhv & himask;
        auto rhlo = rhv & lomask, rhhi = rhv & himask;
        lhgt += popcount(_mm512_cmpgt_epu8_mask(lhlo, rhlo)) + popcount(_mm512_cmpgt_epu8_mask(lhhi, rhhi));
        rhgt += popcount(_mm512_cmpgt_epu8_mask(rhlo, lhlo)) + popcount(_mm512_cmpgt_epu8_mask(rhlo, lhlo));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        const auto lhl = lhs[i] & 0xFu, rhl = rhs[i] & 0xFu,
                   lhh  = lhs[i] & 0xF0u, rhh = rhs[i] & 0xF0u;
        lhgt += lhl > rhl;
        lhgt += lhh > rhh;
        rhgt += rhl > lhl;
        rhgt += rhh > lhh;
    }
#elif __AVX2__
    const size_t nper = sizeof(__m256);
    const size_t nsimd = n / nper;
    auto lomask = _mm256_set1_epi8(0xFu);
    auto himask = _mm256_set1_epi8(static_cast<unsigned char>(0xF0u));
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        auto lhl = lhv & lomask, rhl = rhv & lomask,
             lhh = lhv & himask, rhh = rhv & himask;
        lhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhl, rhl)))
             + popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(lhh, rhh)));
        rhgt += popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhl, lhl)))
             + popcount(_mm256_movemask_epi8(_mm256_cmpgt_epi8_unsigned(rhh, lhh)));
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
static inline std::pair<uint64_t, uint64_t> count_gtlt_nibbles(const char *SK_RESTRICT lhs, const char *SK_RESTRICT rhs, size_t nelem) {
    return count_gtlt_nibbles((const uint8_t *SK_RESTRICT)lhs, (const uint8_t *SK_RESTRICT)rhs, nelem);
}

static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n) {
#if __AVX512F__
    if(reinterpret_cast<uint64_t>(lhs) % 64 == 0 && reinterpret_cast<uint64_t>(rhs) % 64 == 0) return count_gtlt_shorts_aligned(lhs, rhs, n);
#elif __AVX2__
    if(reinterpret_cast<uint64_t>(lhs) % 32 == 0 && reinterpret_cast<uint64_t>(rhs) % 32 == 0) return count_gtlt_shorts_aligned(lhs, rhs, n);
#endif
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
        const auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        const auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        lhgt += popcount(_mm512_cmpgt_epu16_mask(lhv, rhv));
        rhgt += popcount(_mm512_cmpgt_epu16_mask(rhv, lhv));
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
    assert(lhs != rhs);
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(rhv, lhv)));
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
static inline std::pair<uint64_t, uint64_t> count_gtlt_words(const uint32_t *const SK_RESTRICT lhs, const uint32_t *const SK_RESTRICT rhs, size_t n) {
#if __AVX512F__
    if(reinterpret_cast<uint64_t>(lhs) % 64 == 0 && reinterpret_cast<uint64_t>(rhs) % 64 == 0) return count_gtlt_words_aligned(lhs, rhs, n);
#elif __AVX2__
    if(reinterpret_cast<uint64_t>(lhs) % 32 == 0 && reinterpret_cast<uint64_t>(rhs) % 32 == 0) return count_gtlt_words_aligned(lhs, rhs, n);
#endif
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512F__
    const size_t nper = sizeof(__m512) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lhv0 = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv0 = _mm512_loadu_si512((__m512i *)rhs + i);
        uint64_t lv0 = _mm512_cmpgt_epu32_mask(lhv0, rhv0);
        uint64_t rv0 = _mm512_cmpgt_epu32_mask(rhv0, lhv0);
        auto lhv1 = _mm512_loadu_si512((__m512i *)lhs + (i + 1));
        auto rhv1 = _mm512_loadu_si512((__m512i *)rhs + (i + 1));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv1, rhv1);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv1, lhv1);
        auto lhv2 = _mm512_loadu_si512((__m512i *)lhs + (i + 2));
        auto rhv2 = _mm512_loadu_si512((__m512i *)rhs + (i + 2));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv2, rhv2);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv2, lhv2);
        auto lhv3 = _mm512_loadu_si512((__m512i *)lhs + (i + 3));
        auto rhv3 = _mm512_loadu_si512((__m512i *)rhs + (i + 3));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv3, rhv3);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv3, lhv3);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_loadu_si512((__m512i *)lhs + i);
        auto rhv = _mm512_loadu_si512((__m512i *)rhs + i);
        lhgt += popcount(_mm512_cmpgt_epu32_mask(lhv, rhv));
        rhgt += popcount(_mm512_cmpgt_epu32_mask(rhv, lhv));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
    const size_t nper = sizeof(__m256) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    assert(lhs != rhs);
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_loadu_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_loadu_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv, rhv))));
        rhgt += popcount(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv, lhv))));
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
union uf {
    unsigned u;
    float f;
    constexpr uf(unsigned v): u(v) {}
};
static constexpr float from_unsigned(unsigned x) {
    uf tmp(x);
    return tmp.f;
}
static inline std::pair<uint64_t, uint64_t> count_gtlt_words_aligned(const uint32_t *const SK_RESTRICT lhs, const uint32_t *const SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512F__
    const size_t nper = sizeof(__m512) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lhv0 = _mm512_load_si512((__m512i *)lhs + i);
        auto rhv0 = _mm512_load_si512((__m512i *)rhs + i);
        uint64_t lv0 = _mm512_cmpgt_epu32_mask(lhv0, rhv0);
        uint64_t rv0 = _mm512_cmpgt_epu32_mask(rhv0, lhv0);
        auto lhv1 = _mm512_load_si512((__m512i *)lhs + (i + 1));
        auto rhv1 = _mm512_load_si512((__m512i *)rhs + (i + 1));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv1, rhv1);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv1, lhv1);
        auto lhv2 = _mm512_load_si512((__m512i *)lhs + (i + 2));
        auto rhv2 = _mm512_load_si512((__m512i *)rhs + (i + 2));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv2, rhv2);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv2, lhv2);
        auto lhv3 = _mm512_load_si512((__m512i *)lhs + (i + 3));
        auto rhv3 = _mm512_load_si512((__m512i *)rhs + (i + 3));
        lv0 = (lv0 << 16) | _mm512_cmpgt_epu32_mask(lhv3, rhv3);
        rv0 = (rv0 << 16) | _mm512_cmpgt_epu32_mask(rhv3, lhv3);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_load_si512((__m512i *)lhs + i);
        auto rhv = _mm512_load_si512((__m512i *)rhs + i);
        lhgt += popcount(_mm512_cmpgt_epu32_mask(lhv, rhv));
        rhgt += popcount(_mm512_cmpgt_epu32_mask(rhv, lhv));
    }
    for(size_t i = nsimd * nper; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#elif __AVX2__
#define _mm256_cmpgt_epu32(x, y) _mm256_cmpgt_epi32(_mm256_xor_si256((a), _mm256_set1_epi32(0x80000000)), _mm256_xor_si256((b), _mm256_set1_epi32(0x80000000)))
    const size_t nper = sizeof(__m256) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    size_t i = 0;
    for(; i < nsimd4; i += 4) {
        const auto lhv = _mm256_load_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_load_si256((__m256i *)rhs + i);
        const auto lhv1 = _mm256_load_si256((__m256i *)lhs + i + 1);
        const auto rhv1 = _mm256_load_si256((__m256i *)rhs + i + 1);
        const auto lhv2 = _mm256_load_si256((__m256i *)lhs + i + 2);
        const auto rhv2 = _mm256_load_si256((__m256i *)rhs + i + 2);
        const auto lhv3 = _mm256_load_si256((__m256i *)lhs + i + 3);
        const auto rhv3 = _mm256_load_si256((__m256i *)rhs + i + 3);
        lhgt += popcount(
                  (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv, rhv))) << 24)
                | (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv1, rhv1))) << 16)
                | (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv2, rhv2))) << 8)
                | _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv3, rhv3)))
        );
        rhgt += popcount(
                  (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv, lhv))) << 24)
                | (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv1, lhv1))) << 16)
                | (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv2, lhv2))) << 8)
                | _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv3, lhv3)))
        );
    }
    for(; i < nsimd; ++i) {
        const auto lhv = _mm256_load_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_load_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(lhv, rhv))));
        rhgt += popcount(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(rhv, lhv))));
    }
    for(i = nper * nsimd; i < n; ++i) {
        const auto lhv = lhs[i], rhv = rhs[i];
        lhgt += (lhv > rhv); rhgt += (rhv > lhv);
    }
#undef _mm256_cmpgt_epu32
#else
    SK_UNROLL_4
    for(size_t i = 0; i < n; ++i) {
        lhgt += lhs[i] > rhs[i];
        rhgt += rhs[i] > lhs[i];
    }
#endif
    return std::make_pair(lhgt, rhgt);
}

static inline std::pair<uint64_t, uint64_t> count_gtlt_shorts_aligned(const uint16_t *const SK_RESTRICT lhs, const uint16_t *const SK_RESTRICT rhs, size_t n) {
    uint64_t lhgt = 0, rhgt = 0;
#if __AVX512BW__
    const size_t nper = sizeof(__m512) / sizeof(uint16_t);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(size_t i = 0; i < nsimd4; i += 4) {
        auto lhv0 = _mm512_load_si512((__m512i *)lhs + i);
        auto rhv0 = _mm512_load_si512((__m512i *)rhs + i);
        uint64_t lv0 = _mm512_cmpgt_epu16_mask(lhv0, rhv0);
        uint64_t rv0 = _mm512_cmpgt_epu16_mask(rhv0, lhv0);
        auto lhv1 = _mm512_load_si512((__m512i *)lhs + (i + 1));
        auto rhv1 = _mm512_load_si512((__m512i *)rhs + (i + 1));
        lv0 = (lv0 << 32) | _mm512_cmpgt_epu16_mask(lhv1, rhv1);
        rv0 = (rv0 << 32) | _mm512_cmpgt_epu16_mask(rhv1, lhv1);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
        auto lhv2 = _mm512_load_si512((__m512i *)lhs + (i + 2));
        auto rhv2 = _mm512_load_si512((__m512i *)rhs + (i + 2));
        lv0 = _mm512_cmpgt_epu16_mask(lhv2, rhv2);
        rv0 = _mm512_cmpgt_epu16_mask(rhv2, lhv2);
        auto lhv3 = _mm512_load_si512((__m512i *)lhs + (i + 3));
        auto rhv3 = _mm512_load_si512((__m512i *)rhs + (i + 3));
        lv0 = (lv0 << 32) | _mm512_cmpgt_epu16_mask(lhv3, rhv3);
        rv0 = (rv0 << 32) | _mm512_cmpgt_epu16_mask(rhv3, lhv3);
        lhgt += popcount(lv0);
        rhgt += popcount(rv0);
    }
    for(size_t i = nsimd4; i < nsimd; ++i) {
        auto lhv = _mm512_load_si512((__m512i *)lhs + i);
        auto rhv = _mm512_load_si512((__m512i *)rhs + i);
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
        lhv = _mm512_load_si512((__m512i *)lhs + i); rhv = _mm512_load_si512((__m512i *)rhs + i);
        lhsu = _mm512_slli_epi32(lhv, 16); rhsu = _mm512_slli_epi32(rhv, 16);
        lhv = _mm512_srli_epi32(lhv, 16);  rhv = _mm512_srli_epi32(rhv, 16);
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
    SK_UNROLL_4
    for(size_t i = 0; i < nsimd; ++i) {
        const auto lhv = _mm256_load_si256((__m256i *)lhs + i);
        const auto rhv = _mm256_load_si256((__m256i *)rhs + i);
        lhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(lhv, rhv)));
        rhgt += popcount(_mm256_movemask_epi16(_mm256_cmpgt_epi16_unsigned(rhv, lhv)));
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
