#ifndef ISZ_H__
#define ISZ_H__
#include <cstdint>
#include <functional>
#include <algorithm>
#include <cassert>
#include "macros.h"
#if defined(__AVX2__)
#include "sketch/intrinsics.h"
#endif

namespace sketch {
namespace isz {
template<typename Container, typename Cmp=std::less<>>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    //static_assert(std::is_same<decltype(*std::begin(c1)), decltype(*std::begin(c2))>::value, "Containers must derefernce to the same type.");
    //for(const auto v: c1) std::fprintf(stderr, "element is %zu\n", size_t(v));
    assert(std::is_sorted(c2.begin(), c2.end(), cmp));
    assert(std::is_sorted(c1.begin(), c1.end(), cmp));
    auto it1 = std::begin(c1);
    auto it2 = std::begin(c2);
    const auto e1 = std::cend(c1);
    const auto e2 = std::cend(c2);
    if(it1 == e1 || it2 == e2) return 0;
    std::uint64_t ret = 0;
    FOREVER {
        if(*it1 == *it2) { // Easily predicted
            ++ret;
            if(++it1 == e1 || ++it2 == e2) break;
        } else if(cmp(*it1, *it2)) {
            if(++it1 == e1) break;
        } else {
            if(++it2 == e2) break;
        }
    }
    return ret;
}

namespace detail {

// Scalar merge count over raw sorted pointers (shared by the SIMD tails and
// the non-AVX2 fallback). Pairwise-multiplicity semantics, matching
// isz::intersection_size above.
inline std::uint64_t isz_merge_count(const std::uint32_t *a, std::size_t na,
                                     const std::uint32_t *b, std::size_t nb) {
    std::uint64_t ret = 0;
    std::size_t i = 0, j = 0;
    while(i < na && j < nb) {
        const std::uint32_t x = a[i], y = b[j];
        if(x == y) { ++ret; ++i; ++j; }
        else if(x < y) ++i;
        else ++j;
    }
    return ret;
}
inline std::uint64_t isz_merge_count(const std::uint64_t *a, std::size_t na,
                                     const std::uint64_t *b, std::size_t nb) {
    std::uint64_t ret = 0;
    std::size_t i = 0, j = 0;
    while(i < na && j < nb) {
        const std::uint64_t x = a[i], y = b[j];
        if(x == y) { ++ret; ++i; ++j; }
        else if(x < y) ++i;
        else ++j;
    }
    return ret;
}

// Bounded merge count (thresholded comparison). Contract, shared by all
// isz::*_min entry points: returns a value v <= exact with
//   (v >= min_count) == (exact >= min_count).
// So `intersection_size_*_min(...) >= min_count` is the EXACT answer to "is
// the overlap at least min_count?" — that threshold question is what early
// exit preserves: as soon as the running count reaches min_count the answer
// is yes, and once count + min(remaining_a, remaining_b) drops below
// min_count the answer is permanently no. The returned value below the
// threshold is the count so far, NOT the exact count — callers needing the
// exact number when it is small must use the unbounded entry points.
inline std::uint64_t isz_merge_count_min(const std::uint32_t *a, std::size_t na,
                                         const std::uint32_t *b, std::size_t nb,
                                         std::uint64_t min_count) {
    if(min_count == 0) return isz_merge_count(a, na, b, nb);
    std::uint64_t ret = 0;
    std::size_t i = 0, j = 0;
    while(i < na && j < nb) {
        const std::uint32_t x = a[i], y = b[j];
        if(x == y) { ++ret; ++i; ++j; if(ret >= min_count) return ret; }
        else if(x < y) ++i; else ++j;
        const std::size_t ra = na - i, rb = nb - j;
        if(ret + (ra < rb ? ra : rb) < min_count) return ret;
    }
    return ret;
}
inline std::uint64_t isz_merge_count_min(const std::uint64_t *a, std::size_t na,
                                         const std::uint64_t *b, std::size_t nb,
                                         std::uint64_t min_count) {
    if(min_count == 0) return isz_merge_count(a, na, b, nb);
    std::uint64_t ret = 0;
    std::size_t i = 0, j = 0;
    while(i < na && j < nb) {
        const std::uint64_t x = a[i], y = b[j];
        if(x == y) { ++ret; ++i; ++j; if(ret >= min_count) return ret; }
        else if(x < y) ++i; else ++j;
        const std::size_t ra = na - i, rb = nb - j;
        if(ret + (ra < rb ? ra : rb) < min_count) return ret;
    }
    return ret;
}

// FESISA is only profitable on long-enough lists (measured crossover on
// Coffee Lake: parity below ~2k registers, 2.5-6.5x at 8k) — below the
// threshold the scalar merge runs, which also skips the duplicate scan.
inline constexpr std::size_t isz_fesisa_min_len = 2048;

// Sorted-adjacent-duplicate scan: true iff any equal neighbors. Gates the
// FESISA fast paths — the block-max advance and the per-lane match mask are
// exact for duplicate-free lists only, so multiset inputs (e.g. FinalCRMinHash
// registers padded to sketch size with repeated T::max()) fall back to the
// scalar merge, preserving pairwise-multiplicity semantics exactly.
inline bool isz_has_dups(const std::uint32_t *a, std::size_t n) {
    std::size_t i = 0;
#if defined(__AVX2__)
    for(; i + 9 <= n; i += 8) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i v1 = _mm256_loadu_si256((const __m256i *)(a + i + 1));
        if(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v0, v1))))
            return true;
    }
#endif
    for(; i + 1 < n; ++i) if(a[i] == a[i + 1]) return true;
    return false;
}
inline bool isz_has_dups(const std::uint64_t *a, std::size_t n) {
    std::size_t i = 0;
#if defined(__AVX2__)
    for(; i + 5 <= n; i += 4) {
        const __m256i v0 = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i v1 = _mm256_loadu_si256((const __m256i *)(a + i + 1));
        if(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(v0, v1))))
            return true;
    }
#endif
    for(; i + 1 < n; ++i) if(a[i] == a[i + 1]) return true;
    return false;
}

#if defined(__AVX2__)
// FESISA match-matrix masks: bit t of the result is set iff a-lane t equals
// some lane of the b block. u32 compares an 8x8 matrix (shuffle_epi32 covers
// 4 broadcasts per 128-bit half; permute2x128 swaps halves for the other 4);
// u64 compares a 4x4 matrix via permute4x64 broadcasts. Inoue et al. 2014.
inline unsigned isz_block_mask(std::uint32_t, const __m256i va, const __m256i vb) {
    const __m256i vbs = _mm256_permute2x128_si256(vb, vb, 0x01);
    unsigned m = 0;
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vb, 0x00)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vb, 0x55)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vb, 0xAA)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vb, 0xFF)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vbs, 0x00)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vbs, 0x55)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vbs, 0xAA)))));
    m |= unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(va, _mm256_shuffle_epi32(vbs, 0xFF)))));
    return m;
}
inline unsigned isz_block_mask(std::uint64_t, const __m256i va, const __m256i vb) {
    unsigned m = 0;
    m |= unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(va, _mm256_permute4x64_epi64(vb, 0x00)))));
    m |= unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(va, _mm256_permute4x64_epi64(vb, 0x55)))));
    m |= unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(va, _mm256_permute4x64_epi64(vb, 0xAA)))));
    m |= unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(va, _mm256_permute4x64_epi64(vb, 0xFF)))));
    return m;
}
#endif

} // detail

// Counting intersection over sorted integer lists with EXACT
// pairwise-multiplicity semantics (identical to the generic
// isz::intersection_size merge): each common value contributes
// min(multiplicity in a, multiplicity in b).
//
// Fast path: FESISA block intersection (Inoue, Ohoyama, Fukui, "Faster Set
// Intersection with SIMD Instructions by Reducing Branch Mispredictions",
// PVLDB 8(5) 2014) — each step compares a full block of each list through
// the match matrix (8x8 for u32 via shuffle_epi32 + a 128-bit half swap,
// 4x4 for u64 via permute4x64), counts matched a-lanes with popcount, and
// advances both pointers by a branchless block-max comparison; a scalar
// merge tail handles the ragged end. The FESISA advance is exact for
// duplicate-free lists only, so a sorted-adjacent-duplicate scan gates it
// (minhash registers are duplicate-free sets by construction, EXCEPT padded
// counting sketches — FinalCRMinHash pads with repeated T::max()); multiset
// inputs take the scalar merge. Compiled to the scalar merge when __AVX2__
// is unavailable.
inline std::uint64_t intersection_size_sets(const std::uint32_t *a, std::size_t na,
                                            const std::uint32_t *b, std::size_t nb) {
#ifndef NDEBUG
    assert(std::is_sorted(a, a + na));
    assert(std::is_sorted(b, b + nb));
#endif
#if defined(__AVX2__)
    if(na < detail::isz_fesisa_min_len || nb < detail::isz_fesisa_min_len ||
       detail::isz_has_dups(a, na) || detail::isz_has_dups(b, nb))
        return detail::isz_merge_count(a, na, b, nb);
    std::uint64_t count = 0;
    std::size_t i = 0, j = 0;
    while(i + 8 <= na && j + 8 <= nb) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + j));
        count += unsigned(__builtin_popcount(detail::isz_block_mask(std::uint32_t(), va, vb)));
        const std::uint32_t amax = a[i + 7], bmax = b[j + 7];
        i += 8 * std::size_t(amax <= bmax);
        j += 8 * std::size_t(bmax <= amax);
    }
    return count + detail::isz_merge_count(a + i, na - i, b + j, nb - j);
#else
    return detail::isz_merge_count(a, na, b, nb);
#endif
}

inline std::uint64_t intersection_size_sets(const std::uint64_t *a, std::size_t na,
                                            const std::uint64_t *b, std::size_t nb) {
#ifndef NDEBUG
    assert(std::is_sorted(a, a + na));
    assert(std::is_sorted(b, b + nb));
#endif
#if defined(__AVX2__)
    if(na < detail::isz_fesisa_min_len || nb < detail::isz_fesisa_min_len ||
       detail::isz_has_dups(a, na) || detail::isz_has_dups(b, nb))
        return detail::isz_merge_count(a, na, b, nb);
    std::uint64_t count = 0;
    std::size_t i = 0, j = 0;
    while(i + 4 <= na && j + 4 <= nb) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + j));
        count += unsigned(__builtin_popcount(detail::isz_block_mask(std::uint64_t(), va, vb)));
        const std::uint64_t amax = a[i + 3], bmax = b[j + 3];
        i += 4 * std::size_t(amax <= bmax);
        j += 4 * std::size_t(bmax <= amax);
    }
    return count + detail::isz_merge_count(a + i, na - i, b + j, nb - j);
#else
    return detail::isz_merge_count(a, na, b, nb);
#endif
}

// Bounded count (the thresholded-comparison entry point): same exact
// pairwise-multiplicity semantics and duplicate-scan FESISA gating as the
// unbounded overloads, and the same threshold contract as scalar
// detail::isz_merge_count_min — a value v <= exact with
//   (v >= min_count) == (exact >= min_count).
// "Is the overlap at least t?" is answered exactly by
// intersection_size_sets_min(...) >= t, which is what LSH candidate
// filtering wants to ask.
inline std::uint64_t intersection_size_sets_min(const std::uint32_t *a, std::size_t na,
                                                const std::uint32_t *b, std::size_t nb,
                                                std::uint64_t min_count) {
    if(min_count == 0) return intersection_size_sets(a, na, b, nb);
#ifndef NDEBUG
    assert(std::is_sorted(a, a + na));
    assert(std::is_sorted(b, b + nb));
#endif
#if defined(__AVX2__)
    if(na < detail::isz_fesisa_min_len || nb < detail::isz_fesisa_min_len ||
       detail::isz_has_dups(a, na) || detail::isz_has_dups(b, nb))
        return detail::isz_merge_count_min(a, na, b, nb, min_count);
    std::uint64_t count = 0;
    std::size_t i = 0, j = 0;
    while(i + 8 <= na && j + 8 <= nb) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + j));
        count += unsigned(__builtin_popcount(detail::isz_block_mask(std::uint32_t(), va, vb)));
        const std::uint32_t amax = a[i + 7], bmax = b[j + 7];
        i += 8 * std::size_t(amax <= bmax);
        j += 8 * std::size_t(bmax <= amax);
        if(count >= min_count) return count;
        const std::size_t ra = na - i, rb = nb - j;
        if(count + (ra < rb ? ra : rb) < min_count) return count;
    }
    return count + detail::isz_merge_count_min(a + i, na - i, b + j, nb - j, min_count - count);
#else
    return detail::isz_merge_count_min(a, na, b, nb, min_count);
#endif
}

inline std::uint64_t intersection_size_sets_min(const std::uint64_t *a, std::size_t na,
                                                const std::uint64_t *b, std::size_t nb,
                                                std::uint64_t min_count) {
    if(min_count == 0) return intersection_size_sets(a, na, b, nb);
#ifndef NDEBUG
    assert(std::is_sorted(a, a + na));
    assert(std::is_sorted(b, b + nb));
#endif
#if defined(__AVX2__)
    if(na < detail::isz_fesisa_min_len || nb < detail::isz_fesisa_min_len ||
       detail::isz_has_dups(a, na) || detail::isz_has_dups(b, nb))
        return detail::isz_merge_count_min(a, na, b, nb, min_count);
    std::uint64_t count = 0;
    std::size_t i = 0, j = 0;
    while(i + 4 <= na && j + 4 <= nb) {
        const __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        const __m256i vb = _mm256_loadu_si256((const __m256i *)(b + j));
        count += unsigned(__builtin_popcount(detail::isz_block_mask(std::uint64_t(), va, vb)));
        const std::uint64_t amax = a[i + 3], bmax = b[j + 3];
        i += 4 * std::size_t(amax <= bmax);
        j += 4 * std::size_t(bmax <= amax);
        if(count >= min_count) return count;
        const std::size_t ra = na - i, rb = nb - j;
        if(count + (ra < rb ? ra : rb) < min_count) return count;
    }
    return count + detail::isz_merge_count_min(a + i, na - i, b + j, nb - j, min_count - count);
#else
    return detail::isz_merge_count_min(a, na, b, nb, min_count);
#endif
}

// Bounded count over generic sorted containers (pairwise-multiplicity
// semantics, matching isz::intersection_size). Same threshold contract as
// the set-semantics *_min overloads above: (v >= min_count) == (exact >= min_count),
// v <= exact; not the exact count below the threshold.
template<typename Container, typename Cmp=std::less<>>
std::uint64_t intersection_size_min(const Container &c1, const Container &c2,
                                    std::uint64_t min_count, const Cmp &cmp=Cmp()) {
    if(min_count == 0) return intersection_size(c1, c2, cmp);
    auto it1 = std::begin(c1);
    auto it2 = std::begin(c2);
    const auto e1 = std::cend(c1);
    const auto e2 = std::cend(c2);
    const std::uint64_t n1 = c1.size(), n2 = c2.size();
    std::uint64_t i = 0, j = 0, ret = 0;
    while(it1 != e1 && it2 != e2) {
        if(*it1 == *it2) { ++ret; ++it1; ++it2; ++i; ++j; if(ret >= min_count) return ret; }
        else if(cmp(*it1, *it2)) { ++it1; ++i; }
        else { ++it2; ++j; }
        const std::uint64_t r1 = n1 - i, r2 = n2 - j;
        if(ret + (r1 < r2 ? r1 : r2) < min_count) return ret;
    }
    return ret;
}

} // common
} // sketch

#endif
