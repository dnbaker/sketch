#ifndef _HLL_H_
#define _HLL_H_
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

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)
# include "x86intrin.h"
#define HAS_AVX_512 _FEATURE_AVX512F
#endif

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

static INLINE uint64_t roundup64(size_t x) {
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
#ifndef NAVOID_CLZ_UNDEF
static INLINE unsigned clz(unsigned long long x) {
    return x ? __builtin_clzll(x) : sizeof(x) * CHAR_BIT;
}

static INLINE unsigned clz(unsigned long x) {
    return x ? __builtin_clzl(x) : sizeof(x) * CHAR_BIT;
}
#else
static INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}

static INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
#endif

constexpr double make_alpha(size_t m) {
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
    size_t np_;
    const size_t m_;
    double alpha_;
    double relative_error_;
#if HAS_AVX_512
#if !NDEBUG
#pragma message("Building with avx512")
#endif
    std::vector<uint8_t, sse::AlignedAllocator<uint8_t, sse::Alignment::AVX512>> core_;
#elif __AVX2__
#if !NDEBUG
#pragma message("Building with avx2")
#endif
    std::vector<uint8_t, sse::AlignedAllocator<uint8_t, sse::Alignment::AVX>> core_;
#elif __SSE2__
#if !NDEBUG
#pragma message("Building with sse2")
#endif
    std::vector<uint8_t, sse::AlignedAllocator<uint8_t, sse::Alignment::SSE>> core_;
#else
    std::vector<uint8_t> core_;
#endif
    double sum_;
    int is_calculated_;

public:
    // Constructor
    hll_t(size_t np=20): np_(np), m_(1uL << np), alpha_(make_alpha(m_)),
                         relative_error_(1.03896 / std::sqrt(m_)),
                         core_(m_, 0),
                         sum_(0.), is_calculated_(0) {}

    // Call sum to recalculate if you have changed contents.
    void sum() {
        sum_ = 0;
        for(unsigned i(0); i < m_; ++i) sum_ += 1. / (1ull << core_[i]);
        is_calculated_ = 1;
    }

    // Returns cardinality estimate. Sums if not calculated yet.
    double report() {
        if(!is_calculated_) sum();
        const double ret(alpha_ * m_ * m_ / sum_);
        // Correct for small values
        if(ret < m_ * 2.5) {
            int t(0);
            for(unsigned i(0); i < m_; ++i) t += (core_[i] == 0);
            if(t) return m_ * std::log((double)(m_) / t);
        }
        return ret;
        // We don't correct for too large just yet, but we should soon.
    }

    // Returns the size of a symmetric set difference.
    double operator^(hll_t &other) {
        hll_t tmp(*this);
        tmp += other;
        tmp.sum();
        return std::abs(report() - other.report()) - tmp.report();
    }

    // Returns error estimate
    double est_err() {
        return relative_error_ * report();
    }

    INLINE void add(uint64_t hashval) {
        const uint32_t index(hashval >> (64ull - np_));
        const uint32_t lzt(clz(hashval << np_) + 1);
        if(core_[index] < lzt) core_[index] = lzt;
    }

    INLINE void addh(uint64_t element) {
        add(wang_hash(element));
    }

    std::string to_string() {
        return std::to_string(report()) + ", +- " + std::to_string(est_err());
    }

    // Reset.
    void clear() {
         std::fill(core_.begin(), core_.end(), 0u);
         sum_ = is_calculated_ = 0;
    }

    // Assignment Operators
    hll_t &operator=(const hll_t &other) {
        np_ = other.np_;
        core_ = other.core_;
        alpha_ = other.alpha_;
        sum_ = other.sum_;
        relative_error_ = other.relative_error_;
        memcpy((void *)&m_, &other.m_, sizeof(m_)); // Memcpy const
        return *this;
    }

    hll_t &operator=(hll_t &&other) {
        np_ = other.np_;
        memcpy((void *)&m_, &other.m_, sizeof(m_)); // Memcpy const
        alpha_ = other.alpha_;
        relative_error_ = other.relative_error_;
        core_ = std::move(other.core_);
        is_calculated_ = other.is_calculated_;
        sum_ = other.sum_;
        return *this;
    }

    hll_t(const hll_t &other): hll_t(other.m_) {
        *this = other;
    }

    hll_t(hll_t &&other):
        np_(other.np_),
        m_(other.m_),
        alpha_(other.alpha_),
        relative_error_(other.relative_error_),
        core_(std::move(other.core_)),
        sum_(other.sum_),
        is_calculated_(other.is_calculated_) {
    }

    hll_t const &operator+=(const hll_t &other) {
        if(other.np_ != np_)
            LOG_EXIT("np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
#if HAS_AVX_512
        unsigned i;
        __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
        const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
        for(i = 0; i < m_ >> 6; ++i) els[i] = _mm512_or_epi64(els[i], oels[i]);
        if(m_ < 64) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300) && __AVX2__
        unsigned i;
        __m256 *els(reinterpret_cast<__m256 *>(core_.data()));
        const __m256 *oels(reinterpret_cast<const __m256 *>(other.core_.data()));
        for(i = 0; i < m_ >> 5; ++i) els[i] = _mm256_or_si256(els[i], oels[i]);
        if(m_ < 32) for(;i < m_; ++i) core_[i] |= other.core_[i];
#elif defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300) && __SSE2__
        unsigned i;
        __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
        const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
        for(i = 0; i < m_ >> 4; ++i) els[i] = _mm_or_si128(els[i], oels[i]);
        if(m_ < 16) for(; i < m_; ++i) core_[i] |= other.core_[i];
#else
        for(unsigned i(0); i < m_; ++i) core_[i] |= other.core_[i];
#endif
        return *this;
    }

    hll_t operator+(const hll_t &other) const {
        if(other.np_ != np_)
            LOG_EXIT("np_ (%zu) != other.np_ (%zu)\n", np_, other.np_);
        hll_t ret(*this);
        return ret += other;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup64(new_size);
        LOG_DEBUG("Resizing to %zu, with np = %zu\n", new_size, (size_t)std::log2(new_size));
        clear();
        core_.resize(new_size);
    }
    // Getter for is_calculated_
    bool is_ready() {
        return is_calculated_;
    }
};



} // namespace hll

#endif // #ifndef _HLL_H_
