#ifndef SKETCH_DIV_H__
#define SKETCH_DIV_H__

#include <cstdint>
#include <cassert>
#include <utility>
#include <limits>

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
    const uint32_t d_;
    const uint64_t M_;
    Schismatic(uint32_t d): d_(d), M_(computeM_u32(d)) {}
    auto d() const {return d_;}
    INLINE uint32_t div(uint32_t v) const {return fastdiv_u32(v, M_);}
    INLINE uint32_t mod(uint32_t v) const {return fastmod_u32(v, M_, d_);}
    INLINE div_t<uint32_t> divmod(uint32_t v) const {
        auto tmpd = div(v);
        return div_t<uint32_t> {tmpd, v - d_ * tmpd};
    }
};
template<> struct Schismatic<int32_t>: Schismatic<uint32_t> {
    template<typename...Args>Schismatic<int32_t>(Args &&...args):
        Schismatic<uint32_t>(std::forward<Args>(args)...){}
};
template<> struct Schismatic<int64_t>: Schismatic<uint64_t> {
    template<typename...Args>Schismatic<int64_t>(Args &&...args):
        Schismatic<uint64_t>(std::forward<Args>(args)...){}
};

} // namespace schism

#endif /* SKETCH_DIV_H__  */
