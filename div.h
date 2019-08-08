#ifndef SKETCH_DIV_H__
#define SKETCH_DIV_H__

#include <cstdint>
#include <utility>
#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#ifndef CONST_IF
#if __cplusplus >= 201703L
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
  __uint128_t M = UINT64_C(0xFFFFFFFFFFFFFFFF);
  M <<= 64;
  M |= UINT64_C(0xFFFFFFFFFFFFFFFF);
  M /= d;
  M += 1;
  return M;
}

static inline uint64_t mul128_u64(__uint128_t lowbits, uint64_t d) {
  __uint128_t bottom_half = (lowbits & UINT64_C(0xFFFFFFFFFFFFFFFF)) * d; // Won't overflow
  bottom_half >>= 64;  // Only need the top 64 bits, as we'll shift the lower half away;
  __uint128_t top_half = (lowbits >> 64) * d;
  __uint128_t both_halves = bottom_half + top_half; // Both halves are already shifted down by 64
  both_halves >>= 64; // Get top half of both_halves
  return (uint64_t)both_halves;
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
    std::array<uint64_t, shortcircuit ? 1: 0> m32_;
    uint64_t &m32() {assert(shortcircuit); return m32_[0];}
    // We swap location here so that m32 can be 64-bit aligned.
public:
    const auto &d() const {return d_;}
    const uint64_t &m32() const {assert(shortcircuit); return m32_[0];}
    using DivType = div_t<uint64_t>;
    Schismatic(uint64_t d): d_(d), M_(computeM_u64(d)) {
        CONST_IF(shortcircuit) m32() = computeM_u32(d);
    }
    INLINE bool test_limits(uint64_t v) const {
        assert(shortcircuit);
        return d_ >> 32 ? (v >> 32) == 0: false;
    }
    INLINE uint64_t div(uint64_t v) const {
        CONST_IF(shortcircuit)
            return test_limits(v) ? uint64_t(fastdiv_u32(v, m32())): fastdiv_u64(v, m32());
        return fastdiv_u64(v, M_);
    }
    INLINE uint64_t mod(uint64_t v) const {
        CONST_IF(shortcircuit)
            return test_limits(v) ? uint64_t(fastmod_u32(v, m32(), d_)): fastmod_u64(v, m32(), d_);
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
template<> struct Schismatic<int32_t>: Schismatic<uint32_t> {};
template<> struct Schismatic<int64_t>: Schismatic<uint64_t> {};

} // namespace schism

#endif /* SKETCH_DIV_H__  */
