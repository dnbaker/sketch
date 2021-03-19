// adapted to the lemire/testingRNG project by D. Lemire, from https://github.com/wangyi-fudan/wyhash/blob/master/wyhash.h
// This uses mum hashing.
// Further adapted by D. Baker from https://github.com/lemire/testingRNG/blob/42a3a76feef1126d632f7a56181dacb77ba1ccc7/source/wyhash.h

// XXH3 ported from https://github.com/Cyan4973/xxHash/blob/4229399fc96a034fac522525946f5452d5bf0a65/xxh3.h
#ifndef WYHASH_RNG_H__
#define WYHASH_RNG_H__
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cinttypes>
#include <cstdio>
#include <random>
#include <x86intrin.h>

#ifndef CONST_IF
#  if defined(__cpp_if_constexpr) && __cplusplus >= __cpp_if_constexpr
#    define CONST_IF(...) if constexpr(__VA_ARGS__)
#  else
#    define CONST_IF(...) if(__VA_ARGS__)
#  endif
#endif
#ifndef NO_UNIQUE_ADDRESS
#  if __has_cpp_attribute(no_unique_address)
#    define NO_UNIQUE_ADDRESS [[no_unique_address]]
#  else
#    define NO_UNIQUE_ADDRESS
#    if VERBOSE_AF
#      pragma messsage("no unique address not supported")
#    endif
#  endif
#endif

namespace wy {
using std::uint64_t;
using std::size_t;


static inline
#if __cplusplus >= 201402L
constexpr
#endif
uint64_t _wymum(uint64_t x, uint64_t y) {
    __uint128_t l = x;
    l *= y;
    return l ^ (l >> 64);
}

// call wyhash64_seed before calling wyhash64
static inline
#if __cplusplus >= 201402L
constexpr
#endif
uint64_t wyhash64_stateless(uint64_t *seed) {
  *seed += UINT64_C(0x60bee2bee120fc15);
  return _wymum(*seed ^ 0xe7037ed1a0b428dbull, *seed);
}

struct WyHashFunc {
    static
#if __cplusplus >= 201402L
    constexpr
#endif
    uint64_t apply(uint64_t *x) {
        return wyhash64_stateless(x);
    }
};
#ifdef __AVX512DQ__
static inline __m512i wyhash64_stateless(__m512i *seed) {
  __m512i s1 = _mm512_add_epi64(_mm512_load_si512(seed), _mm512_set1_epi64(0x60bee2bee120fc15uLL));
  _mm512_store_si512(seed, s1);
  __m512i s2 = _mm512_xor_epi64(s1, _mm512_set1_epi64(0xe7037ed1a0b428dbull));
  for(unsigned i = 0; i < sizeof(s1) / sizeof(uint64_t); ++i) {
    ((uint64_t *)&s1)[i] = _wymum(((uint64_t *)&s1)[i], ((uint64_t *)&s2)[i]);
  }
  return s1;
}
#endif

struct XXH3Func {
    static constexpr uint64_t PRIME64_3 =  1609587929392839161ULL;  // 0b0001011001010110011001111011000110011110001101110111100111111001
    static constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
    static
#if __cplusplus >= 201402L
    constexpr
#endif
        uint64_t apply(uint64_t *x) {
        *x += PRIME64_1;
        *x ^= *x >> 29;
        *x *= PRIME64_3;
        *x ^= *x >> 32;
        return *x;
    }
};

template<typename T=std::uint64_t, size_t unroll_count=0, typename HashFunc=WyHashFunc>
class WyRand {
    /*
     * According to our benchmarks on Haswell, 2 is the number providing the fastest results.
     */
public:
    static constexpr size_t UNROLL_COUNT = unroll_count ? unroll_count: size_t(1);
    using result_type = T;
private:
    uint64_t state_;
    uint64_t unrolled_stuff_[UNROLL_COUNT];
    unsigned offset_;
    unsigned &off() {return offset_;}
public:
    using ThisType = WyRand<T, unroll_count, HashFunc>;
    WyRand(uint64_t seed=0): state_(seed ? seed: uint64_t(1337)) {
        std::memset(unrolled_stuff_, 0, sizeof(unrolled_stuff_));
        CONST_IF(unroll_count) off() = sizeof(unrolled_stuff_);
    }
    void seed(uint64_t newseed) {
        state_ = newseed;
        CONST_IF(unroll_count) {
            off() = sizeof(unrolled_stuff_);
        }
    }
    const uint8_t *as_bytes() const {return reinterpret_cast<const uint8_t *>(unrolled_stuff_);}
    uint64_t next_value() {
        return HashFunc().apply(&state_);
    }
    static T constexpr min() {return std::numeric_limits<T>::min();}
    static T constexpr max() {return std::numeric_limits<T>::max();}
    T operator()() {
        return this->generate<T>();
    }
    template<typename OT>
    OT generate() {
        CONST_IF(unroll_count) {
            if(off() + sizeof(OT) > sizeof(unrolled_stuff_)) {
                for(size_t i = 0; i < UNROLL_COUNT; unrolled_stuff_[i++] = next_value());
                off() = 0;
            }
            OT ret;
            std::memcpy(&ret, as_bytes() + off(), sizeof(OT));
            off() += sizeof(OT);
            return ret;
        } else {
            CONST_IF(sizeof(OT) <= sizeof(uint64_t)) {
                return static_cast<OT>(this->next_value());
            } else {
                OT ret;
                size_t offset = 0;
                for(size_t i = 0; i < (sizeof(OT) + sizeof(uint64_t) - 1) / sizeof(uint64_t); ++i) {
                    uint64_t v = next_value();
                    std::memcpy(&ret, &v, std::min(sizeof(v), sizeof(OT) - offset * sizeof(uint64_t)));
                    ++offset;
                }
                return ret;
            }
        }
    }
    template<typename T2=T, bool manual_override=false,
             typename=typename std::enable_if<
                manual_override || std::is_integral<T>::value
                >::type
             >
    class buffer_view {
        ThisType &ref;
    public:
        buffer_view(ThisType &ctr): ref{ctr} {}
        using const_pointer = const T2 *;
        using pointer       = T2 *;
        const_pointer cbegin() const {
            return reinterpret_cast<const_pointer>(&ref.unrolled_stuff_[0]);
        }
        const_pointer cend() const {
            return reinterpret_cast<const_pointer>(&ref.unrolled_stuff_[UNROLL_COUNT]);
        }
        pointer begin() {
            return reinterpret_cast<pointer>(&ref.unrolled_stuff_[0]);
        }
        pointer end() {
            return reinterpret_cast<pointer>(&ref.unrolled_stuff_[UNROLL_COUNT]);
        }
    };
    template<typename T2=T, bool manual_override=false>
    buffer_view<T2, manual_override> view() {return buffer_view<T2, manual_override>(*this);}
};
template<typename T=std::uint64_t, size_t unroll_count=2, typename HashFunc=WyHashFunc>
using WyHash = WyRand<T, unroll_count, HashFunc>; // aliases to fastest generator. May be deprecated.
template<typename T=std::uint64_t, size_t unroll_count=2>
using XX3Rand = WyRand<T, unroll_count, XXH3Func>;


} // namespace wy

#endif /* #ifndef WYHASH_RNG_H__ */
