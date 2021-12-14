#ifndef LP_CQF_H__
#define LP_CQF_H__
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <ratio>
#include <vector>

#include <x86intrin.h>

#include "aesctr/wy.h"
#include "sketch/sseutil.h"
#include "sketch/div.h"
#include "sketch/macros.h"

#ifndef INLINE
#define INLINE __attribute__((always_inline)) inline
#endif

namespace sketch {

template<typename T>
struct IntegerSizeEquivalent {
    using type = std::conditional_t<sizeof(T) == 4, uint32_t,
                   std::conditional_t<sizeof(T) == 8, uint64_t,
                       std::conditional_t<sizeof(T) == 2, uint16_t,
                           std::conditional_t<sizeof(T) == 1, uint8_t,
                               std::conditional_t<sizeof(T) == 16, __uint128_t, std::nullptr_t>>>>>;
    static_assert(!std::is_same_v<type, std::nullptr_t>, "IntegerSizeEquivalent requires an object of 1,2,4,8, or 16 bytes");
};

enum LPCQFFlags {
    IS_POW2 = 1,
    IS_APPROXINC = 2,
    IS_COUNTSKETCH = 4,
    IS_QUADRATIC_PROBING = 8
};


static INLINE uint32_t as_uint(const float x) {
    uint32_t ret;
    std::memcpy(&ret, &x, sizeof(ret));
    return ret;
}
static INLINE float as_float(const uint32_t x) {
    float ret;
    std::memcpy(&ret, &x, sizeof(ret));
    return ret;
}


// Approximate counting using math from
// Optimal bounds for approximate counting, Jelani Nelson, Huacheng Yu
// 2020.
// https://arxiv.org/abs/2010.02116

// Convert float/half and half/float using intrinsics if available
// and manually otherwise. 5-10x performance penalty for manual.

INLINE float half_to_float(const uint16_t x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    float ret;
#if __F16C__ && __SSE__
    ret = _mm_cvtph_ps(_mm_set1_epi16(x))[0];
#else
    const uint32_t e = (x&0x7C00)>>10; // exponent
    const uint32_t m = (x&0x03FF)<<13; // mantissa
    const uint32_t v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
    ret = as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000))); // sign : normalized : denormalized
#endif
    return ret;
}

#if __F16C__ && __SSE__
template<int ROUND=0>
#endif
INLINE uint16_t float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    uint16_t ret;
#if (__F16C__ && __SSE__)
    const auto ph = _mm_cvtps_ph(_mm_set1_ps(x), ROUND);
    std::memcpy(&ret, &ph, sizeof(ret));
#else
    const uint32_t b = as_uint(x)+0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
    const uint32_t e = (b&0x7F800000)>>23; // exponent
    const uint32_t m = b&0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
    ret = (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; // sign : normalized : denormalized : saturate
#endif
    return ret;
}

template<typename T, typename=std::enable_if_t<!std::is_same_v<float, T> && !std::is_same_v<uint16_t, T>>>
T halfcvt(T x) {throw std::runtime_error("Used the wrong halfcvt; this should only be run on float and uint16_t");}

template<size_t num, size_t denom, size_t N = 64>
constexpr std::array<long double, N> get_ipowers() {
    std::array<long double, N> ret{1.L};
    constexpr long double approxlogb = static_cast<long double>(num) / denom;
    for(size_t i = 0;++i < N;) {
        ret[i] = ret[i - 1] / approxlogb;
    }
    return ret;
}
template<size_t num, size_t denom, size_t N>
constexpr std::array<long double, N> POWERS = get_ipowers<num, denom, N>();

template<typename BaseT = uint64_t, size_t sigbits=8, int flags = IS_QUADRATIC_PROBING, size_t argnum = 2, size_t argdenom = 1, typename ModT=uint32_t, long long int NCACHED_POWERS=64>
struct LPCQF {
    using T = typename IntegerSizeEquivalent<BaseT>::type;
    using MyType = LPCQF<BaseT, sigbits, flags, argnum, argdenom, ModT>;
    //using Ratio = std::ratio<argnum, argdenom>;
    static constexpr std::ratio<argnum, argdenom> ratio = std::ratio<argnum, argdenom> ();
    static constexpr size_t num = ratio.num;
    static constexpr size_t denom = ratio.den;
    static constexpr long double approxlogb = static_cast<long double>(num) / denom;
    static constexpr size_t countbits = sizeof(T) * CHAR_BIT - sigbits;
    static constexpr size_t countmask = (size_t(1) << countbits) - 1;
    static constexpr bool is_pow2 = flags & IS_POW2;
    static constexpr bool approx_inc = flags & IS_APPROXINC;
    static constexpr bool countsketch_increment = flags & IS_COUNTSKETCH;
    static constexpr bool quadratic_probing = flags & IS_QUADRATIC_PROBING;
    static constexpr bool is_floating = std::is_floating_point_v<BaseT>;
    static constexpr bool is_apow2 = num == 2 && denom == 1;
private:
    // Helper functions for approximate increment
    static long double ainc_increment_prob(signed long long n) {
        long double ret;
        if(n < NCACHED_POWERS) ret = POWERS<num, denom, NCACHED_POWERS>[n];
        else ret = std::pow(approxlogb, -n);
        return ret;
    }
    static long double ainc_estimate_count(signed long long n) {
        if constexpr(num == 2 && denom == 1) {
            if(n < 64) return (1ull << n) - 1ull;
            return std::ldexp(1., n) - 1.L;
        } else {
            const long double v = n < NCACHED_POWERS ? POWERS<denom, num, NCACHED_POWERS>[n]: std::pow(approxlogb, n);
            return (v - 1.L) / (approxlogb - 1.L);
        }
    }
    template<typename T>
    void ainc(T &counter) {
        if constexpr(is_apow2) {
            if(numdraws < counter) {
                rv = wy::wyhash64_stateless(&rseed);
                numdraws = 64;
            }
            const T old = counter;
            if((rv & (UINT64_C(-1) >> (64 - counter))) == 0)
                ++counter;
            rv >>= old; numdraws -= old;
        } else {
            if(static_cast<long double>(wy::wyhash64_stateless(&rseed)) * 0x1p-64L < ainc_increment_prob(counter))
                ++counter;
        }
    }

    static_assert(sizeof(T) * CHAR_BIT > sigbits, "T must be >= sigbits size");
    static_assert(countbits <= sizeof(T) * CHAR_BIT, "T must be >= countbits size");
    static_assert(approxlogb > 1., "approxlogb must be > 1");
    static_assert(!is_floating || (sigbits == 16 || sigbits == 32), "Floating needs 16 or 32-bit remainders for counting.");
    static_assert(!approx_inc || (!is_floating && !countsketch_increment), "Approximate increment cannot use floating-point representations or count-sketch incrementing.");
    uint64_t rv;
    uint64_t rseed = 13;
    unsigned int numdraws = 0;
    schism::Schismatic<ModT> div_;
    //std::unique_ptr<MyType> leftovers_;

    std::vector<T, sse::AlignedAllocator<T>> data_;
    int l2n = 0;
    uint64_t bitmask = 0xFFFFFFFFFFFFFFFFull;
    size_t size_;

public:
    static constexpr auto hash(uint64_t key) {
          key = (~key) + (key << 21); // key = (key << 21) - key - 1;
          key = key ^ (key >> 24);
          key = (key + (key << 3)) + (key << 8); // key * 265
          key = key ^ (key >> 14);
          key = (key + (key << 2)) + (key << 4); // key * 21
          key = key ^ (key >> 28);
          key = key + (key << 31);
          return key;
    }
    LPCQF(size_t nregs, size_t seed=0): div_(nregs) {
        rseed = seed ? seed: nregs;
        if(nregs > std::numeric_limits<ModT>::max()) throw std::invalid_argument(std::string("nregs ") + std::to_string(nregs) + " is > than ModT size. Use a 64-bit ModT to build an LPCQF of that size.");
        if(is_pow2) {
            if(nregs & (nregs - 1)) throw std::invalid_argument("LPCQF of power of 2 size requires a power-of-two size.");
            l2n = 64 - __builtin_clzll(nregs) - 1;
            bitmask = 0xFFFFFFFFFFFFFFFFull >> (64 - l2n);
        }
        data_.resize(nregs);
        size_ = nregs;
    }
    std::conditional_t<is_floating, double, uint64_t> inner_product(const MyType &o) {
        if(size_ != o.size_) throw std::invalid_argument("Can't compare LPCQF of different sizes");
        std::conditional_t<is_floating, double, uint64_t> ret = 0;
        if constexpr(approx_inc) throw std::invalid_argument("Not yet implemented: approx_inc inner product.");
        // TODO: SIMD Implementation
        for(size_t i = 0; i < size_; ++i) {
            if(data_[i] && o.data_[i]) {
                T lhsig = data_[i] >> countbits;
                T rhsig = o.data_[i] >> countbits;
                if(lhsig == rhsig) {
                    ret = std::fma(extract_res(data_[i]), extract_res(o.data_[i]), ret);
                }
            }
        }
        return ret;
    }
    INLINE BaseT extract_res(T x) const {
        if constexpr(!is_floating) {
            return x & countmask;
        } else {
            if constexpr(sigbits == 32) {
                //std::fprintf(stderr, "Integer %u becomes %g\n", int(x & countmask), as_float(x & countmask));
                return as_float(x & countmask);
            } else if constexpr(sigbits == 16) {
                //std::fprintf(stderr, "Integer %u becomes %g\n", int(x & countmask), half_to_float(x & countmask));
                return half_to_float(x & countmask);
            } else {
                throw std::runtime_error("sigbits should be 32 or 16 if floating point sizes are used.");
            }
        }
        return BaseT(0);
    }
    std::conditional_t<approx_inc, long double, BaseT> count_estimate(uint64_t item) const {
        uint64_t hv = hash(item);
        ModT hi = is_pow2 ? ModT(hv & bitmask): ModT(div_.mod(hv));
        const ModT sig = sigbits ? hv & ModT((1ull << sigbits) - 1): ModT(0);
        ModT osig;
        size_t step = 0;
        size_t stepnum = -1;
        for(;++stepnum != data_.size();) {
            const T &reg = data_[hi];
            if(!reg) {
#ifndef VERBOSE_AF
                std::fprintf(stderr, "Item %zu was not found in the table, returning 0.\n", size_t(item));
#endif
                return 0.;
            }
            if constexpr(sigbits > 0) osig = reg >> countbits;
            if(sigbits == 0|| osig == sig) {
                BaseT ret = extract_res(reg);
                if constexpr(approx_inc)
                    return ainc_estimate_count(ret);
                return ret;
            }
            if(quadratic_probing) hi += ++step;
            else                ++hi;
            hi = is_pow2 ? ModT(hi & bitmask): ModT(div_.mod(hi));
        }
        return 0.L;
    }
    static constexpr T encode_res(BaseT val) {
        if constexpr(!is_floating) {return val;}
        else {
            uint32_t ret;
            if constexpr(sigbits == 16) {
                ret = float_to_half(float(val));
            }
            else if constexpr(sigbits == 32) {
                float t = val;
                std::memcpy(&ret, &t, sizeof(ret));
            } else {throw std::runtime_error("Should not happen.");}
            return ret;
        }
    }
    template<typename CT, typename=std::enable_if_t<std::is_arithmetic_v<CT>>>
    void update(uint64_t item, CT count) {
        if(approx_inc && unlikely(count < CT(0))) throw std::invalid_argument(std::string("Update with negative count ") + std::to_string(count) + " is not permitted in approximate counting mode.");
        static_assert(std::is_signed_v<CT> || !countsketch_increment, "Make sure the type is signed or not countsketch");
        uint64_t hv = hash(item);
        auto hi = is_pow2 ? ModT(hv & bitmask): ModT(div_.mod(hv));
        T sig, osig;
        if constexpr(sigbits > 0) sig = hv & ModT((1ull << sigbits) - 1);
        size_t step = 0;
        size_t stepnum = -1;
        for(;++stepnum < data_.size();) {
            if(!data_[hi]) {
                if constexpr(approx_inc) {
                    T insert = 1;
                    for(size_t i = 1; i < static_cast<size_t>(count); ainc(insert), ++i);
                    data_[hi] = (T(sig) << countbits) | insert;
                    assert(data_[hi] != T(0));
                } else {
                    T val = encode_res(count);
                    if constexpr(sigbits)
                        val |= (T(sig) << countbits);
                    data_[hi] = val;
                }
                return;
            } else {
                if constexpr(sigbits > 0) osig = data_[hi] >> countbits;
                if(sigbits == 0 || osig == sig) {
                    if constexpr (approx_inc) {
                        T current_count = data_[hi] & countmask;
                        if(current_count >= ((1ull << countbits) - 1)) return; // Saturated
                        for(size_t i = 0; i < size_t(count) && likely(current_count != ((1ull << countbits) - 1)); ++i)
                            ainc(current_count);
                        data_[hi] = (sig << countbits) | current_count;
                    } else if constexpr(countsketch_increment) {
                        const bool flip_sign = hv >> 63;
                        if constexpr(is_floating) {
                            data_[hi] = (osig << countbits) | encode_res(extract_res(data_[hi]) + (flip_sign ? count: -count));
                        } else {
                            T newval = (data_[hi] & countmask) + (flip_sign ? count: -count);
                            data_[hi] = (osig << countbits) | newval;
                        }
                    } else {
                        if constexpr(is_floating) {
                            data_[hi] = (osig << countbits) | encode_res(extract_res(data_[hi]) + count);
                        } else {
                            data_[hi] += count;
                        }
                    }
                    return;
                }
            }
            if constexpr(quadratic_probing) {
                hi += ++step;
            } else {
                ++hi;
            }
            if constexpr(is_pow2) {
                assert(ModT(hi & bitmask) == ModT(div_.mod(hi)));
            }
            hi = is_pow2 ? ModT(hi & bitmask): ModT(div_.mod(hi));
        }

        std::fprintf(stderr, "Failed to find empty bucket in table of size %zu\n", data_.size());
        throw std::runtime_error("CQF exceeded size. Resizing not yet implemented.");
    }
    template<typename F>
    void for_each(F &&f) {
        for(auto &v: data_) {
            if(!v) continue; // Skip empty buckets
            const T rem = v >> countbits;
            if constexpr(is_floating) {
                BaseT countv;
                T rv = v & countmask;
                std::memcpy(&countv, &rv, sizeof(T));
                f(rem, countv);
            } else {
                f(rem, v & countmask);
            }
        }
    }
    void update(uint64_t item) {
        static constexpr std::conditional_t<is_floating, double, std::conditional_t<countsketch_increment, std::make_signed_t<T>, std::make_unsigned_t<T>>> inc = 1;
        update(item, inc);
    }
    void reset() {
        std::fill_n(data_.data(), size_, T(0));
    }
#if __AVX2__
    static INLINE __m256i hash(__m256i element) {
        __m256i key = _mm256_add_epi64(_mm256_slli_epi64(element, 21), ~element);
        key = _mm256_srli_epi64(key, 24) ^ key;
        key = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64(key, 3), _mm256_slli_epi64(key, 8)), key);
        key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 14));
        key = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_slli_epi64(key, 2), _mm256_slli_epi64(key, 4)), key);
        key = _mm256_xor_si256(key, _mm256_srli_epi64(key, 28));
        key = _mm256_add_epi64(_mm256_slli_epi64(key, 31), key);
        return key;
    }
#endif
#if __AVX512F__
    static INLINE __m512i hash(__m512i element) {
        __m512i key = _mm512_add_epi64(_mm512_slli_epi64(element, 21), ~element);
        key = _mm512_srli_epi64(key, 24) ^ key;
        key = _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64(key, 3), _mm512_slli_epi64(key, 8)), key);
        key = _mm512_xor_si512(key, _mm512_srli_epi64(key, 14));
        key = _mm512_add_epi64(
                _mm512_add_epi64(_mm512_slli_epi64(key, 2), _mm512_slli_epi64(key, 4)), key);
        key = _mm512_xor_si512(key, _mm512_srli_epi64(key, 28));
        key = _mm512_add_epi64(_mm512_slli_epi64(key, 31), key);
        return key;
    }
#endif
};

}

#endif
