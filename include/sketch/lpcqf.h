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

#include "sseutil.h"
#include "div.h"

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

INLINE float halfcvt(uint16_t x) {
    return ((x&0x8000)<<16) | (((x&0x7c00)+0x1C000)<<13) | ((x&0x03FF)<<13);
}
INLINE uint16_t hashcvt(float item) {
    uint32_t x;
    std::memcpy(&x, &item, sizeof(float));
    return ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);
}

template<typename BaseT = uint64_t, size_t sigbits=8, int flags = IS_QUADRATIC_PROBING, size_t num = 2, size_t denom = 1, typename ModT=uint32_t>
struct LPCQF {
    using T = typename IntegerSizeEquivalent<BaseT>::type;
    //static constexpr std::ratio<num, denom> approxlogbase;
    static constexpr double approxlogb = double(num) / denom;
    static constexpr size_t countbits = sizeof(T) * CHAR_BIT - sigbits;
    static constexpr size_t countmask = (size_t(1) << countbits) - 1;
    static constexpr bool is_pow2 = flags & IS_POW2;
    static constexpr bool approx_inc = flags & IS_APPROXINC;
    static constexpr bool countsketch_increment = flags & IS_COUNTSKETCH;
    static constexpr bool quadratic_probing = flags & IS_QUADRATIC_PROBING;
    static constexpr bool is_floating = std::is_floating_point_v<BaseT>;
    using MyType = LPCQF<BaseT, sigbits, flags, num, denom, ModT>;
    static_assert(sizeof(T) * CHAR_BIT > sigbits, "T must be >= sigbits size");
    static_assert(countbits < sizeof(T) * CHAR_BIT, "T must be >= countbits size");
    static_assert(approxlogb > 1., "approxlogb must be > 1");
    static_assert(!is_floating || (sigbits == 16 || sigbits == 32), "Floating needs 16 or 32-bit remainders for counting.");
private:
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
    LPCQF(size_t nregs): div_(nregs) {
        if(is_pow2) {
            if(nregs & (nregs - 1)) throw std::invalid_argument("LPCQF of power of 2 size requires a power-of-two size.");
            l2n = 64 - __builtin_clzll(nregs) - 1;
            bitmask = 0xFFFFFFFFFFFFFFFFull >> (64 - l2n);
        }
        data_.resize(nregs);
        size_ = nregs;
        if(is_floating) {
            std::fprintf(stderr, "Error: Floating-point support does not work yet. Do not use.\n");
        }
    }
    std::conditional_t<is_floating, double, uint64_t> inner_product(const MyType &o) {
        if(size_ != o.size_) throw std::invalid_argument("Can't compare LPCQF of different sizes");
        std::conditional_t<is_floating, double, uint64_t> ret = 0;
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
                float v;
                uint32_t tmp = x & countmask;
                std::memcpy(&v, &tmp, sizeof(uint32_t));
                return v;
            } else if constexpr(sigbits == 16) {
                return halfcvt(uint16_t(x & countmask));
            } else {
                throw std::runtime_error("sigbits should be 32 or 16 if floating point sizes are used.");
            }
        }
        return BaseT(0);
    }
    BaseT count_estimate(uint64_t item) const {
        uint64_t hv = hash(item);
        ModT hi = is_pow2 ? ModT(hv & bitmask): ModT(div_.mod(hv));
        const ModT sig = hv & ModT((1ull << sigbits) - 1);
        size_t step = 0;
        size_t stepnum = -1;
        for(;++stepnum != data_.size();) {
            const T &reg = data_[hi];
            if(!reg)
                return static_cast<BaseT>(0);
            if((reg >> countbits) == sig) {
                return extract_res(reg);
            }
            if(quadratic_probing) hi += ++step;
            else                ++hi;
        }
        return static_cast<BaseT>(0);
    }
    static constexpr T encode_res(BaseT val) {
        if constexpr(!is_floating) {return val;}
        else {
            uint32_t ret = 0;
            if constexpr(sigbits == 16) {ret = halfcvt(float(val));}
            else if constexpr(sigbits == 32) {
                float t = val;
                std::memcpy(&ret, &t, sizeof(ret));
            } else {throw std::runtime_error("Should not happen.");}
            return ret;
        }
    }
    template<typename CT, typename=std::enable_if_t<std::is_integral_v<CT>>>
    void update(uint64_t item, CT count) {
        if constexpr(is_floating) {
            std::fprintf(stderr, "Error: Floating-point support does not work yet. Do not use.");
        }
        static_assert(std::is_signed_v<CT> || !countsketch_increment, "Make sure the type is signed or not countsketch");
        uint64_t hv = hash(item);
        auto hi = is_pow2 ? ModT(hv & bitmask): ModT(div_.mod(hv));
        const ModT sig = hv & ModT((1ull << sigbits) - 1);
        size_t step = 0;
        size_t stepnum = -1;
        for(;++stepnum < data_.size();) {
            if(!data_[hi]) {
                T val = encode_res(count);
                data_[hi] = (T(sig) << countbits) | val;
                return;
            } else if(auto osig = (data_[hi] >> countbits); osig == sig) {
                if constexpr (approx_inc) { throw std::runtime_error("Not implemented.");}
                else if constexpr(countsketch_increment) {
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
            if constexpr(quadratic_probing) {
                hi += ++step;
            } else {
                ++hi;
            }
            assert( ModT(hi & bitmask) == ModT(div_.mod(hi)));
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
        update(item, 1);
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
