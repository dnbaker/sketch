#ifndef LP_CQF_H__
#define LP_CQF_H__
#include <ratio>
#include <climits>
#include <vector>
#include <x86intrin.h>
#include "sseutil.h"
#include <stdexcept>

namespace sketch {

template<typename T>
struct IntegerSizeEquivalent {
    using IT = std::conditional_t<sizeof(T) == 4, uint32_t,
                   std::conditional_t<sizeof(T) == 8, uint64_t,
                       std::conditional_t<sizeof(T) == 2, uint16_t,
                           std::conditional_t<sizeof(T) == 1, uint8_t,
                               std::conditional_t<sizeof(T) == 16, __uint128_t, std::nullptr_t>>>>>;
    static_assert(!std::is_same_v<IT, std::nullptr_t>, "IntegerSizeEquivalent requires an object of 1,2,4,8, or 16 bytes");
};

enum LPCQFFlags {
    IS_POW2 = 1,
    IS_FLOAT = 2,
    IS_APPROXINC = 4,
    IS_COUNTSKETCH = 8
};

template<typename BaseT = uint64_t, size_t sigbits=8, int flags = 0, size_t num = 2, size_t denom = 1>
struct LPCQF {
    static constexpr std::ratio<num, denom> approxlogbase;
    static constexpr double approxlogb = double(num) / denom;
    static constexpr size_t countbits = sizeof(BaseT) * CHAR_BIT - sigbits;
    static constexpr bool is_pow2 = flags & IS_POW2;
    static constexpr bool is_float = flags & IS_FLOAT;
    static constexpr bool approx_inc = flags & IS_APPROXINC;
    static constexpr bool counsketch_increment = flags & IS_COUNTSKETCH;
    static_assert(sizeof(BaseT) * CHAR_BIT > sigbits, "BaseT must be >= sigbits size");
    static_assert(countbits < sizeof(BaseT) * CHAR_BIT, "BaseT must be >= countbits size");
    static_assert(approxlogb > 1., "approxlogb must be > 1");
private:
    std::vector<BaseT, sse::AlignedAllocator<BaseT>> data_;
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
    LPCQF(size_t nregs) {
        if(is_pow2) {
            if(nregs & (nregs - 1)) throw std::invalid_argument("LPCQF of power of 2 size requires a power-of-two size.");
        } else {
            
        }
        data_.resize(nregs);
    }
    template<typename T, typename=std::enable_if_t<std::is_integral_v<T>>>
    void update(uint64_t item, T count) {
        item = hash(item);
    }
    void update(uint64_t item) {
        update(item, 1u);
    }
#if __AVX2__
    INLINE __m256i operator()(__m256i element) const {
        __m256i key = _mm256_add_epi64(_mm256_slli_epi64(element, 21), ~element);
        key = _mm256_srli_epi64(key, 24) ^ key;
        key = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64(key, 3), _mm256_slli_epi64(key, 8)), key);
        key = _mm256_xor_si256(key, __mm256_srli_epi64(key, 14));
        key = _mm256_add_epi64(
                _mm256_add_epi64(_mm256_slli_epi64(key, 2), _mm256_slli_epi64(key, 4)), key);
        key = _mm256_xor_si256(__mm256_srli_epi64(key, 28));
        key = _mm256_add_epi64(_mm256_slli_epi64(key, 31), key);
        return key;
    }
#endif
#if __AVX512F__
    INLINE __m512i operator()(__m512i element) const {
        __m512i key = _mm512_add_epi64(_mm512_slli_epi64(element, 21), ~element);
        key = _mm512_srli_epi64(key, 24) ^ key;
        key = _mm512_add_epi64(_mm512_add_epi64(_mm512_slli_epi64(key, 3), _mm512_slli_epi64(key, 8)), key);
        key = _mm512_xor_si512(key, __mm512_srli_epi64(key, 14));
        key = _mm512_add_epi64(
                _mm512_add_epi64(_mm512_slli_epi64(key, 2), _mm512_slli_epi64(key, 4)), key);
        key = _mm512_xor_si512(__mm512_srli_epi64(key, 28));
        key = _mm512_add_epi64(_mm512_slli_epi64(key, 31), key);
        return key;
    }
#endif
};

}

#endif
