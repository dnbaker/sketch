#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "sseutil.h"
#include "math.h"
#include "unistd.h"
#include "x86intrin.h"
#include "kthread.h"
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif
#include "libpopcnt/libpopcnt.h"
#include "compact_vector/include/compact_vector.hpp"
#ifndef _VEC_H__
#  define NO_SLEEF
#  define NO_BLAZE
#  include "vec.h" // Import vec.h, but disable blaze and sleef.
#endif

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__ || __AVX512__)
#endif

#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH 1
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if defined(NDEBUG)
#  if NDEBUG == 0
#    undef NDEBUG
#  endif
#endif

#ifndef FOREVER
#  define FOREVER for(;;)
#endif
#ifndef ASSERT_INT_T
#  define ASSERT_INT_T(T) typename=typename ::std::enable_if<::std::is_integral<(T)>::value>::type
#endif

#if __has_cpp_attribute(no_unique_address)
#  define NO_ADDRESS [[no_unique_address]]
#else
#  define NO_ADDRESS
#endif

#if __cplusplus >= 201703L
#define CONST_IF if constexpr
#else
#define CONST_IF if
#endif

namespace sketch {
namespace common {

template<typename BloomType>
inline double jaccard_index(const BloomType &h1, const BloomType &h2) {
    return h1.jaccard_index(h2);
}
template<typename BloomType>
inline double jaccard_index(BloomType &h1, BloomType &h2) {
    return h1.jaccard_index(h2);
}

using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::size_t;
using Space = vec::SIMDTypes<uint64_t>;
using Type  = typename vec::SIMDTypes<uint64_t>::Type;
using VType = typename vec::SIMDTypes<uint64_t>::VType;
#if HAS_AVX_512
static_assert(sizeof(Space::VType) == sizeof(__m512i), "Must be the right size");
#endif

template<typename ValueType>
#if HAS_AVX_512
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX512>;
#elif __AVX2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX>;
#elif __SSE2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::SSE>;
#else
using Allocator = std::allocator<ValueType, sse::Alignment::Normal>;
#endif

#ifdef NOT_THREADSAFE
using DefaultCompactVectorType = ::compact::vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;
#else
using DefaultCompactVectorType = ::compact::ts_vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;
#endif

template<typename T>
static INLINE T roundup(T x) noexcept {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    if(sizeof(x) > 1)
        x |= x >> 8;
    if(sizeof(x) > 2)
        x |= x >> 16;
    if(sizeof(x) > 4)
        x |= x >> 32;
    return ++x;
}

// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It a bijection within [0,1<<64)
// and can be inverted with irving_inv_hash.
struct WangHash {
    INLINE auto operator()(uint64_t key) const {
          key = (~key) + (key << 21); // key = (key << 21) - key - 1;
          key = key ^ (key >> 24);
          key = (key + (key << 3)) + (key << 8); // key * 265
          key = key ^ (key >> 14);
          key = (key + (key << 2)) + (key << 4); // key * 21
          key = key ^ (key >> 28);
          key = key + (key << 31);
          return key;
    }
    INLINE uint32_t operator()(uint32_t key) const {
        key += ~(key << 15);
        key ^=  (key >> 10);
        key +=  (key << 3);
        key ^=  (key >> 6);
        key += ~(key << 11);
        key ^=  (key >> 16);
        return key;
    }
    INLINE Type operator()(Type element) const {
        VType key = Space::add(Space::slli(element, 21), ~element); // key = (~key) + (key << 21);
        key = Space::srli(key.simd_, 24) ^ key.simd_; //key ^ (key >> 24)
        key = Space::add(Space::add(Space::slli(key.simd_, 3), Space::slli(key.simd_, 8)), key.simd_); // (key + (key << 3)) + (key << 8);
        key = key.simd_ ^ Space::srli(key.simd_, 14);  // key ^ (key >> 14);
        key = Space::add(Space::add(Space::slli(key.simd_, 2), Space::slli(key.simd_, 4)), key.simd_); // (key + (key << 2)) + (key << 4); // key * 21
        key = key.simd_ ^ Space::srli(key.simd_, 28); // key ^ (key >> 28);
        key = Space::add(Space::slli(key.simd_, 31), key.simd_);    // key + (key << 31);
        return key.simd_;
    }
#if VECTOR_WIDTH > 16
    INLINE auto operator()(__m128i key) const {
        key = _mm_add_epi64(~key, _mm_slli_epi64(key, 21)); // key = (key << 21) - key - 1;
        key ^= _mm_srli_epi64(key, 24);
        key = _mm_add_epi64(key, _mm_add_epi64(_mm_slli_epi64(key, 3), _mm_slli_epi64(key, 8)));
        key ^= _mm_srli_epi64(key, 14);
        key = _mm_add_epi64(_mm_add_epi64(key, _mm_slli_epi64(key, 2)), _mm_slli_epi64(key, 4));
        key ^= _mm_srli_epi64(key, 28);
        key = _mm_add_epi64(_mm_slli_epi64(key, 31), key);
        return key;
    }
#endif
};

template<size_t N>
static std::array<uint64_t, N> make_coefficients(uint64_t seedseed) {
    std::array<uint64_t, N> ret;
    std::mt19937_64 mt(seedseed);
    for(auto &e: ret) e = mt();
    return ret;
}

template<size_t k>
class KWiseIndependentPolynomialHash {
    const std::array<uint64_t, k> coeffs_;
    static constexpr uint64_t mod = 9223372036854775807ull;
public:
    KWiseIndependentPolynomialHash(uint64_t seedseed=137): coeffs_(make_coefficients<k>(seedseed)) {
        static_assert(k, "k must be nonzero");
    }
    uint64_t operator()(uint64_t val) const {
        uint64_t ret = coeffs_[0];
        uint64_t exp = val;
        for(size_t i = 1; i < k; ++i) {
            ret = (ret + (exp * coeffs_[i] % mod)) % mod;
            exp = (val * exp) % mod;
        }
        return ret;
    }
    Type operator()(VType val) const {
        // Data parallel across same coefficients.
        VType ret = Space::set1(coeffs_[0]);
        VType exp = val;
        for(size_t i = 1; i < k; ++i) {
#if HAS_AVX_512
            auto tmp = Space::mullo(exp.simd_, Space::set1(coeffs_[i]));
            tmp.for_each([](auto &x) {x %= mod;});
            ret = Space::add(ret, tmp.simd_);
            ret.for_each([](auto &x) {x %= mod;});
            exp = Space::mullo(exp.simd_, val.simd_);
            exp.for_each([](auto &x) {x %= mod;});
#else
            for(uint32_t j = 0; j < Space::COUNT; ++j) {
                ret.arr_[j] = (ret.arr_[j] + (exp.arr_[j] * coeffs_[i] % mod)) % mod;
                exp.arr_[j] = (exp.arr_[j] * val.arr_[j]) % mod;
            }
#endif
        }
        return ret.simd_;
    }
};

namespace lut {
static const uint8_t nhashesper64bitword [] {
/*
# Auto-generated using:
print("    0xFFu, %s" % ", ".join(str(64 // x) for x in range(1, 63)))
    0xFFu, 64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

*/
    0xFFu, 64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
} // namespace lut

struct MurFinHash {
    static constexpr uint64_t C1 = 0xff51afd7ed558ccduLL, C2 = 0xc4ceb9fe1a85ec53uLL;
    INLINE uint64_t operator()(uint64_t key) const {
        key ^= key >> 33;
        key *= C1;
        key ^= key >> 33;
        key *= C2;
        key ^= key >> 33;
        return key;
    }
    INLINE Type operator()(Type key) const {
        return this->operator()(*(reinterpret_cast<VType *>(&key)));
    }
#if 0
&& VECTOR_WIDTH > 16
    INLINE auto operator()(__m128i key) const {
        using namespace vec;
        key ^= _mm_srli_epi64(key, 33);
        key = _mm_mul_epi64x(key, C1);
        key ^= _mm_srli_epi64(key,  33);
        key = _mm_mul_epi64x(key, C2);
        key ^= _mm_srli_epi64(key,  33);
        return key;
    }
#endif
    INLINE Type operator()(VType key) const {
#if 1
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        key.for_each([](auto &x) {x *= C1;});
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        key.for_each([](auto &x) {x *= C2;});
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#else
        __m128i *p = (__m128i *)&key;
        for(unsigned i = 0; i < sizeof(key) / sizeof(*p); ++i)
            *p = this->operator()(*p);
#endif
        return key.simd_;
    }
};
static INLINE uint64_t finalize(uint64_t key) {
    return MurFinHash()(key);
}

namespace op {

template<typename T>
struct multiplies {
    T operator()(T x, T y) const { return x * y;}
    VType operator()(VType x, VType y) const {
#if HAS_AVX_512
        return Space::mullo(x.simd_, y.simd_);
#else
        uint64_t *p1 = reinterpret_cast<uint64_t *>(&x), *p2 = reinterpret_cast<uint64_t *>(&y);
        for(uint32_t i = 0; i < sizeof(VType) / sizeof(uint64_t); ++i) {
            p1[i] *= p2[i];
        }
        return x;
#endif
    }
};
template<typename T>
struct plus {
    T operator()(T x, T y) const { return x + y;}
    VType operator()(VType x, VType y) const { return Space::add(x.simd_, y.simd_);}
};
template<typename T>
struct bit_xor {
    T operator()(T x, T y) const { return x ^ y;}
    VType operator()(VType x, VType y) const { return Space::xor_fn(x.simd_, y.simd_);}
};
}

namespace multinv {
// From Daniel Lemire's Blog: https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2017/09/18/inverse.c
// Blog post: https://lemire.me/blog/2017/09/18/computing-the-inverse-of-odd-integers/
// 25 cycles (longest it would take) + 1 cycle for a multiply is at least 3x as fast as performing an IDIV
// but it only works if the integer is odd.

static inline constexpr uint32_t f32(uint32_t x, uint32_t y) { return y * (2 - y * x); }

static constexpr uint32_t findInverse32(uint32_t x) {
  uint32_t y = (3 * x) ^ 2;
  y = f32(x, y);
  y = f32(x, y);
  y = f32(x, y);
  return y;
}

static inline constexpr uint64_t f64(uint64_t x, uint64_t y) { return y * (2 - y * x); }

static inline uint64_t findMultInverse64(uint64_t x) {
  assert(x & 1 || !std::fprintf(stderr, "Can't get multiplicative inverse of an even number."));
  uint64_t y = (3 * x) ^ 2;
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  return y;
}

template<typename T> inline T findmultinv(T v) {
    return findmultinv<typename std::make_unsigned<T>::type>(v);
}
template<> inline uint64_t findmultinv<uint64_t>(uint64_t v) {return findMultInverse64(v);}
template<> inline uint32_t findmultinv<uint32_t>(uint32_t v) {return findInverse32(v);}


template<typename T>
struct Inverse64 {
    uint64_t operator()(uint64_t x) const {return x;}
    template<typename T2>
    T2 apply(T2 x) const {return this->operator()(x);}
};

template<>
struct Inverse64<op::multiplies<uint64_t>> {
    uint64_t operator()(uint64_t x) const {return findMultInverse64(x);}
    uint64_t apply(uint64_t x) const {return this->operator()(x);}
};

template<>
struct Inverse64<std::plus<uint64_t>> {
    uint64_t operator()(uint64_t x) const {return std::numeric_limits<uint64_t>::max() - x + 1;}
    uint64_t apply(uint64_t x) const {return this->operator()(x);}
};


} // namespace multinv

template<size_t n, bool left>
struct Rot {
    template<typename T>
    T constexpr operator()(T val) const {
        static_assert(n < sizeof(T) * CHAR_BIT, "Can't shift more than the width of the type.");
        return left ? (val << n) ^ (val >> (64 - n))
                    : (val >> n) ^ (val << (64 - n));
    }
    template<typename T, typename T2>
    T constexpr operator()(T val, const T2 &oval) const { // DO nothing
        return this->operator()(val);
    }
    template<typename T>
    using InverseOperation = Rot<64 - n, !left>;
};
template<size_t n> using RotL = Rot<n, true>;
template<size_t n> using RotR = Rot<n, false>;

template<size_t n> struct ShiftXor;


// InvShiftXor and ShiftXor are
// Feistel ciphers.
// For more details, visit https://naml.us/post/inverse-of-a-hash-function/
template<size_t n>
struct InvShiftXor {
    uint64_t constexpr operator()(uint64_t v) const {
        uint64_t tmp = v ^ (v >> n);
        for(unsigned i = 0; i < 64 / n - 2; ++i)
            tmp = v ^ (tmp >> n);
        v = v ^ (tmp>>n);
        return v;
    }
    using InverseOperation = ShiftXor<n>;
};

template<size_t n>
struct ShiftXor {
    uint64_t constexpr operator()(uint64_t v) const {
        return v ^ (v >> n);
    }
    using InverseOperation = InvShiftXor<n>;
};

using RotL33 = RotL<33>;
using RotR33 = RotR<33>;
using RotL31 = RotL<31>;
using RotR31 = RotR<31>;



template<typename Operation, typename InverseOperation=Operation>
struct InvH {
    uint64_t seed_, inverse_;
    const Operation op;
    const InverseOperation iop;

    InvH(uint64_t seed):
            seed_(seed | std::is_same<Operation, op::multiplies<uint64_t>>::value),
            inverse_(multinv::Inverse64<Operation>()(seed_)), op(), iop() {}
    // To ensure that it is actually reversible.
    uint64_t inverse(uint64_t hv) const {
        hv = iop(hv, inverse_);
        return hv;
    }
    VType inverse(VType hv) const {
        hv = iop(hv.simd_, Space::set1(inverse_));
        return hv;
    }
    uint64_t operator()(uint64_t h) const {
        h = op(h, seed_);
        return h;
    }
    VType operator()(VType h) const {
        const VType s = Space::set1(seed_);
        h = op(h, s);
        return h;
    }
};

// Reversible, runtime-configurable hashes


template<typename InvH1, typename InvH2>
struct FusedReversible {
    InvH1 op1;
    InvH2 op2;
    FusedReversible(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL):
        op1(seed1), op2(seed2) {}
    template<typename T>
    T operator()(T h) const {
        h = op1(h);
        h = op2(h);
        return h;
    }
    template<typename T>
    T inverse(T hv) const {
        hv = op1.inverse(op2.inverse(hv));
        return hv;
    }
};
template<typename InvH1, typename InvH2, typename InvH3>
struct FusedReversible3 {
    InvH1 op1;
    InvH2 op2;
    InvH3 op3;
    FusedReversible3(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL):
        op1(seed1), op2(seed2), op3((seed1 * seed2 + seed2) | 1) {}
    template<typename T>
    T operator()(T h) const {
        h = op3(op2(op1(h)));
        return h;
    }
    template<typename T>
    T inverse(T hv) const {
        hv = op1.inverse(op2.inverse(op3.inverse(hv)));
        return hv;
    }
};

using InvXor = InvH<op::bit_xor<uint64_t>>;
using InvMul = InvH<op::multiplies<uint64_t>>;
using InvAdd = InvH<op::plus<uint64_t>>;
template<size_t n>
using RotN = InvH<RotL<n>, RotR<64 - n>>;

struct XorMultiply: public FusedReversible<InvXor, InvMul > {
    XorMultiply(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible<InvXor, InvMul >(seed1, seed2) {}
};
struct MultiplyAdd: public FusedReversible<InvMul, InvAdd> {
    MultiplyAdd(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible<InvMul, InvAdd>(seed1, seed2) {}
};
struct MultiplyAddXor: public FusedReversible3<InvMul,InvAdd,InvXor> {
    MultiplyAddXor(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible3<InvMul,InvAdd,InvXor>(seed1, seed2) {}
};
template<size_t shift>
struct MultiplyAddXoRot: public FusedReversible3<InvMul,InvXor,RotN<shift>> {
    MultiplyAddXoRot(uint64_t seed1, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible3<InvMul,InvXor, RotN<shift>>(seed1, seed2) {}
};

template<typename HashType>
struct RecursiveReversibleHash {
    std::vector<HashType> v_;
    template<typename... Args>
    RecursiveReversibleHash(size_t n, uint64_t seed1=1337, Args &&... args) {
        std::mt19937_64 mt(seed1);
        while(v_.size() < n)
            v_.emplace_back(mt() | 1, mt(), std::forward<Args>(args)...);
    }
    template<typename T>
    T operator()(T v) const {
        std::for_each(v_.begin(), v_.end(), [&](const auto &hash) {v = hash(v);});
        return v;
    }
    template<typename T>
    T inverse(T hv) const {
        std::for_each(v_.rbegin(), v_.rend(), [&](const auto &hash) {hv = hash.inverse(hv);});
        return hv;
    }
};

struct XorMultiplyNVec: public RecursiveReversibleHash<XorMultiply> {
    XorMultiplyNVec(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<XorMultiply>(n, seed1) {}
};
struct MultiplyAddNVec: public RecursiveReversibleHash<MultiplyAdd> {
    MultiplyAddNVec(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<MultiplyAdd>(n, seed1) {}
};
struct MultiplyAddXorNVec: public RecursiveReversibleHash<MultiplyAddXor> {
    MultiplyAddXorNVec(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<MultiplyAddXor>(n, seed1) {}
};
template<size_t shift>
struct MultiplyAddXoRotNVec: public RecursiveReversibleHash<MultiplyAddXoRot<shift>> {
    MultiplyAddXoRotNVec(size_t n, uint64_t seed1=0xB0BAF377D00Dc001uLL):
        RecursiveReversibleHash<MultiplyAddXoRot<shift>>(n, seed1) {}
};

template<size_t shift, size_t n>
struct MultiplyAddXoRotN: MultiplyAddXoRotNVec<shift> {
    MultiplyAddXoRotN(): MultiplyAddXoRotNVec<shift>(n) {}
};
template<size_t n>
struct MultiplyAddXorN: MultiplyAddXorNVec {
    MultiplyAddXorN(): MultiplyAddXorNVec(n) {}
};
template<size_t n>
struct MultiplyAddN: MultiplyAddNVec{
    MultiplyAddN(): MultiplyAddNVec(n) {}
};
template<size_t n>
struct XorMultiplyN: XorMultiplyNVec{
    XorMultiplyN(): XorMultiplyNVec(n) {}
};

template<typename T>
static constexpr inline bool is_pow2(T val) {
    return val && (val & (val - 1)) == 0;
}

template<typename T>
class TD;

INLINE auto popcount(uint64_t val) noexcept {
#ifdef AVOID_ASM_POPCNT
// From cqf https://github.com/splatlab/cqf/
    asm("popcnt %[val], %[val]"
            : [val] "+r" (val)
            :
            : "cc");
    return val;
#else
    // According to GodBolt, gcc7.3 fails to inline this function call even at -Ofast.
    //
    //
    return __builtin_popcountll(val);
#endif
}
inline unsigned popcount(__m64 val) noexcept {return popcount(*reinterpret_cast<uint64_t *>(&val));}

template<typename T>
static INLINE uint64_t vatpos(const T v, size_t ind) {
    return reinterpret_cast<const uint64_t *>(&v)[ind];
}

template<typename T>
static INLINE uint64_t sum_of_u64s(const T val) {
    uint64_t sum = vatpos(val, 0);
    for(size_t i = 1; i < sizeof(T) / sizeof(uint64_t); ++i)
        sum += vatpos(val, i);
    return sum;
}
template<typename T>
INLINE auto popcnt_fn(T val);
template<>
INLINE auto popcnt_fn(Type val) {

#define VAL_AS_ARR(ind) reinterpret_cast<const uint64_t *>(&val)[ind]
#if HAS_AVX_512
#  if defined(__AVX512VPOPCNTDQ__)
#    define FUNCTION_CALL ::_mm512_popcnt_epi64(val)
#  else
#    define FUNCTION_CALL popcnt512(val)
#  endif
#elif __AVX2__
// This is supposed to be the fastest option according to the README at https://github.com/kimwalisch/libpopcnt
#define FUNCTION_CALL popcnt256(val)
#elif __SSE2__
#define FUNCTION_CALL _mm_set_epi64x(_mm_cvtsi128_si64(val), _mm_cvtsi128_si64(_mm_unpackhi_epi64(val, val)))
#else
#  error("Need SSE2. TODO: make this work for non-SIMD architectures")
#endif
    return FUNCTION_CALL;
#undef FUNCTION_CALL
#undef VAL_AS_ARR
}
template<>
INLINE auto popcnt_fn(VType val) {
    return popcnt_fn(val.simd_);
}

namespace sort {
// insertion_sort from https://github.com/orlp/pdqsort
// Slightly modified stylistically.
template<class Iter, class Compare>
inline void insertion_sort(Iter begin, Iter end, Compare comp) {
    using T = typename std::iterator_traits<Iter>::value_type;

    for (Iter cur = begin + 1; cur < end; ++cur) {
        Iter sift = cur;
        Iter sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already positioned correctly.
        if (comp(*sift, *sift_1)) {
            T tmp = std::move(*sift);

            do { *sift-- = std::move(*sift_1); }
            while (sift != begin && comp(tmp, *--sift_1));

            *sift = std::move(tmp);
        }
    }
}
template<class Iter>
inline void insertion_sort(Iter begin, Iter end) {
    insertion_sort(begin, end, std::less<std::decay_t<decltype(*begin)>>());
}
#ifndef SORT_ALGORITHM
template<typename... Args>
void default_sort(Args &&... args) {std::sort(std::forward<Args>(args)...);}
#else
template<typename... Args>
void default_sort(Args &&... args) {SORT_ALGORITHM(std::forward<Args>(args)...);}
#endif

} // namespace sort


struct DoNothing {
    template<typename... Args>void operator()(const Args &&...  args)const{}
    template<typename T>void operator()(const T &x)const{}
};

class NotImplementedError: public std::runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): std::runtime_error(std::forward<Args>(args)...) {}
};
namespace detail {
// Overloads for setting memory to 0 for either compact vectors
// or std::vectors
template<typename T, typename AllocatorType=typename T::allocator>
static inline void zero_memory(std::vector<T, AllocatorType> &v, size_t newsz) {
    std::memset(v.data(), 0, v.size() * sizeof(v[0]));
    v.resize(newsz);
    std::fprintf(stderr, "New size of container: %zu\n", newsz);
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::ts_vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}
} // namespace detail

enum MHCardinalityMode: uint8_t {
    HARMONIC_MEAN,
    ARITHMETIC_MEAN,
    HLL_METHOD, // Should perform worse than harmonic
};

} // namespace common
} // namespace sketch
