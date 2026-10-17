#ifndef DBSKETCH_HASH_H__
#define DBSKETCH_HASH_H__
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "aesctr/wy.h"
#include "fixed_vector.h"
#include "xxHash/xxh3.h"
#include "macros.h"

#ifndef VEC_DISABLED__
#include "vec/vec.h"
#endif

namespace sketch {
inline namespace hash {
using std::uint64_t;
using std::uint32_t;
using std::size_t;

#ifndef VEC_DISABLED__
using Type  = typename vec::SIMDTypes<uint64_t>::Type;
using VType = typename vec::SIMDTypes<uint64_t>::VType;
using Space = vec::SIMDTypes<uint64_t>;
#endif
static inline uint64_t mthash(uint64_t x) {
    std::mt19937_64 mt(x);
    return mt();
}
// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It is a bijection within [0,1<<64)
// and can be inverted with WangHash::inverse (member below; the same
// algorithm published as "irving_inv_hash" on the original site).
struct WangHash {
    template<typename...Args> WangHash(Args &&...) {}
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
    INLINE auto operator()(int64_t key) const {return operator()(uint64_t(key));}
    INLINE uint32_t operator()(uint32_t key) const {
        key += ~(key << 15);
        key ^=  (key >> 10);
        key +=  (key << 3);
        key ^=  (key >> 6);
        key += ~(key << 11);
        key ^=  (key >> 16);
        return key;
    }
    // Inverse of the 32-bit operator() above (each forward step undone in
    // reverse order; the un-xorshift rounds use shift-doubling so every
    // shift stays < 32). Without this overload, w.inverse(w(uint32_t))
    // would silently convert to the 64-bit inverse and return garbage.
    static constexpr uint32_t inverse(uint32_t key) {
        key = key ^ (key >> 16);                                   // undo key ^= key >> 16
        key = (key + 1) * UINT32_C(4196353);                       // undo key += ~(key << 11): * (1 - 2^11)^-1
        key = key ^ (key >> 6); key = key ^ (key >> 12); key = key ^ (key >> 24); // undo key ^= key >> 6
        key = key * UINT32_C(954437177);                           // undo key += key << 3: * 9^-1
        key = key ^ (key >> 10); key = key ^ (key >> 20);          // undo key ^= key >> 10
        key = (key + 1) * UINT32_C(1073774593);                    // undo key += ~(key << 15): * (1 - 2^15)^-1
        return key;
    }
    INLINE auto inverse(int32_t key) const {return inverse(uint32_t(key));}
    INLINE auto operator()(int32_t key) const {return operator()(uint32_t(key));}
#ifndef VEC_DISABLED__
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
#endif
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
    INLINE __uint128_t operator()(__uint128_t key) const {
        return this->operator()(uint64_t(key)) | (__uint128_t(this->operator()(uint64_t(key >> 64))) << 64);
    }
    INLINE __uint128_t inverse(__uint128_t key) const {
        return inverse(static_cast<uint64_t>(key)) | (__uint128_t(inverse(uint64_t(key >> 64))) << 64);
    }
    INLINE uint64_t inverse(uint64_t key) const {
        // https://naml.us/blog/tag/thomas-wang
        uint64_t tmp;
        // Invert key = key + (key << 31)
        tmp = key-(key<<31);
        key = key-(tmp<<31);
        // Invert key = key ^ (key >> 28)
        tmp = key^key>>28;
        key = key^tmp>>28;
        // Invert key *= 21
        key *= UINT64_C(14933078535860113213);
        // Invert key = key ^ (key >> 14)
        tmp = key^key>>14;
        tmp = key^tmp>>14;
        tmp = key^tmp>>14;
        key = key^tmp>>14;
        // Invert key *= 265
        key *= UINT64_C(15244667743933553977);
        // Invert key = key ^ (key >> 24)
        tmp = key^key>>24;
        key = key^tmp>>24;
        // Invert key = (~key) + (key << 21)
        tmp = ~key;
        tmp = ~(key-(tmp<<21));
        tmp = ~(key-(tmp<<21));
        key = ~(key-(tmp<<21));
        return key;
    }
    template<typename T>
    INLINE uint64_t operator()(const T&x) const {return this->operator()(static_cast<uint64_t>(x));}
};

template<typename BaseHash>
struct SeededHash: public BaseHash {
    const uint64_t seed_;
    template<typename...A>
    SeededHash(uint64_t seed, A &&...a): seed_(seed), BaseHash(std::forward<A>(a)...) {}
    auto operator()(uint64_t item) const {return BaseHash::operator()(item ^ seed_);}
};

// pcg32
// The purpose of this is fast random number generation for {Super,Bag}MinHash
struct pcg32_random_t {   // Internals are *Private*.
    uint64_t state;                    // RNG state.  All values are possible.
    uint64_t inc;                      // Controls which RNG sequence (stream) is
                                       // selected. Must *always* be odd.
};

static INLINE uint32_t pcg32_random_r(pcg32_random_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

struct PCGen: pcg32_random_t {
    PCGen(uint64_t seed, uint64_t inc=3930499866110305181uLL){
        this->state = seed; this->inc = inc|1; // Ensures that increment is odd
    }
    PCGen &seed(uint64_t newseed) {this->state = newseed; return *this;}
    uint32_t operator()() {return pcg32_random_r(this);}
    uint64_t make_u64() {return (uint64_t(this->operator()()) << 32) | this->operator()();}
    static constexpr uint32_t max() {return std::numeric_limits<uint32_t>::max();}
    static constexpr uint32_t min() {return std::numeric_limits<uint32_t>::min();}
};
using int96_t = std::array<uint32_t, 3>;

//#define NO_USE_SIAM 1
template<size_t N>
static auto make_coefficients64(uint64_t seedseed) {
    std::mt19937_64 mt(seedseed);
	std::array<uint64_t, N> ret;
    for(auto &e: ret) e = mt() % ((1ull << 61) - 1);
    return ret;
}
template<size_t N>
static auto make_coefficients(uint64_t seedseed) {
    std::mt19937_64 mt(seedseed);
    std::array<int96_t, N> ret;
    for(auto &e: ret) {
#if __BYTE_ORDER__ == 1234
        e[0] = mt();
        e[1] = mt();
        e[2] = mt() % ((1u << 25) - 1);
#elif __BYTE_ORDER__ == 4321
        e[0] = mt() % ((1 << 25) - 1);
        e[1] = mt();
        e[2] = mt();
#else
#error("big or little")
#endif
    }
    return ret;
}
namespace siam {

// From Tabulation Based 5-Universal Hashing and Linear Probing
//
static constexpr uint64_t Prime89_0  = (((uint64_t)1)<<32)-1;
static constexpr uint64_t Prime89_1  = (((uint64_t)1)<<32)-1;
static constexpr uint64_t Prime89_2  = (((uint64_t)1)<<25)-1;
static constexpr uint64_t Prime89_21 = (((uint64_t)1)<<57)-1;

inline uint64_t Mod64Prime89(const int96_t r) {
#if __BYTE_ORDER__ == 1234
    uint64_t r0, r1, r2; //r2r1r0 = r&Prime89 + r>>89
    r2 = r[2];
    r1 = r[1];
    r0 = r[0] + (r2>>25);
    r2 &= Prime89_2;
    return (r2 == Prime89_2 && r1 == Prime89_1 && r0 >= Prime89_0) ?(r0 - Prime89_0) : (r0 + (r1<<32));
#elif __BYTE_ORDER__ == 4321
    uint64_t r0, r1, r2; //r2r1r0 = r&Prime89 + r>>89
    r2 = r[0];
    r1 = r[1];
    r0 = r[2] + (r2>>25);
    r2 &= Prime89_2;
    return (r2 == Prime89_2 && r1 == Prime89_1 && r0 >= Prime89_0) ?(r0 - Prime89_0) : (r0 + (r1<<32));
#else
#error("Do not support endianness besides big and little")
#endif
}/*Computes a 96-bit r such thatr mod Prime89 == (ax+b) mod Prime89exploiting the structure of Prime89.*/

static constexpr uint64_t HIGH(uint64_t x) {return x >> 32;}
static constexpr uint64_t LOW(uint64_t x) {return x & 0x00000000FFFFFFFFull;}

inline void MultAddPrime89(int96_t & r, uint64_t x, const int96_t &a, const int96_t &b)
{
    uint64_t x1, x0, c21, c20, c11, c10, c01, c00;
    uint64_t d0, d1, d2, d3;
	uint64_t s0, s1, carry;
    x1 = HIGH(x);
    x0 = LOW(x);
    c21 = a[2]*x1;
    c20 = a[2]*x0;
    c11 = a[1]*x1;
    c10 = a[1]*x0;
    c01 = a[0]*x1;
    c00 = a[0]*x0;
    d0 = (c20>>25)+(c11>>25)+(c10>>57)+(c01>>57);
    d1 = (c21<<7);
    d2 = (c10&Prime89_21) + (c01&Prime89_21);
    d3 = (c20&Prime89_2) +(c11&Prime89_2) + (c21>>57);
    s0 = b[0] + LOW(c00) + LOW(d0) + LOW(d1);
    r[0] = LOW(s0);
    carry = HIGH(s0);
    s1 = b[1] + HIGH(c00) + HIGH(d0) +HIGH(d1) + LOW(d2) + carry;
    r[1] = LOW(s1);
    carry = HIGH(s1);
    r[2] = b[2] + HIGH(d2) + d3 + carry;
}
// CWtrick64 for 64-bit key x (Prime = 2ˆ89-1)

template<size_t k>
inline uint64_t CWtrick64(uint64_t x, const std::array<int96_t, k> &keys) {
	static_assert(k > 2, "If you only need 2, don't use this function.");
    int96_t r;
    MultAddPrime89(r,x,keys[0],keys[1]);
	for(size_t i = 2; i < k; ++i)
    	MultAddPrime89(r,x,r,keys[i]);
    //MultAddPrime89(r,x,r,E); We don't need this
    return Mod64Prime89(r);
}

}

namespace nosiam {

INLINE __uint128_t mod127(__uint128_t x) {
    static constexpr __uint128_t mod = (__uint128_t(1) << 127) - 1;
    x = (x >> 127) + (x & mod);
    if(x > mod) x -= mod;
    return x;
}


INLINE __uint128_t mod61(__uint128_t x) {
    static constexpr uint64_t mod = (__uint128_t(1) << 61) - 1;
    do x = (x >> 61) + (x & mod); while(x > mod * 2);
    if(x >= mod) x -= mod;
    return x;
}
INLINE uint64_t mod61(uint64_t x) {
    static constexpr uint64_t mod = (uint64_t(1) << 61) - 1;
    do x = (x >> 61) + (x & mod); while(x > mod * 2);
    if(x >= mod) x -= mod;
    return x;
}

INLINE uint64_t mulmod61(uint64_t x1, uint64_t x2) {
    return mod61(__uint128_t(x1) * x2);
}

template<size_t n>
inline uint64_t i61hash(uint64_t x, const std::array<uint64_t, n> & keys) {
    uint64_t tsum = mulmod61(x, keys[1]);
    tsum += keys[0];
    uint64_t xp = x;
    for(size_t i = 2; i < n; ++i) {
        if(i % 8 == 0) tsum = mod61(tsum);
        xp = mulmod61(xp, x);
        tsum += mulmod61(xp, keys[i]);
    }
	return mod61(tsum);
}
inline uint64_t wy61hash(uint64_t x, size_t n, uint64_t seed) {
    wy::WyHash<uint64_t, 0> gen(seed);
    uint64_t tsum = gen();
    tsum += mulmod61(x, gen());
    uint64_t xp = x;
    for(size_t i = 2; i < n; ++i) {
        tsum = mod61(tsum);
        xp = mulmod61(xp, x);
        tsum += mulmod61(xp, gen());
    }
	return mod61(tsum);
}
template<size_t k>
inline uint64_t i128hash(uint64_t x, const std::array<uint64_t, k> & keys) {
    // Use 2**127
    __uint128_t sum = mod127(mod127(x * keys[1]) + keys[0]);
    __uint128_t xp = x * x;
    for(size_t i = 2; i < k; ++i) {
        xp = mod127(xp);
        sum = mod127(sum + mod127(xp * keys[i]));
        xp *= x;
    }
    return sum;
}

} // nosiam

template<size_t k>
class KWiseIndependentPolynomialHash {
    static_assert(k, "k must be positive");
    const std::array<int96_t, k> coeffs_;
    static constexpr bool is_kwise_independent(size_t val) {return val <= k;}
public:
    KWiseIndependentPolynomialHash(uint64_t seedseed=137): coeffs_(make_coefficients<k>(seedseed)) {
    }
    uint64_t operator()(uint64_t val) const {
		return siam::CWtrick64(val, coeffs_);
    }
#ifndef VEC_DISABLED__
    Type operator()(VType val) const {
        throw std::runtime_error("Not implemented");
        return val.simd_;
    }
#endif
#ifdef DUMMY_INVERSE
    uint64_t inverse(uint64_t val) { return val;} // This is a lie for compatibility only
#endif
};

template<size_t k>
class KWiseIndependentPolynomialHash61 {
    static_assert(k, "k must be positive");
    const std::array<uint64_t, k> coeffs_;
	static constexpr uint64_t mod = (uint64_t(1) << 61) - 1;
    static constexpr bool is_kwise_independent(size_t val) {return val <= k;}
public:
    KWiseIndependentPolynomialHash61(uint64_t seedseed=137): coeffs_(make_coefficients64<k>(seedseed)) {
    }
    uint64_t operator()(uint64_t val) const {
		return nosiam::i61hash(val, coeffs_);
    }
#ifndef VEC_DISABLED__
    Type operator()(VType val) const {
        throw std::runtime_error("Not implemented");
        return val.simd_;
    }
#endif
#ifdef DUMMY_INVERSE
    uint64_t inverse(uint64_t val) { return val;} // This is a lie for compatibility only
#endif
};


template<typename Hasher=SeededHash<WangHash>>
struct HasherSet {
    std::vector<Hasher> hashers_;
    HasherSet(size_t nh, uint64_t seedseed=137) {
        std::mt19937_64 mt(seedseed);
        while(hashers_.size() < nh)
            hashers_.emplace_back(mt());
    }
    size_t size() const {return hashers_.size();}
    uint64_t operator()(uint64_t v, uint32_t ind) const {
        return hashers_[ind](v);
    }
    uint64_t operator()(uint64_t) const {throw std::runtime_error("Should not be called.");}
};
template<typename Hasher=WangHash>
struct XORSeedHasherSet {
#ifndef VEC_DISABLED__
    static constexpr size_t ALN = sizeof(vec::SIMDTypes<uint64_t>::VType);
#else
    static constexpr size_t ALN = 16;
#endif
    Hasher hasher_;
    fixed::vector<uint64_t, ALN> seeds_;
    template<typename...Args>
    XORSeedHasherSet(size_t nh, uint64_t seedseed=137, Args &&... args):
        hasher_(std::forward<Args>(args)...),
        seeds_(nh)
    {
        std::mt19937_64 mt(seedseed);
        for(uint32_t i = 0; i < nh; seeds_[i++] = mt());
    }
    size_t size() const {return seeds_.size();}
    uint64_t operator()(uint64_t v, uint32_t ind) const {
        return hasher_(v ^ seeds_[ind]);
    }
#ifndef VEC_DISABLED__
    fixed::vector<uint64_t> operator()(uint64_t v) const {
        using VT = typename vec::SIMDTypes<uint64_t>::VType;
        VT broadcast = vec::SIMDTypes<uint64_t>::set1(v);
        fixed::vector<uint64_t> ret(size());
        size_t i = 0;
        while(i + vec::SIMDTypes<uint64_t>::COUNT < size()) {
            VT tmpv = vec::SIMDTypes<uint64_t>::xor_fn(
                vec::SIMDTypes<uint64_t>::load(reinterpret_cast<const typename vec::SIMDTypes<uint64_t>::Type *>(&seeds_[i])),
                broadcast.simd_
            );
            tmpv.for_each([&](uint64_t hv) {
                ret[i++] = hv;
            });
        }
        for(;i < size();++i)
            ret[i] = hasher_(v ^ seeds_[i]);
        return ret;
    }
#endif
};

template<size_t k>
struct KWiseHasherSet: public HasherSet<KWiseIndependentPolynomialHash<k>> {
    using super = HasherSet<KWiseIndependentPolynomialHash<k>>;

    template<typename...Args>
    KWiseHasherSet(Args &&...args): super(std::forward<Args>(args)...) {}
};

#if 0
template<typename Hasher>
struct HasherSet {
    std::vector<uint64_t> seed_;
    KWiseHasherSet(size_t nh, uint64_t seedseed=137) {
        std::mt19937_64 mt(seedseed);
        while(hashers_.size() < nh)
            hashers_.emplace_back(mt());
    }
    size_t size() const {return hashers_.size();}
    uint64_t operator()(uint64_t v, uint32_t ind) const {
        return hashers_[ind](v);
    }
    uint64_t operator()(uint64_t v) const {throw std::runtime_error("Should not be called.");}
};
#endif

namespace op {

template<typename T>
struct multiplies {
    T operator()(T x, T y) const { return x * y;}
#ifndef VEC_DISABLED__
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
#endif
};
template<typename T>
struct plus {
    T operator()(T x, T y) const { return x + y;}
#ifndef VEC_DISABLED__
    VType operator()(VType x, VType y) const { return Space::add(x.simd_, y.simd_);}
#endif
};
template<typename T>
struct bit_xor {
    T operator()(T x, T y) const { return x ^ y;}
#ifndef VEC_DISABLED__
    VType operator()(VType x, VType y) const { return Space::xor_fn(x.simd_, y.simd_);}
#endif
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
static inline constexpr uint64_t findMultInverse64CE(uint64_t x) {
  uint64_t y = (3 * x) ^ 2;
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  return y;
}
static inline uint64_t findMultInverse64(uint64_t x) {
  assert(x & 1); // Odd inputs only: an even number has no multiplicative inverse mod 2^64.
  uint64_t y = (3 * x) ^ 2;
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  y = f64(x, y);
  return y;
}
template<typename T> inline T findmultinv(T v) {return findmultinv<typename std::make_unsigned<T>::type>(v);}
template<> inline uint64_t findmultinv<uint64_t>(uint64_t v) {return findMultInverse64(v);}
template<> inline uint32_t findmultinv<uint32_t>(uint32_t v) {return findInverse32(v);}


template<typename T>
struct Inverse64 {
    uint64_t operator()(uint64_t x) const {return x;}
    template<typename T2> T2 apply(T2 x) const {return this->operator()(x);}
};

template<>
struct Inverse64<op::multiplies<uint64_t>> {
    uint64_t operator()(uint64_t x) const {return findMultInverse64(x);}
    uint64_t apply(uint64_t x) const {return this->operator()(x);}
};

template<>
struct Inverse64<op::plus<uint64_t>> {
    uint64_t operator()(uint64_t x) const {return std::numeric_limits<uint64_t>::max() - x + 1;}
    uint64_t apply(uint64_t x) const {return this->operator()(x);}
};

} // namespace multinv

struct MurFinHash {
    template<typename...Args> MurFinHash(Args &&...) {}
    static constexpr uint64_t C1 = 0xff51afd7ed558ccduLL, C2 = 0xc4ceb9fe1a85ec53uLL;
    static constexpr uint64_t IC1 = multinv::findMultInverse64CE(C1), IC2 = multinv::findMultInverse64CE(C2);
    INLINE uint64_t operator()(uint64_t key) const {
        key ^= key >> 33;
        key *= C1;
        key ^= key >> 33;
        key *= C2;
        key ^= key >> 33;
        return key;
    }
    // True inverse: this finalizer is a bijection (two odd multiplies between
    // three xorshift-rights), so the round trip is exact.
    INLINE uint64_t inverse(uint64_t key) const {
        key ^= key >> 33;
        key *= IC2;
        key ^= key >> 33;
        key *= IC1;
        key ^= key >> 33;
        return key;
    }
#ifndef VEC_DISABLED__
    INLINE Type operator()(Type key) const {
        return this->operator()(*(reinterpret_cast<VType *>(&key)));
    }
#endif
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
#ifndef VEC_DISABLED__
    INLINE Type operator()(VType key) const {
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        key.for_each([](auto &x) {x *= C1;});
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        key.for_each([](auto &x) {x *= C2;});
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        return key.simd_;
    }
#endif
};

static INLINE uint64_t finalize(uint64_t key) {
    return MurFinHash()(key);
}


template<size_t n>
struct RShiftXor;
template<size_t n>
struct InvRShiftXor;
template<size_t n>
struct InvRShiftXor {
    static_assert(n > 0 && n < sizeof(uint64_t) * CHAR_BIT, "shift must be in [1, 64)");
    template<typename...Args>
    constexpr InvRShiftXor(Args &&...) {}
    static constexpr size_t enditer = (sizeof(uint64_t) * CHAR_BIT) / n;
    uint64_t constexpr operator()(uint64_t v) const {
        uint64_t ret = v ^ v >> n;
        for(size_t i = 1; i < enditer; ++i)
            ret = v ^ ret >> n;
        return ret;
    }
    uint64_t inverse(uint64_t x) const {return InverseOperation()(x);}
    using InverseOperation = RShiftXor<n>;
};

template<size_t n>
struct RShiftXor {
    static_assert(n > 0 && n < sizeof(uint64_t) * CHAR_BIT, "shift must be in [1, 64)");
    template<typename...Args>
    constexpr RShiftXor(Args &&...) {}
    uint64_t constexpr operator()(uint64_t v) const {return v ^ (v >> n);}
#ifdef __SSE2__
    __m128i operator()(__m128i v) const {
        return _mm_xor_si128(_mm_srli_epi64(v, n), v);
    }
#endif
#ifdef __AVX2__
    __m256i operator()(__m256i v) const {
        return _mm256_xor_si256(_mm256_srli_epi64(v, n), v);
    }
#endif
#ifdef __AVX512DQ__
    __m512i operator()(__m512i v) const {
        return _mm512_xor_si512(_mm512_srli_epi64(v, n), v);
    }
#endif
    uint64_t inverse(uint64_t x) const {return InverseOperation()(x);}
    using InverseOperation = InvRShiftXor<n>;
};
template<size_t n> using ShiftXor = RShiftXor<n>;
template<size_t n> using InvShiftXor = InvRShiftXor<n>;

template<uint64_t XORVALUE=UINT64_C(0xe37e28c4271b5a2d)>
struct XORConstantHasher {
    template<typename...Args>
    constexpr XORConstantHasher(Args &&...) {}
    uint64_t operator()(uint64_t h) const {
        return h ^ XORVALUE;
    }
    uint64_t inverse(uint64_t h) const {return h ^ XORVALUE;}
};

template<size_t n>
struct LShiftXor;
template<size_t n>
struct InvLShiftXor;
template<size_t n>
struct LShiftXor {
    static_assert(n > 0 && n < sizeof(uint64_t) * CHAR_BIT, "shift must be in [1, 64)");
    template<typename...Args>
    constexpr LShiftXor(Args...) {}
    uint64_t constexpr operator()(uint64_t v) const {return v ^ (v << n);}
    uint64_t constexpr inverse(uint64_t v) const {return InverseOperation()(v);}
    using InverseOperation = InvLShiftXor<n>;
};
template<size_t n>
struct InvLShiftXor {
    static_assert(n > 0 && n < sizeof(uint64_t) * CHAR_BIT, "shift must be in [1, 64)");
    template<typename...Args>
    constexpr InvLShiftXor(Args &&...) {}
    static constexpr size_t enditer = (sizeof(uint64_t) * CHAR_BIT) / n;
    uint64_t constexpr operator()(uint64_t v) const {
        uint64_t ret = v ^ v << n;
        for(size_t i = 1; i < enditer; ++i)
            ret = v ^ ret << n;
        return ret;
    }
    constexpr uint64_t inverse(uint64_t v) const {return InverseOperation()(v);}
    using InverseOperation = LShiftXor<n>;
};


template<size_t n, bool left>
struct Rot {
    template<typename...Args>
    constexpr Rot(Args &&...) {}
    template<typename T>
    INLINE T constexpr operator()(T val) const {
        static_assert(n > 0 && n < sizeof(T) * CHAR_BIT, "Rotation amount must be in [1, width of T).");
        return left ? T(val << n) ^ T(val >> (sizeof(T) * CHAR_BIT - n))
                    : T(val >> n) ^ T(val << (sizeof(T) * CHAR_BIT - n));
    }
    template<typename T>
    INLINE T constexpr inverse(T val) const {
        return InverseOperation()(val);
    }
    template<typename T, typename T2>
    INLINE T constexpr operator()(T val, const T2 &) const {
        return this->operator()(val);
    }
    using InverseOperation = Rot<n, !left>;
};
template<size_t n> using RotL = Rot<n, true>;
template<size_t n> using RotR = Rot<n, false>;
struct BitFlip {
    template<typename...Args>
    constexpr BitFlip(Args &&...) {}
    template<typename T>
    INLINE T constexpr operator()(T val) const {
        return ~val;
    }
    template<typename T>
    INLINE T constexpr inverse(T val) const {
        return InverseOperation()(val);
    }
    template<typename T, typename T2>
    INLINE T constexpr operator()(T val, const T2 &) const {
        return this->operator()(val);
    }
    using InverseOperation = BitFlip;
};

using RotL33 = RotL<33>;
using RotR33 = RotR<33>;
using RotL31 = RotL<31>;
using RotR31 = RotR<31>;

template<typename Operation, typename InverseOperation=Operation>
struct InvH {
    uint64_t seed_, inverse_; // Note: inverse_ is only used for op::multiplies
    const Operation op;
    const InverseOperation iop;

    InvH(uint64_t seed):
            seed_(seed | std::is_same<Operation, op::multiplies<uint64_t>>::value), // Ensures that the seed is odd if multiplies, as an odd number is needed for reversibility.
            inverse_(multinv::Inverse64<Operation>()(seed_)), op(), iop() {}
    // To ensure that it is actually reversible.
    INLINE uint64_t inverse(uint64_t hv) const {
        hv = iop(hv, inverse_);
        return hv;
    }
    INLINE uint64_t operator()(uint64_t h) const {
        h = op(h, seed_);
        return h;
    }
#ifndef VEC_DISABLED__
    INLINE VType inverse(VType hv) const {
        hv = iop(hv.simd_, Space::set1(inverse_));
        return hv;
    }
    INLINE VType operator()(VType h) const {
        const VType s = Space::set1(seed_);
        h = op(h, s);
        return h;
    }
#endif
};

template<uint64_t base_seed>
struct CEIXOR {
    static constexpr uint64_t seed_ = base_seed;
    static constexpr uint32_t seed32_ = uint32_t(base_seed);
    static constexpr uint64_t inverse_ = seed_;
    static constexpr uint64_t seed2_ = WangHash::hash(base_seed);
    static constexpr uint64_t inverse2_ = seed2_;
    template<typename...Args>
    constexpr CEIXOR(Args &&...) {}

    // To ensure that it is actually reversible.
    INLINE constexpr uint64_t inverse(uint64_t hv) const {
        return hv ^ seed_;
    }
    INLINE constexpr uint64_t operator()(uint64_t h) const {
        return h ^ seed_;
    }
    INLINE constexpr uint32_t inverse(uint32_t hv) const {
        return hv ^ seed32_;
    }
    INLINE constexpr uint32_t operator()(uint32_t h) const {
        return h ^ seed32_;
    }
    INLINE constexpr __uint128_t inverse(__uint128_t hv) const {
        return (uint64_t(hv) ^ inverse_) | (((hv >> 64) ^ inverse2_) << 64);
    }
    INLINE constexpr __uint128_t operator()(__uint128_t h) const {
        return (uint64_t(h) ^ seed_) | (((h >> 64) ^ seed2_) << 64);
    }
};


template<uint64_t base_seed>
struct CEIMul {
    static constexpr uint64_t seed_ = base_seed | 1;
    static constexpr uint32_t seed32_ = uint32_t(base_seed | 1);
    static constexpr uint64_t inverse_ = multinv::findMultInverse64CE(seed_); // Note: inverse_ is only used for op::multiplies
    static constexpr uint64_t inverse32_ = multinv::findInverse32(seed_); // Note: inverse_ is only used for op::multiplies
    static constexpr uint64_t seed2_ = WangHash::hash(base_seed) | 1;
    static constexpr uint64_t inverse2_ = multinv::findMultInverse64CE(seed2_); // multiplicative inverse of the HIGH-lane seed
    template<typename...Args>
    constexpr CEIMul(Args &&...) {}

    // To ensure that it is actually reversible.
    INLINE constexpr uint64_t operator()(uint64_t h) const {
        return h * seed_;
    }
    INLINE constexpr uint32_t operator()(uint32_t h) const {
        return h * seed32_;
    }
    INLINE constexpr __uint128_t operator()(__uint128_t h) const {
        return (uint64_t(h) * seed_) | (((h >> 64) * seed2_) << 64);
    }
    INLINE constexpr uint64_t inverse(uint64_t hv) const {
        return hv * inverse_;
    }
    INLINE constexpr uint32_t inverse(uint32_t hv) const {
        return hv * inverse32_;
    }
    INLINE constexpr __uint128_t inverse(__uint128_t hv) const {
        return (uint64_t(hv) * inverse_) | (((hv >> 64) * inverse2_) << 64);
    }
};

template<typename Type, typename...Types>
struct CEIFused: public CEIFused<Types...> {
    using CEIFused<Types...>::inverse;
    using CEIFused<Types...>::operator();
    Type op;
    template<typename...Args>
    constexpr CEIFused(Args &&...) {}
    template<typename T>
    INLINE constexpr T operator()(T h) const {
        return CEIFused<Types...>::operator()(op(h));
    }
    template<typename T>
    INLINE constexpr T inverse(T hv) const {
        return op.inverse(CEIFused<Types...>::inverse(hv));
    }
};
template<typename Type>
struct CEIFused<Type> {
    template<typename...Args>
    constexpr CEIFused(Args &&...) {}
    Type op;
    template<typename T>
    INLINE constexpr T operator()(T h) const {return op(h);}
    template<typename T>
    INLINE constexpr T inverse(T hv) const {return op.inverse(hv);}
};
template<typename T1, typename T2, typename T3>
struct CEIFused3: public CEIFused<T1, T2, T3> {};


struct CEHasher: public CEIFused3<CEIXOR<0x533f8c2151b20f97>, CEIMul<0x9a98567ed20c127d>, CEIXOR<0x691a9d706391077a>> {
template<typename...Args> constexpr CEHasher(Args &&...) {}
};

// Full-diffusion reversible candidates (2026-10 audit additions), built from
// the corrected primitives above. SplitMix64Hash passes the same flip-one-bit
// avalanche gate as WangHash (hashtest). NOTE: a single bare multiply does
// NOT avalanche: XorMultiply/MultiplyAdd/CEHasher map input bit 63 to output
// bit 63 alone, and a lone xorshift-multiply-xorshift (one multiply) leaves
// structurally biased cells (revhash_bench measures both). Two multiplies
// with interleaved xorshifts are the minimum structure that mixes fully.

// Invertible splitmix64-finalizer bijection (public-domain constants):
// xorshift-multiply with two odd multiplies. The reference point for 64-bit
// avalanche quality; the fastest invertible candidate in revhash_bench that
// matches WangHash's avalanche (and beats it on the biased-cell metric).
struct SplitMix64Hash: public CEIFused<RShiftXor<30>, CEIMul<0xbf58476d1ce4e5b9>, RShiftXor<27>, CEIMul<0x94d049bb133111eb>, RShiftXor<31>> {
    template<typename...Args> constexpr SplitMix64Hash(Args &&...) {}
};

// Reversible, runtime-configurable hashes


template<typename InvH1, typename InvH2>
struct FusedReversible {
    InvH1 op1;
    InvH2 op2;
    FusedReversible(uint64_t seed1=0, uint64_t seed2=1):
        op1(mthash(seed1) | 1), op2(mthash(seed2) | 1) {}
    template<typename T>
    INLINE T operator()(T h) const {return op2(op1(h));}
    template<typename T>
    INLINE T inverse(T hv) const {return op1.inverse(op2.inverse(hv));}
};
template<typename InvH1, typename InvH2, typename InvH3>
struct FusedReversible3 {
    InvH1 op1;
    InvH2 op2;
    InvH3 op3;
    FusedReversible3(uint64_t seed1=0x9a98567ed20c127d, uint64_t seed2=0xe37e28c4271b5a1duLL):
        op1(mthash(seed1) | 1), op2(mthash(seed2) | 1), op3(mthash((seed1 ^ seed2) + seed1) | 1)
    {}
    template<typename T>
    INLINE T operator()(T h) const {return op3(op2(op1(h)));}
    template<typename T>
    INLINE T inverse(T hv) const {
        return op1.inverse(op2.inverse(op3.inverse(hv)));
    }
};

using InvXor = InvH<op::bit_xor<uint64_t>>;
using InvMul = InvH<op::multiplies<uint64_t>>;
using InvAdd = InvH<op::plus<uint64_t>>;
//using InvAdd = InvH<op::plus<uint64_t>, op::minus<uint64_t>>;
template<size_t n>
using RotN = InvH<RotL<n>, RotR<n>>;
struct XorMultiply: public FusedReversible<InvXor, InvMul > {
    XorMultiply(uint64_t seed1=0x9a98567ed20c127d, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible<InvXor, InvMul >(seed1 | 1, seed2 | 1) {}
};
struct MultiplyAdd: public FusedReversible<InvMul, InvAdd> {
    MultiplyAdd(uint64_t seed1=0x9a98567ed20c127d, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible<InvMul, InvAdd>(seed1 | 1, seed2 | 1) {}
};
struct MultiplyAddXor: public FusedReversible3<InvAdd,InvMul,InvXor> {
    using base = FusedReversible3<InvAdd,InvMul,InvXor>;
    MultiplyAddXor(uint64_t seed1=0x9a98567ed20c127d, uint64_t seed2=0xe37e28c4271b5a1duLL): base(seed1 | 1, seed2 | 1) {}
};
template<size_t shift>
struct MultiplyAddXoRot: public FusedReversible3<InvMul,InvXor,RotN<shift>> {
    // NOTE (2026-10 audit): the constructor now forwards raw seeds like its
    // XorMultiply/MultiplyAdd/MultiplyAddXor siblings (FusedReversible3
    // applies the mthash exactly once). Previously it pre-mthashed the seeds
    // here and FusedReversible3 hashed again; generated streams change for
    // the same seed pair.
    MultiplyAddXoRot(uint64_t seed1=0x9a98567ed20c127d, uint64_t seed2=0xe37e28c4271b5a1duLL): FusedReversible3<InvMul,InvXor, RotN<shift>>(seed1 | 1, seed2 | 1) {}
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

// ---------------------------------------------------------------------------
// Triangularity and masked reduction (reduced-key decode for cqfbase_t / cht)
//
// Invertible hashes are the canonical choice for REDUCED-KEY techniques:
// cqfbase_t<...> (include/sketch/cqf.h) and cht take pre-hashed keys and
// split the hash value as parts(hv) = {(hv >> r) & (nslots - 1), hv & ((1 <<
// r) - 1)}: the TOP bits select the bucket, the LOW bits are the remainder.
// What invertibility buys depends on what the table stores:
//
//  * Full-width decode: if the table (or a companion index) keeps the full
//    H(x), the original key is x = H.inverse(H(x)) for ANY invertible H.
//  * Reduced-only decode: if the table stores ONLY the reduced bits, key
//    recovery additionally requires the stored bits to be SELF-INVERTIBLE.
//    A LEFT-TRIANGULAR hash has that property: its low k output bits depend
//    only on its low k input bits, so x mod 2^BITS is recoverable from
//    H(x) mod 2^BITS alone (e.g. the stored remainder). A RIGHT-TRIANGULAR
//    hash is the mirror image (high k output bits depend only on high k
//    input bits), which serves the bucket side: x's top bits are recoverable
//    from the bucket alone.
//
// Left-triangular ops: odd multiply (CEIMul, InvMul), add/xor of a constant
// (CEIXOR, InvAdd, InvXor, XORConstantHasher), LShiftXor (v ^ (v << n)).
//   -> left-triangular hashes: CEHasher, XorMultiply, MultiplyAdd,
//      MultiplyAddXor, TriangularReversibleHash, any CEIFused chain of the
//      above ops.
// Right-triangular ops: RShiftXor / InvRShiftXor (v ^ (v >> n)).
//   -> right-triangular hashes: RShiftXor chains, xor-of-constant hashes.
// Neither (full 64-bit diffusion required, use full-width decode):
//   WangHash, SplitMix64Hash, MurFinHash, anything containing Rot
//   (MultiplyAddXoRot). A RecursiveReversibleHash<H> has whatever
//   triangularity H has (e.g. XorMultiplyNVec is left-triangular).
//
// MaskedReducedHash<H, BITS> below is the documented masked-reduction
// pattern: reduced key = H{}(key) & ((1 << BITS) - 1). invert_low /
// invert_high provide the two reduced-only decode paths, statically guarded
// by the matching triangularity trait.
// ---------------------------------------------------------------------------
template<typename H>
struct is_left_triangular: std::false_type {};
template<typename H>
struct is_right_triangular: std::false_type {};
// left-triangular building blocks
template<uint64_t S> struct is_left_triangular<CEIMul<S>>: std::true_type {};
template<uint64_t S> struct is_left_triangular<CEIXOR<S>>: std::true_type {};
template<uint64_t S> struct is_left_triangular<XORConstantHasher<S>>: std::true_type {};
template<size_t n> struct is_left_triangular<LShiftXor<n>>: std::true_type {};
template<size_t n> struct is_left_triangular<InvLShiftXor<n>>: std::true_type {};
template<> struct is_left_triangular<InvMul>: std::true_type {};
template<> struct is_left_triangular<InvAdd>: std::true_type {};
template<> struct is_left_triangular<InvXor>: std::true_type {};
// right-triangular building blocks
template<size_t n> struct is_right_triangular<RShiftXor<n>>: std::true_type {};
template<size_t n> struct is_right_triangular<InvRShiftXor<n>>: std::true_type {};
// xor-by-constant is both
template<> struct is_right_triangular<InvXor>: std::true_type {};
template<uint64_t S> struct is_right_triangular<XORConstantHasher<S>>: std::true_type {};
// chains: triangular iff every op is triangular
template<typename T, typename...Ts>
struct is_left_triangular<CEIFused<T, Ts...>>: std::integral_constant<bool, is_left_triangular<T>::value && is_left_triangular<CEIFused<Ts...>>::value> {};
template<typename T> struct is_left_triangular<CEIFused<T>>: is_left_triangular<T> {};
template<typename T, typename...Ts>
struct is_right_triangular<CEIFused<T, Ts...>>: std::integral_constant<bool, is_right_triangular<T>::value && is_right_triangular<CEIFused<Ts...>>::value> {};
template<typename T> struct is_right_triangular<CEIFused<T>>: is_right_triangular<T> {};
template<typename A, typename B>
struct is_left_triangular<FusedReversible<A, B>>: std::integral_constant<bool, is_left_triangular<A>::value && is_left_triangular<B>::value> {};
template<typename A, typename B, typename C>
struct is_left_triangular<FusedReversible3<A, B, C>>: std::integral_constant<bool, is_left_triangular<A>::value && is_left_triangular<B>::value && is_left_triangular<C>::value> {};
template<typename A, typename B>
struct is_right_triangular<FusedReversible<A, B>>: std::integral_constant<bool, is_right_triangular<A>::value && is_right_triangular<B>::value> {};
template<typename A, typename B, typename C>
struct is_right_triangular<FusedReversible3<A, B, C>>: std::integral_constant<bool, is_right_triangular<A>::value && is_right_triangular<B>::value && is_right_triangular<C>::value> {};
// CEIFused3 and the named fused hashes delegate to their bases
template<typename T1, typename T2, typename T3>
struct is_left_triangular<CEIFused3<T1, T2, T3>>: is_left_triangular<CEIFused<T1, T2, T3>> {};
template<typename T1, typename T2, typename T3>
struct is_right_triangular<CEIFused3<T1, T2, T3>>: is_right_triangular<CEIFused<T1, T2, T3>> {};
template<> struct is_left_triangular<CEHasher>: is_left_triangular<CEIFused3<CEIXOR<0x533f8c2151b20f97>, CEIMul<0x9a98567ed20c127d>, CEIXOR<0x691a9d706391077a>>> {};
template<> struct is_right_triangular<CEHasher>: is_right_triangular<CEIFused3<CEIXOR<0x533f8c2151b20f97>, CEIMul<0x9a98567ed20c127d>, CEIXOR<0x691a9d706391077a>>> {};
template<> struct is_left_triangular<XorMultiply>: is_left_triangular<FusedReversible<InvXor, InvMul>> {};
template<> struct is_right_triangular<XorMultiply>: is_right_triangular<FusedReversible<InvXor, InvMul>> {};
template<> struct is_left_triangular<MultiplyAdd>: is_left_triangular<FusedReversible<InvMul, InvAdd>> {};
template<> struct is_right_triangular<MultiplyAdd>: is_right_triangular<FusedReversible<InvMul, InvAdd>> {};
template<> struct is_left_triangular<MultiplyAddXor>: is_left_triangular<FusedReversible3<InvAdd, InvMul, InvXor>> {};
template<> struct is_right_triangular<MultiplyAddXor>: is_right_triangular<FusedReversible3<InvAdd, InvMul, InvXor>> {};
template<> struct is_left_triangular<SplitMix64Hash>: is_left_triangular<CEIFused<RShiftXor<30>, CEIMul<0xbf58476d1ce4e5b9>, RShiftXor<27>, CEIMul<0x94d049bb133111eb>, RShiftXor<31>>> {};
template<> struct is_right_triangular<SplitMix64Hash>: is_right_triangular<CEIFused<RShiftXor<30>, CEIMul<0xbf58476d1ce4e5b9>, RShiftXor<27>, CEIMul<0x94d049bb133111eb>, RShiftXor<31>>> {};

// Documented masked-reduction pattern for reduced-key tables (cqf/cht):
// reduced key = H{}(key) & ((1 << BITS) - 1).
template<class H, size_t BITS>
struct MaskedReducedHash {
    static_assert(BITS > 0 && BITS < sizeof(uint64_t) * CHAR_BIT, "BITS must be in [1, 64)");
    static constexpr size_t bits = BITS;
    static constexpr uint64_t mask() {return (uint64_t(1) << BITS) - 1;}
    H h_;
    template<typename...Args> MaskedReducedHash(Args &&...args): h_(std::forward<Args>(args)...) {}
    uint64_t operator()(uint64_t key) const {return h_(key) & mask();}
    // Full-width decode: requires the UNMASKED hash value to be stored.
    uint64_t invert_full(uint64_t hval) const {return h_.inverse(hval);}
    // Reduced-only decode (the remainder path): recovers key mod 2^BITS from
    // the stored BITS low bits of H(key) alone. Valid iff H is left-triangular
    // (then H and its inverse act within the low BITS bits).
    uint64_t invert_low(uint64_t reduced) const {
        static_assert(is_left_triangular<H>::value, "invert_low requires a left-triangular hash (see the triangularity block above)");
        return h_.inverse(reduced) & mask();
    }
    // Reduced-only decode (the bucket path): recovers the key's BITS top bits
    // from the stored top BITS bits of H(key) alone. Valid iff H is right-triangular.
    uint64_t invert_high(uint64_t reduced_top) const {
        static_assert(is_right_triangular<H>::value, "invert_high requires a right-triangular hash (see the triangularity block above)");
        return h_.inverse(reduced_top << (sizeof(uint64_t) * CHAR_BIT - BITS)) >> (sizeof(uint64_t) * CHAR_BIT - BITS);
    }
};

// Left-triangular reversible hash for reduced-key decode: odd multiply then
// LShiftXor (both left-triangular), so x mod 2^BITS is recoverable from
// h(x) mod 2^BITS alone via MaskedReducedHash<..., BITS>::invert_low.
// NOTE: triangularity precludes full avalanche (high input bits stay local);
// that is the deliberate quality trade for reduced-only decode.
template<uint64_t mulseed=0x9a98567ed20c127d, size_t shift=23>
using TriangularReversibleHash = CEIFused<CEIMul<mulseed>, LShiftXor<shift>>;

struct XXH3PairHasher {
    template<typename CType>
    uint64_t hash(uint64_t x, CType count) const {
       return uint64_t(XXH3_64bits_withSeed(&x, sizeof(x), count));
    }
    template<typename CType>
    uint64_t operator()(uint64_t x, CType count) const {
        return uint64_t(XXH3_64bits_withSeed(&x, sizeof(x), count));
    }
};
template<typename FT=double>
struct Gamma21 {
    mutable std::uniform_real_distribution<FT> urd_;
    // use mutable so that the member function can be const
    template<typename Gen>
    FT operator()(Gen &x) const {
        FT ret = urd_(x);
        ret *= urd_(x);
        ret = -std::log(ret);
        return ret;
    }
};
} // namespace hash
} // namespace sketch

#endif /* #ifndef DBSKETCH_HASH_H__ */
