#ifndef AESCTR_H
#define AESCTR_H

// Taken from https://github.com/lemire/testingRNG
// Added C++ interface compatible with std::shuffle, &c.

// contributed by Samuel Neves

#include <cassert>
#include <cstddef>
#include <limits>
#include <cstdint>
#include <cstring>
#include <array>
#include <type_traits>
#include <immintrin.h>

#if __cplusplus >= 201703L
#define AES_MAYBE_UNUSED [[maybe_unused]]
#else
#define AES_MAYBE_UNUSED
#endif

#ifndef TYPES_TEMPLATES
#define TYPES_TEMPLATES
namespace types {
    template<typename T>
    struct is_integral: std::false_type {};
    template<>struct is_integral<unsigned char>: std::true_type {};
    template<>struct is_integral<signed char>: std::true_type {};
    template<>struct is_integral<unsigned short>: std::true_type {};
    template<>struct is_integral<signed short>: std::true_type {};
    template<>struct is_integral<unsigned int>: std::true_type {};
    template<>struct is_integral<signed int>: std::true_type {};
    template<>struct is_integral<unsigned long>: std::true_type {};
    template<>struct is_integral<signed long>: std::true_type {};
    template<>struct is_integral<unsigned long long>: std::true_type {};
    template<>struct is_integral<signed long long>: std::true_type {};
#if __cplusplus >= 201703L
    template<class T> inline constexpr bool is_integral_v = is_integral<T>::value;
#endif

    template<typename T> struct is_simd: std::false_type {};
    template<typename T> struct is_simd_int: std::false_type {};
    template<typename T> struct is_simd_float: std::false_type {};

#if __SSE2__
    template<>struct is_simd<__m128i>: std::true_type {};
    template<>struct is_simd<__m128>:  std::true_type {};
    template<>struct is_simd_int<__m128i>: std::true_type {};
    template<>struct is_simd_float<__m128>: std::true_type {};
#endif
#if __AVX2__
    template<>struct is_simd<__m256i>: std::true_type {};
    template<>struct is_simd<__m256>:  std::true_type {};
    template<>struct is_simd_int<__m256i>: std::true_type {};
    template<>struct is_simd_float<__m256>: std::true_type {};
#endif
#if __AVX512__
    template<>struct is_simd<__m512i>: std::true_type {};
    template<>struct is_simd<__m512>:  std::true_type {};
    template<>struct is_simd_int<__m512i>: std::true_type {};
    template<>struct is_simd_float<__m512>: std::true_type {};
#endif
#if __cplusplus >= 201703L
    template<class T> inline constexpr bool is_simd_v = is_simd<T>::value;
    template<class T> inline constexpr bool is_simd_int_v = is_simd_int<T>::value;
    template<class T> inline constexpr bool is_simd_float_v = is_simd_float<T>::value;
#endif
} // namespace types
#endif



namespace aes {

using std::uint64_t;
using std::uint8_t;
using std::size_t;


#define AES_ROUND(rcon, index)                                                 \
  do {                                                                         \
    __m128i k2 = _mm_aeskeygenassist_si128(k, rcon);                           \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                                \
    k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3, 3, 3, 3)));      \
    seed_[index] = k;                                                    \
  } while (0)

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)
#endif
#if HAS_AVX_512
#define VEC_ALIGNMENT_FOR_BUFFER 64
#elif __AVX2__
#define VEC_ALIGNMENT_FOR_BUFFER 32
#else
#define VEC_ALIGNMENT_FOR_BUFFER 16
#endif



template<typename GeneratedType=uint64_t, size_t UNROLL_COUNT=4,
         typename=typename std::enable_if<
            types::is_integral<GeneratedType>::value || types::is_simd_int<GeneratedType>::value
            >::type
        >
class AesCtr {
    static const size_t AESCTR_ROUNDS = 10;
    uint8_t state_[sizeof(__m128i) * UNROLL_COUNT] __attribute__ ((aligned (VEC_ALIGNMENT_FOR_BUFFER)));
    __m128i ctr_[UNROLL_COUNT];
    __m128i seed_[AESCTR_ROUNDS + 1];
    __m128i work[UNROLL_COUNT];
    unsigned offset_;

    // Unrollers
    template<size_t ind, size_t todo>
    struct aes_unroll_impl {
        void operator()(__m128i *ret, AesCtr &state) const {
            ret[ind] = _mm_xor_si128(state.ctr_[ind], state.seed_[0]);
            aes_unroll_impl<ind + 1, todo - 1>()(ret, state);
        }
        void aesenc(__m128i *ret, __m128i subkey) const {
            ret[ind] = _mm_aesenc_si128(ret[ind], subkey);
            aes_unroll_impl<ind + 1, todo - 1>().aesenc(ret, subkey);
        }
        template<size_t NUMROLL>
        void round_and_enc(__m128i *ret, AesCtr &state) const {
            const __m128i subkey = state.seed_[ind];
            aes_unroll_impl<0, NUMROLL>().aesenc(ret, subkey);
            aes_unroll_impl<ind + 1, todo - 1>().template round_and_enc<NUMROLL>(ret, state);
        }
        void add_store(__m128i *work, AesCtr &state) const {
          state.ctr_[ind] =
              _mm_add_epi64(state.ctr_[ind], _mm_set_epi64x(0, UNROLL_COUNT));
              _mm_store_si128(
                  reinterpret_cast<__m128i *>(&state.state_[16 * ind]),
                  _mm_aesenclast_si128(work[ind], state.seed_[AESCTR_ROUNDS]));
          aes_unroll_impl<ind + 1, todo - 1>().add_store(work, state);
        }
    };
    // Termination conditions
    template<size_t ind>
    struct aes_unroll_impl<ind, 0> {
        void operator()(AES_MAYBE_UNUSED __m128i *ret, AES_MAYBE_UNUSED AesCtr &state) const {}
        void aesenc(AES_MAYBE_UNUSED __m128i *ret, AES_MAYBE_UNUSED __m128i subkey) const {}
        template<size_t NUMROLL>
        void round_and_enc(AES_MAYBE_UNUSED __m128i *ret, AES_MAYBE_UNUSED AesCtr &state) const {}
        void add_store(AES_MAYBE_UNUSED __m128i *work, AES_MAYBE_UNUSED AesCtr &state) const {}
    };

public:
    using result_type = GeneratedType;
    constexpr AesCtr(uint64_t seedval=0) {
        seed(seedval);
    }
    void generate_new_values() {
        aes_unroll_impl<0, UNROLL_COUNT>()(work, *this);
        aes_unroll_impl<1, AESCTR_ROUNDS - 1>().template round_and_enc<UNROLL_COUNT>(work, *this);
        aes_unroll_impl<0, UNROLL_COUNT>().add_store(work, *this);
        offset_ = 0;
    }
    result_type operator()() {
        if (__builtin_expect(offset_ >= sizeof(__m128i) * UNROLL_COUNT, 0))
            generate_new_values(); // sets offset_ to 0.
        result_type ret;
        std::memcpy(&ret, &state_[offset_], sizeof(ret));
        offset_ += sizeof(result_type);
        return ret;
    }
    static constexpr result_type max() {return std::numeric_limits<result_type>::max();}
    static constexpr result_type min() {return std::numeric_limits<result_type>::min();}
    void seed(uint64_t k) {
        seed(_mm_set_epi64x(0, k));
    }
    void seed(__m128i k) {
      seed_[0] = k;
      // D. Lemire manually unrolled following loop since _mm_aeskeygenassist_si128
      // requires immediates

      AES_ROUND(0x01, 1);
      AES_ROUND(0x02, 2);
      AES_ROUND(0x04, 3);
      AES_ROUND(0x08, 4);
      AES_ROUND(0x10, 5);
      AES_ROUND(0x20, 6);
      AES_ROUND(0x40, 7);
      AES_ROUND(0x80, 8);
      AES_ROUND(0x1b, 9);
      AES_ROUND(0x36, 10);

      for (unsigned i = 0; i < UNROLL_COUNT; ++i) ctr_[i] = _mm_set_epi64x(0, i);
      offset_ = sizeof(__m128i) * UNROLL_COUNT;
    }
    result_type operator[](size_t count) const {
        static constexpr unsigned DIV   = sizeof(__m128i) / sizeof(result_type);
        static constexpr unsigned BMASK = DIV - 1;
        const unsigned offset_(count & BMASK);
        result_type ret[DIV];
        count /= DIV;
        __m128i tmp(_mm_xor_si128(_mm_set_epi64x(0, count), seed_[0]));
        for (unsigned r = 1; r <= AESCTR_ROUNDS - 1; tmp = _mm_aesenc_si128(tmp, seed_[r++]));
        _mm_store_si128(reinterpret_cast<__m128i *>(ret), _mm_aesenclast_si128(tmp, seed_[AESCTR_ROUNDS]));
        return ret[offset_];
    }
    static constexpr size_t BUFSIZE = sizeof(state_);
    const uint8_t *buf() const {return &state_[0];}
    using ThisType = AesCtr<GeneratedType, UNROLL_COUNT>;

    template<typename T, bool manual_override=false,
             typename=typename std::enable_if<
                manual_override || types::is_integral<T>::value || types::is_simd_int<T>::value
                >::type
             >
    class buffer_view {
        ThisType &ref;
    public:
        buffer_view(ThisType &ctr): ref{ctr} {}
        using const_pointer = const T *;
        using pointer       = T *;
        const_pointer cbegin() const {
            return reinterpret_cast<const_pointer>(&ref.state_[0]);
        }
        const_pointer cend() const {
            return reinterpret_cast<const_pointer>(&ref.state_[BUFSIZE]);
        }
        pointer begin() {
            return reinterpret_cast<pointer>(&ref.state_[0]);
        }
        pointer end() {
            return reinterpret_cast<pointer>(&ref.state_[BUFSIZE]);
        }
    };
    template<typename T, bool manual_override=false>
    buffer_view<T, manual_override> view() {return buffer_view<T, manual_override>(*this);}
};
#undef AES_ROUND


template<typename size_type, size_t arrsize>
constexpr std::array<size_type, arrsize> seed_to_array(size_type seedseed) {
    std::array<size_type, arrsize> ret{};
    aes::AesCtr<size_type> gen(seedseed);
    for(auto &el: ret) el = gen();
    return ret;
}

template<typename T>
struct is_aes: std::false_type {};

template<typename T, size_t n>
struct is_aes<AesCtr<T, n>>: std::true_type {};

} // namespace aes

#undef AESCTR_UNROLL
#undef AESCTR_ROUNDS
#undef AES_MAYBE_UNUSED

#endif
