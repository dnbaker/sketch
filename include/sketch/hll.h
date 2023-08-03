#ifndef HLL_H_
#define HLL_H_
#ifndef NO_SLEEF
#define NO_SLEEF
#endif
#include "integral.h"
#include "vec/vec.h"
#include "common.h"
#include "hash.h"
#include "hedley.h"

namespace sketch {

inline namespace minhash {
template<typename HS>
class WideHyperLogLogHasher;

}

inline namespace hll {
namespace detail {
template<typename FloatType>
static constexpr FloatType gen_sigma(FloatType x) {
    if(x == 1.) return std::numeric_limits<FloatType>::infinity();
    FloatType z(x);
    for(FloatType zp(0.), y(1.); z != zp;) {
        x *= x; zp = z; z += x * y; y += y;
        if(std::isnan(z)) {
#ifndef __CUDACC__
            std::fprintf(stderr, "[W:%s:%d] Reached nan. Returning the last usable number.\n", __PRETTY_FUNCTION__, __LINE__);
#endif
            return zp;
        }
    }
    return z;
}
template<typename FloatType>
static constexpr FloatType gen_tau(FloatType x) {
    if (x == 0. || x == 1.) {
        return 0.;
    }
    FloatType z(1-x), tmp(0.), y(1.), zp(x);
    while(zp != z) {
        x = std::sqrt(x);
        zp = z;
        y *= 0.5;
        tmp = (1. - x);
        z -= tmp * tmp * y;
    }
    return z / 3.;
}
// Based off https://github.com/oertl/hyperloglog-sketch-estimation-paper/blob/master/c%2B%2B/cardinality_estimation.hpp

} /* detail */ } /* hll */ } /* sketch */




namespace sketch {
inline namespace hll {
enum EstimationMethod: uint8_t {
    ORIGINAL       = 0,
    ERTL_IMPROVED  = 1,
    ERTL_MLE       = 2
};

#ifndef VEC_DISABLED__
using Type = typename vec::SIMDTypes<uint64_t>::Type;
using VType = typename vec::SIMDTypes<uint64_t>::VType;
#endif
using hash::WangHash;
using hash::MurFinHash;

static constexpr const char *JESTIM_STRINGS []
{
    "ORIGINAL", "ERTL_IMPROVED", "ERTL_MLE", "ERTL_JOINT_MLE"
};
enum JointEstimationMethod: uint8_t {
    J_ORIGINAL       = ORIGINAL,
    J_ERTL_IMPROVED  = ERTL_IMPROVED, // Improved but biased method
    J_ERTL_MLE       = ERTL_MLE, // element-wise max, followed by MLE
    ERTL_JOINT_MLE = 3, // Ertl special version
    J_ERTL_JOINT_MLE = ERTL_JOINT_MLE,
};

static inline std::string to_string(JointEstimationMethod est) {
    switch(est) {case J_ORIGINAL: return "Original"; case J_ERTL_IMPROVED: return "Improved"; case J_ERTL_MLE: return "MLE"; case J_ERTL_JOINT_MLE: return "JMLE"; default: return "UNKNOWN";};
}
static inline std::string to_string(EstimationMethod est) {
    return to_string(static_cast<JointEstimationMethod>(est));
}


static const char *EST_STRS [] {
    "original",
    "ertl_improved",
    "ertl_mle",
    "ertl_joint_mle"
};

#ifdef MANUAL_CHECKS
#  ifndef VERIFY_SUM
#    define VERIFY_SUM 1
#  endif
#  ifndef VERIFY_SIMD_JOINT
#    define VERIFY_SIMD_JOINT 1
#  endif
#  ifndef VERIFY_SIMD
#    define VERIFY_SIMD 1
#  endif
#endif


using CountArrayType = std::array<uint32_t, 64>;

namespace detail {
template<typename T>
static double ertl_ml_estimate(const T& c, unsigned p, unsigned q, double relerr=1e-2); // forward declaration
template<typename Container>
inline std::array<uint32_t, 64> sum_counts(const Container &con);
}
#if VERIFY_SIMD_JOINT
/*
 *Returns the estimated number of elements:
 * [0] uniquely in h1
 * [1] uniquely in h2
 * [2] in the intersection
 * size of the union is [0] + [1] + [2]
 * size of the intersection is [2]
 */
template<typename IType, size_t N, typename=typename std::enable_if<std::is_integral<IType>::value>::type>
std::string counts2str(const std::array<IType, N> &arr) {
    std::string ret;
    for(const auto el: arr) {
        ret += std::to_string(el);
        ret += ',';
    }
    ret.pop_back();
    return ret;
}

template<typename HllType>
std::array<double, 3> ertl_joint_simple(const HllType &h1, const HllType &h2) {
    using detail::ertl_ml_estimate;
    std::array<double, 3> ret;
    auto p = h1.p();
    auto q = h1.q();
    auto c1 = detail::sum_counts(h1.core());
    auto c2 = detail::sum_counts(h2.core());
    const double cAX = ertl_ml_estimate(c1, h1.p(), h1.q());
    const double cBX = ertl_ml_estimate(c2, h2.p(), h2.q());
    //std::fprintf(stderr, "cAX ml est: %lf. cBX ml els: %lf\n", cAX, cBX);
    //const double cBX = hl2.creport();
    //const double cBX = hl2.creport();
    auto tmph = h1 + h2;
    const double cABX = tmph.report();
    std::array<uint32_t, 64> countsAXBhalf{0}, countsBXAhalf{0};
    countsAXBhalf[q] = countsBXAhalf[q] = h1.m();
    std::array<uint32_t, 64> cg1{0}, cg2{0}, ceq{0};
    {
        const auto &core1(h1.core()), &core2(h2.core());
        for(uint64_t i(0); i < core1.size(); ++i) {
            switch((core1[i] > core2[i]) << 1 | (core2[i] > core1[i])) {
                case 0:
                    ++ceq[core1[i]]; break;
                case 1:
                    ++cg2[core2[i]];
                    break;
                case 2:
                    ++cg1[core1[i]];
                    break;
                default:
                    HEDLEY_UNREACHABLE();
            }
        }
    }
    for(unsigned _q = 0; _q < q; ++_q) {
        // Handle AXBhalf
        countsAXBhalf[_q] = cg1[_q] + ceq[_q] + cg2[_q + 1];
        assert(countsAXBhalf[q] >= countsAXBhalf[_q]);
        countsAXBhalf[q] -= countsAXBhalf[_q];

        // Handle BXAhalf
        countsBXAhalf[_q] = cg2[_q] + ceq[_q] + cg1[_q + 1];
        assert(countsBXAhalf[q] >= countsBXAhalf[_q]);
        countsBXAhalf[q] -= countsBXAhalf[_q];
    }
    double cAXBhalf = ertl_ml_estimate(countsAXBhalf, p, q - 1);
    double cBXAhalf = ertl_ml_estimate(countsBXAhalf, p, q - 1);
    ret[0] = cABX - cBX;
    ret[1] = cABX - cAX;
    double cX1 = (1.5 * cBX + 1.5*cAX - cBXAhalf - cAXBhalf);
    double cX2 = 2.*(cBXAhalf + cAXBhalf) - 3.*cABX;
    ret[2] = std::max(0., 0.5 * (cX1 + cX2));
#if !NDEBUG
    std::fprintf(stderr, "Halves of contribution: %lf, %lf. Initial est: %lf. Result: %lf\n", cX1, cX2, cABX, ret[2]);
#endif
    return ret;
}
#endif


namespace detail {
    // Miscellaneous requirements.
static constexpr double LARGE_RANGE_CORRECTION_THRESHOLD = (1ull << 32) / 30.;
static constexpr double TWO_POW_32 = 1ull << 32;



template<typename CountArrType>
static double calculate_estimate(const CountArrType &counts,
                                 JointEstimationMethod estim, uint64_t m, uint32_t p, double alpha, double relerr=1e-2) noexcept {
    assert(estim <= 3);
#if ENABLE_COMPUTED_GOTO
    static constexpr void *arr [] {&&ORREST, &&ERTL_IMPROVED_EST, &&ERTL_MLE_EST, &&ERTL_JOINT_MLE_EST};
    goto *arr[estim];
    ORREST: {
#else
    switch(estim) {
        case J_ORIGINAL: {
#endif
        assert(estim != static_cast<JointEstimationMethod>(ERTL_MLE));
        double sum = counts[0];
        for(unsigned i = 1; i < 64 - p + 1; ++i) if(counts[i]) sum += std::ldexp(counts[i], -i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
        //for(unsigned i = 1; i < 64 - p + 1; ++i) sum += std::ldexp(counts[i], -i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
        double value(alpha * m * m / sum);
        if(value < 2.5 * m) {
            if(counts[0]) {
                value = m * std::log(static_cast<double>(m) / counts[0]);
            }
        } else if(value > detail::LARGE_RANGE_CORRECTION_THRESHOLD) {
            // Reuse sum variable to hold correction.
            // I do think I've seen worse accuracy with the large range correction, but I would need to rerun experiments to be sure.
            sum = -std::pow(2.0L, 32) * std::log1p(-std::ldexp(value, -32));
            if(!std::isnan(sum)) value = sum;
        }
        return value;
    }
#if ENABLE_COMPUTED_GOTO
    ERTL_IMPROVED_EST:
#else
        case J_ERTL_IMPROVED:
#endif
    {
        static const double divinv = 1. / (2.L*std::log(2.L));
        double z = m * detail::gen_tau(static_cast<double>((m-counts[64 - p + 1]))/static_cast<double>(m));
        for(unsigned i = 64-p; i; z += counts[i--], z *= 0.5); // Reuse value variable to avoid an additional allocation.
        z += m * detail::gen_sigma(static_cast<double>(counts[0])/static_cast<double>(m));
        return m * divinv * m / z;
    }
#if ENABLE_COMPUTED_GOTO
    ERTL_MLE_EST:
    ERTL_JOINT_MLE_EST:
#else
    case J_ERTL_MLE:
    case J_ERTL_JOINT_MLE:
#endif
        return ertl_ml_estimate(counts, p, 64 - p, relerr);
    default:
        std::fprintf(stderr, "Unknown estimation method.\n");
        HEDLEY_UNREACHABLE();
    }
}

template<typename CountArrType>
static double calculate_estimate(const CountArrType &counts,
                                 EstimationMethod estim, uint64_t m, uint32_t p, double alpha, double relerr=1e-2) noexcept {
    return calculate_estimate(counts, static_cast<JointEstimationMethod>(estim), m, p, alpha, relerr);
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<uint64_t> *counts_; // Array decayed to pointer.
    const CoreType          &core_;
    const uint64_t              l_;
    const uint64_t             pb_; // Per-batch
};


#if !defined(VEC_DISABLED__) && !defined(__aarch64__)
union SIMDHolder {

public:

#define DEC_MAX(fn) static constexpr decltype(&fn) max_fn = &fn
#define DEC_MAX16(fn) static constexpr decltype(&fn) max_fn16 = &fn
#define DEC_MAX32(fn) static constexpr decltype(&fn) max_fn32 = &fn
#define DEC_MAX64(fn) static constexpr decltype(&fn) max_fn64 = &fn
#define DEC_GT(fn)  static constexpr decltype(&fn) gt_fn  = &fn
#define DEC_EQ(fn)  static constexpr decltype(&fn) eq_fn  = &fn

// vpcmpub has roughly the same latency as
// vpcmpOPub, but twice the throughput, so we use it instead.
// (the *epu8_mask version over the *epu8_mask)
#if HAS_AVX_512
    using SType    = __m512i;
    DEC_MAX16(_mm512_max_epu16);
    DEC_MAX32(_mm512_max_epu32);
    DEC_MAX64(_mm512_max_epu64);
#  if __AVX512BW__
    DEC_MAX(_mm512_max_epu8);
    using MaskType = __mmask64;
    static_assert(sizeof(MaskType) == sizeof(__mmask64), "Mask type should be 64-bits in size.");
    static_assert(sizeof(__mmask64) == sizeof(unsigned long long), "Mask type should be 64-bits in size.");
    DEC_GT(_mm512_cmpgt_epu8_mask);
    DEC_EQ(_mm512_cmpeq_epu8_mask);
#  else
    __m256i subs[2];
    using MaskType = SIMDHolder;
    static SIMDHolder gt_fn(__m512i a, __m512i b) {
        SIMDHolder ret;
        ret.subs[0] = _mm256_cmpgt_epi8(*(__m256i *)(&a), *(__m256i *)(&b));
        ret.subs[1] = _mm256_cmpgt_epi8(*(__m256i *)(((uint8_t *)&a) + 32), *(__m256i *)(((uint8_t *)&b) + 32));
#if !NDEBUG
        SIMDHolder ac(a), bc(b);
        for(unsigned i(0); i < sizeof(ret); ++i) {
            assert(!!ret.vals[i] == !!(ac.vals[i] > bc.vals[i]));
        }
#endif
        return ret;
    }
    static SIMDHolder max_fn(__m512i a, __m512i b) {
        SIMDHolder ret;
        ret.subs[0] = _mm256_max_epu8(*(__m256i *)(&a), *(__m256i *)(&b));
        ret.subs[1] = _mm256_max_epu8(*(__m256i *)(((uint8_t *)&a) + 32), *(__m256i *)(((uint8_t *)&b) + 32));
        return ret;
    }
    static SIMDHolder eq_fn(__m512i a, __m512i b) {
        SIMDHolder ret;
        ret.subs[0] = _mm256_cmpeq_epi8(*(__m256i *)(&a), *(__m256i *)(&b));
        ret.subs[1] = _mm256_cmpeq_epi8(*(__m256i *)(((uint8_t *)&a) + 32), *(__m256i *)(((uint8_t *)&b) + 32));
        return ret;
    }
#  endif
#elif __AVX2__
    using SType = __m256i;
    using MaskType = SIMDHolder;
    DEC_MAX(_mm256_max_epu8);
    DEC_MAX16(_mm256_max_epu16);
    DEC_MAX32(_mm256_max_epu32);
    DEC_EQ (_mm256_cmpeq_epi8);
    DEC_GT (_mm256_cmpgt_epi8);
    uint64_t sub64s[sizeof(__m256i) / sizeof(uint64_t)];
    static SIMDHolder max_fn64(__m256i a, __m256i b) {
        SIMDHolder ret;
        for(unsigned i = 0; i < sizeof(__m256i) / sizeof(uint64_t); ++i)
            ret.sub64s[i] = std::max(((uint64_t *)&a)[i], ((uint64_t *)&b)[i]);
        return ret;
    }
    //DEC_GT(_mm256_cmpgt_epu8_mask);
#elif __SSE4_1__
    using SType = __m128i;
    using MaskType = SIMDHolder;
    DEC_MAX(_mm_max_epu8);
    DEC_MAX16(_mm_max_epu16);
    DEC_MAX32(_mm_max_epu32);
    DEC_GT(_mm_cmpgt_epi8);
    DEC_EQ(_mm_cmpeq_epi8);
    uint64_t sub64s[sizeof(__m128i) / sizeof(uint64_t)];
    static SIMDHolder max_fn64(__m128i a, __m128i b) {
        SIMDHolder ret;
        for(unsigned i = 0; i < sizeof(__m128i) / sizeof(uint64_t); ++i)
            ret.sub64s[i] = std::max(((uint64_t *)&a)[i], ((uint64_t *)&b)[i]);
        return ret;
    }
#else
#  error("Need at least SSE4.1")
#endif
#undef DEC_MAX
#undef DEC_GT
#undef DEC_EQ
#undef DEC_MAX16
#undef DEC_MAX32
#undef DEC_MAX64

    SIMDHolder() {} // empty constructor
    SIMDHolder(SType val_) {val = val_;}
    operator SType &() {return val;}
    operator const SType &() const {return val;}
    static constexpr size_t nels  = sizeof(SType) / sizeof(uint8_t);
    static constexpr size_t nel16s  = sizeof(SType) / sizeof(uint16_t);
    static constexpr size_t nel32s  = sizeof(SType) / sizeof(uint32_t);
    static constexpr size_t nel64s  = sizeof(SType) / sizeof(uint64_t);
    static constexpr size_t nbits = sizeof(SType) / sizeof(uint8_t) * CHAR_BIT;
    using u8arr = uint8_t[nels];
    using u16arr = uint16_t[nels / 2];
    using u32arr = uint32_t[nels / 4];
    using u64arr = uint64_t[nels / 8];
    SType val;
    u8arr vals;
    u16arr val16s;
    u32arr val32s;
    u64arr val64s;
    template<typename T>
    void inc_counts(T &arr) const {
        unroller<T, 0, nels> ur;
        ur(*this, arr);
    }
    template<typename T>
    void inc_counts_lut(T &arr, int jump) const {
#if ENABLE_COMPUTED_GOTO
        static constexpr void *labels [] {&&B8, &&B16, &&B32, &&B64};
        goto *labels[jump];
        B8: inc_counts(arr);    return;
        B16: inc_counts16(arr); return;
        B32: inc_counts32(arr); return;
        B64: inc_counts64(arr); return;
#else
        switch(jump) {
            case 0: inc_counts(arr);   break;
            case 1: inc_counts16(arr); break;
            case 2: inc_counts32(arr); break;
            case 3: inc_counts64(arr); break;
            default: HEDLEY_UNREACHABLE();
        }
#endif
    }
#ifndef VEC_DISABLED__
    template<typename IT, typename T>
    void inc_counts_by_type(T &arr) const {
        CONST_IF(sizeof(IT) == 1) inc_counts(arr);
        else CONST_IF(sizeof(IT) == 2) inc_counts16(arr);
        else CONST_IF(sizeof(IT) == 4) inc_counts32(arr);
        else CONST_IF(sizeof(IT) == 8) inc_counts64(arr);
        else throw std::runtime_error(std::string("Unsupported type of size: ") + std::to_string(sizeof(IT)));
    }
#endif
    template<typename T, size_t iternum, size_t niter_left> struct unroller {
        void operator()(const SIMDHolder &ref, T &arr) const {
            ++arr[ref.vals[iternum]];
            unroller<T, iternum+1, niter_left-1>()(ref, arr);
        }
        void op16(const SIMDHolder &ref, T &arr) const {
            ++arr[ref.val16s[iternum]];
            unroller<T, iternum+1, niter_left-1>().op16(ref, arr);
        }
        void op32(const SIMDHolder &ref, T &arr) const {
            ++arr[ref.val32s[iternum]];
            unroller<T, iternum+1, niter_left-1>().op32(ref, arr);
        }
        void op64(const SIMDHolder &ref, T &arr) const {
            ++arr[ref.val64s[iternum]];
            unroller<T, iternum+1, niter_left-1>().op64(ref, arr);
        }
    };
    template<typename T, size_t iternum> struct unroller<T, iternum, 0> {
        void operator()(const SIMDHolder &, T &) const {}
        void op16(const SIMDHolder &, T &) const {}
        void op32(const SIMDHolder &, T &) const {}
        void op64(const SIMDHolder &, T &) const {}
    };
#define DEC_INC(nbits)\
    template<typename T>\
    void inc_counts##nbits (T &arr) const {\
        static_assert(std::is_integral<std::decay_t<decltype(arr[0])>>::value, "Counts must be integral.");\
        unroller<T, 0, nel##nbits##s> ur;\
        ur.op##nbits(*this, arr);\
    }
    DEC_INC(16)
    DEC_INC(32)
    DEC_INC(64)
#undef DEC_INC
    static_assert(sizeof(SType) == sizeof(u8arr), "both items in the union must have the same size");
};
#endif

struct joint_unroller {
#ifndef VEC_DISABLED__
    using MType = SIMDHolder::MaskType;
    using SType = SIMDHolder::SType;
    // Woof....
#if defined(__AVX512BW__)
    static_assert(sizeof(MType) == sizeof(uint64_t), "Must be 64 bits");
#endif
    template<typename T, size_t iternum, size_t niter_left> struct ju_impl {
    INLINE void operator()(const SIMDHolder &ref1, const SIMDHolder &ref2, const SIMDHolder &u, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq, MType gtmask1, MType gtmask2, MType eqmask) const {
            ++arrh1[ref1.vals[iternum]];
            ++arrh2[ref2.vals[iternum]];
            ++arru[u.vals[iternum]];
#if __AVX512BW__
            arrg1[ref1.vals[iternum]] += gtmask1 & 1;
            arreq[ref1.vals[iternum]] += eqmask  & 1;
            arrg2[ref2.vals[iternum]] += gtmask2 & 1;
            gtmask1 >>= 1; gtmask2 >>= 1; eqmask  >>= 1;
            // TODO: Consider packing these into an SIMD type and shifting them as a set.
#else
            static_assert(sizeof(MType) == sizeof(SIMDHolder), "Wrong size?");
            arrg1[ref1.vals[iternum]] += gtmask1.vals[iternum] != 0;
            arreq[ref1.vals[iternum]] += eqmask.vals [iternum] != 0;
            arrg2[ref2.vals[iternum]] += gtmask2.vals[iternum] != 0;
#endif
            ju_impl<T, iternum+1, niter_left-1> ju;
            ju(ref1, ref2, u, arrh1, arrh2, arru, arrg1, arrg2, arreq, gtmask1, gtmask2, eqmask);
        }
    };
    template<typename T, size_t iternum> struct ju_impl<T, iternum, 0> {
        INLINE void operator()(const SIMDHolder &, const SIMDHolder &, const SIMDHolder &, T &, T &, T &, T &, T &, T &, MType, MType, MType) const {}
    };
    template<typename T>
    INLINE void operator()(const SIMDHolder &ref1, const SIMDHolder &ref2, const SIMDHolder &u, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq) const {
        ju_impl<T, 0, SIMDHolder::nels> ju;
#if __AVX512BW__
        auto g1 = SIMDHolder::gt_fn(ref1.val, ref2.val);
        auto g2 = SIMDHolder::gt_fn(ref2.val, ref1.val);
        auto eq = SIMDHolder::eq_fn(ref1.val, ref2.val);
#else
        auto g1 = SIMDHolder(SIMDHolder::gt_fn(ref1.val, ref2.val));
        auto g2 = SIMDHolder(SIMDHolder::gt_fn(ref2.val, ref1.val));
        auto eq = SIMDHolder(SIMDHolder::eq_fn(ref1.val, ref2.val));
#endif
        static_assert(std::is_same<MType, std::decay_t<decltype(g1)>>::value, "g1 should be the same time as MType");
        ju(ref1, ref2, u, arrh1, arrh2, arru, arrg1, arrg2, arreq, g1, g2, eq);
    }
    template<typename T>
    INLINE void sum_arrays(const SType *arr1, const SType *arr2, const SType *const arr1end, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq) const {
        SIMDHolder v1, v2, u;
        do {
            v1.val = *arr1++;
            v2.val = *arr2++;
            u.val  = SIMDHolder::max_fn(v1.val, v2.val);
            this->operator()(v1, v2, u, arrh1, arrh2, arru, arrg1, arrg2, arreq);
        } while(arr1 < arr1end);
    }
    template<typename T, typename VectorType>
    INLINE void sum_arrays(const VectorType &c1, const VectorType &c2, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq) const {
        assert(c1.size() == c2.size() || !std::fprintf(stderr, "Sizes: %zu, %zu\n", c1.size(), c2.size()));
        assert((c1.size() & (SIMDHolder::nels - 1)) == 0);
        sum_arrays(reinterpret_cast<const SType *>(&c1[0]), reinterpret_cast<const SType *>(&c2[0]), reinterpret_cast<const SType *>(&*c1.cend()), arrh1, arrh2, arru, arrg1, arrg2, arreq);
    }
#else
    template<typename T, typename VectorType>
    INLINE void sum_arrays(const VectorType &c1, const VectorType &c2, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq) const {
        #pragma omp simd
        for(size_t i = 0; i < c1.size(); ++i) {
            const auto token1 = c1[i];
            const auto token2 = c2[i];
            const auto token_max = std::max(token1, token2);
            const bool gt = (token1 > token2);
            const bool lt = !gt;
            const bool eq = (token1 == token2);
            // Update base counts
            ++arrh1[token1]; ++arrh2[token2]; ++arru[token_max];
            // Update gt/eq counts
            arrg1[token1] += gt;
            arrg2[token2] += lt;
            arreq[token_max] += eq;
        }
    }
#endif /* VEC_DISABLED__*/
};

#ifndef VEC_DISABLED__
template<typename T>
inline void inc_counts(T &counts, const SIMDHolder *p, const SIMDHolder *pend) {
    static_assert(std::is_integral<std::decay_t<decltype(counts[0])>>::value, "Counts must be integral.");
    SIMDHolder tmp;
    do {
        tmp = *p++;
        tmp.inc_counts(counts);
    } while(p < pend);
}


static inline std::array<uint32_t, 64> sum_counts(const SIMDHolder *p, const SIMDHolder *pend) {
    // Should add Contiguous Container requirement.
    std::array<uint32_t, 64> counts{0};
    const size_t nelem = (uint8_t *)pend - (uint8_t *)p;
    if(nelem < sizeof(*p)) {
        const size_t e = (uint8_t *)pend - (uint8_t *)p;
        for(size_t i = 0; i < e; ++i) ++counts[((uint8_t *)p)[i]];
    } else {
        inc_counts(counts, p, pend);
    }
    return counts;
}

template<typename Container>
inline std::array<uint32_t, 64> sum_counts(const Container &con) {
    //static_assert(std::is_same<std::decay_t<decltype(con[0])>, uint8_t>::value, "Container must contain 8-bit unsigned integers.");
    return sum_counts(reinterpret_cast<const SIMDHolder *>(&*std::cbegin(con)), reinterpret_cast<const SIMDHolder *>(&*std::cend(con)));
}


template<typename T, typename Container>
inline void inc_counts(T &counts, const Container &con) {
    //static_assert(std::is_same<std::decay_t<decltype(con[0])>, uint8_t>::value, "Container must contain 8-bit unsigned integers.");
    return inc_counts(counts, reinterpret_cast<const SIMDHolder *>(&*std::cbegin(con)), reinterpret_cast<const SIMDHolder *>(&*std::cend(con)));
}


#else
template<typename Container>
inline std::array<uint32_t, 64> sum_counts(const Container &con) {
    std::array<uint32_t, 64> ret{0};
    for(const uint8_t x: ret) {
        ++ret[x];
    }
    return ret;
}
#endif

template<typename CoreType>
void parsum_helper(void *data_, long index, int) {
    parsum_data_t<CoreType> &data(*reinterpret_cast<parsum_data_t<CoreType> *>(data_));
    uint64_t local_counts[64]{0};
#ifndef VEC_DISABLED__
    SIMDHolder tmp;
    const SIMDHolder *p(reinterpret_cast<const SIMDHolder *>(&data.core_[index * data.pb_])),
                     *pend(reinterpret_cast<const SIMDHolder *>(&data.core_[std::min(data.l_, (index+1) * data.pb_)]));
    do {
        tmp = *p++;
        tmp.inc_counts(local_counts);
    } while(p < pend);
#else
    for(const uint8_t token: data.core_) {
        ++local_counts[token];
    }
#endif
    for(uint64_t i = 0; i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

inline std::set<uint64_t> seeds_from_seed(uint64_t seed, size_t size) {
    std::mt19937_64 mt(seed);
    std::set<uint64_t> rset;
    while(rset.size() < size) rset.emplace(mt());
    return rset;
}
template<typename T>
static double ertl_ml_estimate(const T& c, unsigned p, unsigned q, double relerr) {
/*
    Note --
    Putting all these optimizations together finally gives the new cardinality estimation
    algorithm presented as Algorithm 8. The algorithm requires mainly only elementary
    operations. For very large cardinalities it makes sense to use the strong (46) instead
    of the weak lower bound (47) as second starting point for the secant method. The
    stronger bound is a much better approximation especially for large cardinalities, where
    the extra logarithm evaluation is amortized by savings in the number of iteration cycles.
   -Ertl paper.
TODO:  Consider adding this change to the method. This could improve our performance for other
*/
    const uint64_t m = 1ull << p;
    if (c[q+1] == m) return std::numeric_limits<double>::infinity();

    int kMin, kMax;
    for(kMin=0; c[kMin]==0; ++kMin);
    int kMinPrime = std::max(1, kMin);
    for(kMax=q+1; kMax && c[kMax]==0; --kMax);
    int kMaxPrime = std::min(static_cast<int>(q), kMax);
    double z = 0.;
    for(int k = kMaxPrime; k >= kMinPrime; z = 0.5*z + c[k--]);
    z = ldexp(z, -kMinPrime);
    unsigned cPrime = c[q+1];
    if(q) cPrime += c[kMaxPrime];
    double gprev;
    double x;
    double a = z + c[0];
    int mPrime = m - c[0];
    gprev = z + ldexp(c[q+1], -q); // Reuse gprev, setting to 0 after.
    x = gprev <= 1.5*a ? mPrime/(0.5*gprev+a): (mPrime/gprev)*std::log1p(gprev/a);
    gprev = 0;
    double deltaX = x;
    relerr /= std::sqrt(m);
    while(deltaX > x*relerr) {
        int kappaMinus1;
        frexp(x, &kappaMinus1);
        double xPrime = ldexp(x, -std::max(static_cast<int>(kMaxPrime+1), kappaMinus1+2));
        double xPrime2 = xPrime*xPrime;
        double h = xPrime - xPrime2/3 + (xPrime2*xPrime2)*(1./45. - xPrime2/472.5);
        for(int k = kappaMinus1; k >= kMaxPrime; --k) {
            double hPrime = 1. - h;
            h = (xPrime + h*hPrime)/(xPrime+hPrime);
            xPrime += xPrime;
        }
        double g = cPrime*h;
        for(int k = kMaxPrime-1; k >= kMinPrime; --k) {
            double hPrime = 1. - h;
            h = (xPrime + h*hPrime)/(xPrime+hPrime);
            xPrime += xPrime;
            g += c[k] * h;
        }
        g += x*a;
        if(gprev < g && g <= mPrime) deltaX *= (g-mPrime)/(gprev-g);
        else                         deltaX  = 0;
        x += deltaX;
        gprev = g;
    }
    return x*m;
}

template<typename HllType>
double ertl_ml_estimate(const HllType& c, double relerr=1e-2) {
    return ertl_ml_estimate(detail::sum_counts(c.core()), c.p(), c.q(), relerr);
}

} // namespace detail

template<typename HllType>
std::array<double, 3> ertl_joint(const HllType &h1, const HllType &h2) {
    assert(h1.m() == h2.m() || !std::fprintf(stderr, "sizes don't match! Size1: %zu. Size2: %zu\n", h1.size(), h2.size()));
    std::array<double, 3> ret;
    if(h1.get_jestim() != ERTL_JOINT_MLE) {
        ret[2] = h1.union_size(h2);
        ret[0] = h1.creport();
        ret[1] = h2.creport();
        ret[2] = ret[0] + ret[1] - ret[2];
        ret[0] -= ret[2];
        ret[1] -= ret[2];
        ret[2] = std::max(ret[2], 0.);
        return ret;
    }
    using detail::ertl_ml_estimate;
    auto p = h1.p();
    auto q = h1.q();
    std::array<uint32_t, 64> c1{0}, c2{0}, cu{0}, ceq{0}, cg1{0}, cg2{0};
    detail::joint_unroller ju;
    ju.sum_arrays(h1.core(), h2.core(), c1, c2, cu, cg1, cg2, ceq);
    const double cAX = h1.get_is_ready() ? h1.creport() : ertl_ml_estimate(c1, h1.p(), h1.q());
    const double cBX = h2.get_is_ready() ? h2.creport() : ertl_ml_estimate(c2, h2.p(), h2.q());
    const double cABX = ertl_ml_estimate(cu, h1.p(), h1.q());
    std::fprintf(stderr, "Made initials: %lf, %lf, %lf\n", cAX, cBX, cABX);
    std::array<uint32_t, 64> countsAXBhalf;
    std::array<uint32_t, 64> countsBXAhalf;
    countsAXBhalf[q] = h1.m();
    countsBXAhalf[q] = h1.m();
    for(unsigned _q = 0; _q < q; ++_q) {
        // Handle AXBhalf
        countsAXBhalf[_q] = cg1[_q] + ceq[_q] + cg2[_q + 1];
        assert(countsAXBhalf[q] >= countsAXBhalf[_q]);
        countsAXBhalf[q] -= countsAXBhalf[_q];

        // Handle BXAhalf
        countsBXAhalf[_q] = cg2[_q] + ceq[_q] + cg1[_q + 1];
        assert(countsBXAhalf[q] >= countsBXAhalf[_q]);
        countsBXAhalf[q] -= countsBXAhalf[_q];
    }
    double cAXBhalf = ertl_ml_estimate(countsAXBhalf, p, q - 1);
    double cBXAhalf = ertl_ml_estimate(countsBXAhalf, p, q - 1);
    //std::fprintf(stderr, "Made halves: %lf, %lf\n", cAXBhalf, cBXAhalf);
    ret[0] = cABX - cBX;
    ret[1] = cABX - cAX;
    double cX1 = (1.5 * cBX + 1.5*cAX - cBXAhalf - cAXBhalf);
    double cX2 = 2.*(cBXAhalf + cAXBhalf) - 3.*cABX;
    ret[2] = std::max(0., 0.5 * (cX1 + cX2));
    return ret;
}

template<typename HllType>
std::array<double, 3> ertl_joint(HllType &h1, HllType &h2) {
    if(h1.get_jestim() != ERTL_JOINT_MLE) h1.csum(), h2.csum();
    return ertl_joint(static_cast<const HllType &>(h1), static_cast<const HllType &>(h2));
}



static constexpr double make_alpha(size_t m) {
    switch(m) {
        case 16: return .673;
        case 32: return .697;
        case 64: return .709;
        default: return 0.7213 / (1 + 1.079/m);
    }
}



template<typename HashStruct=WangHash>
class hllbase_t {
// HyperLogLog implementation.
// To make it general, the actual point of entry is a 64-bit integer hash function.
// Therefore, you have to perform a hash function to convert various types into a suitable query.
// We could also cut our memory requirements by switching to only using 6 bits per element,
// (up to 64 leading zeros), though the gains would be relatively small
// given how memory-efficient this structure is.

// Attributes
protected:
    std::vector<uint8_t, common::Allocator<uint8_t>> core_;
    mutable double                          value_;
    uint32_t                                   np_;
    EstimationMethod                        estim_;
    JointEstimationMethod                  jestim_;
    HashStruct                                 hf_;
public:
    using final_type = hllbase_t<HashStruct>;
    using HashType = HashStruct;
#if LZ_COUNTER
    std::array<std::atomic<uint64_t>, 64> clz_counts_; // To check for bias in insertion
#endif

    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this),
                              core_.size() * sizeof(core_[0]));
    }
    void reset() {
        std::fill(core_.begin(), core_.end(), uint64_t(0));
        value_ = -1.;
    }
    uint64_t hash(uint64_t val) const {return hf_(val);}
    uint64_t m() const {return static_cast<uint64_t>(1) << np_;}
    double alpha()          const {return make_alpha(m());}
    double relative_error() const {return 1.03896 / std::sqrt(static_cast<double>(m()));}
    bool operator==(const hllbase_t &o) const {
        return np_ == o.np_ &&
               std::equal(core_.begin(), core_.end(), o.core_.begin());
    }
    bool operator!=(const hllbase_t &o) const {
        return !this->operator==(o);
    }
    // Constructor
    template<typename... Args>
    explicit hllbase_t(size_t np, EstimationMethod estim,
                       JointEstimationMethod jestim,
                       Args &&... args):
        value_(-1.), np_(np),
        estim_(estim), jestim_(jestim)
        , hf_(std::forward<Args>(args)...)
    {
#if LZ_COUNTER
        for(size_t i = 0; i < clz_counts_.size(); ++i)
            clz_counts_[i].store(uint64_t(0));
#endif
        VERBOSE_ONLY(std::fprintf(stderr, "np = %zu, estim = %d, jest = %d\n", np, estim, jestim);)
        core_.resize(static_cast<uint64_t>(1) << np);
    }
    explicit hllbase_t(size_t np, HashStruct &&hs): hllbase_t(np, ERTL_MLE, (JointEstimationMethod)ERTL_MLE, std::move(hs)) {}
    explicit hllbase_t(size_t np, EstimationMethod estim=ERTL_MLE): hllbase_t(np, estim, (JointEstimationMethod)ERTL_MLE) {}
    explicit hllbase_t(): hllbase_t(size_t(0), EstimationMethod::ERTL_MLE, (JointEstimationMethod)ERTL_MLE) {}
    template<typename... Args>
    hllbase_t(const std::string &path, Args &&... args): hf_(std::forward<Args>(args)...) {read(path);}
    template<typename... Args>
    hllbase_t(gzFile fp, Args &&... args): hllbase_t(size_t(0), ERTL_MLE, (JointEstimationMethod)ERTL_MLE, std::forward<Args>(args)...) {this->read(fp);}

    // Call sum to recalculate if you have changed contents.
    void sum() const noexcept {
        const auto counts(detail::sum_counts(core_)); // std::array<uint32_t, 64>
        value_ = detail::calculate_estimate(counts, estim_, m(), np_, alpha());
    }
    void csum() const noexcept {if(!is_calculated()) sum();}

    // Returns cardinality estimate. Sums if not calculated yet.
    double creport() const noexcept {
        csum();
        return value_;
    }
    auto cfinalize() const noexcept {return finalize();}
    auto finalize() const & noexcept {
        sum();
        return *this;
    }
    auto finalize() & noexcept {
        return static_cast<const hllbase_t &>(*this).finalize();
    }
    auto finalize() && noexcept {
        return static_cast<const hllbase_t &&>(*this).finalize();
    }
    auto finalize() const && noexcept {
        auto ret(std::move(*this));
        ret.sum();
        const_cast<hllbase_t &>(*this).free();
        return ret;
    }
    double report() const noexcept {
        return creport();
    }
    double cardinality_estimate() const noexcept { return creport();}

    // Returns error estimate
    double cest_err() const {
        if(!is_calculated()) throw std::runtime_error("Result must be calculated in order to report.");
        return relative_error() * creport();
    }
    double est_err() const noexcept {
        return cest_err();
    }
    // Returns string representation
    std::string to_string() const {
        std::string params(std::string("p:") + std::to_string(np_) + '|' + EST_STRS[estim_] + ";");
        return (params + (is_calculated() ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                                         : desc_string()));
    }
    // Descriptive string.
    std::string desc_string() const {
        return std::string("Size: ") + std::to_string(np_) + ". nb: " + std::to_string(m()) + " error: " + std::to_string(relative_error()) + " , + method: " + EST_STRS[estim_];
    }

    INLINE void add(uint64_t hashval) noexcept {
        const uint32_t index(q() == 64 ? uint32_t(0): uint32_t(hashval >> q()));
        const uint8_t lzt = clz(((hashval << 1)|1) << (np_ - 1)) + 1;
#ifndef NOT_THREADSAFE
        for(;core_[index] < lzt;
             __sync_bool_compare_and_swap(&core_[index], core_[index], lzt));
#else
        if(core_[index] < lzt) core_[index] = lzt;
#endif

#if LZ_COUNTER
        ++clz_counts_[clz(((hashval << 1)|1) << (np_ - 1)) + 1];
#endif
    }

    INLINE void addh(uint64_t element) noexcept {
        element = hf_(element);
        add(element);
    }
    INLINE void addh(const std::string &element) noexcept {
#ifdef ENABLE_CLHASH
        CONST_IF(std::is_same<HashStruct, clhasher>::value) {
            add(hf_(element));
        } else {
#endif
            add(std::hash<std::string>{}(element));
#ifdef ENABLE_CLHASH
        }
#endif
    }
#ifndef VEC_DISABLED__
    INLINE void addh(VType element) noexcept {
        element = hf_(element.simd_);
        add(element);
    }
    INLINE void add(VType element) noexcept {
        element.for_each([&](uint64_t &val) {add(val);});
    }
#endif
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) noexcept {
        static_assert(std::is_same<std::decay_t<decltype(hasher(element))>, uint64_t>::value, "Must return 64-bit hash");
        add(hasher(element));
    }
#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) noexcept {
        static_assert(std::is_same<std::decay_t<decltype(hasher(s, len))>, uint64_t>::value, "Must return 64-bit hash");
        add(hasher(s, len));
    }
#endif
    void parsum(int nthreads=-1, size_t pb=4096) {
        if(nthreads < 0) nthreads = nthreads > 0 ? nthreads: std::thread::hardware_concurrency();
        std::atomic<uint64_t> acounts[64];
        std::fill(std::begin(acounts), std::end(acounts), 0);
        detail::parsum_data_t<decltype(core_)> data{acounts, core_, m(), pb};
        const uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
        kt_for(nthreads, detail::parsum_helper<decltype(core_)>, &data, nr);
        uint64_t counts[64];
        std::memcpy(counts, acounts, sizeof(counts));
        value_ = detail::calculate_estimate(counts, estim_, m(), np_, alpha());
    }
    ssize_t printf(std::FILE *fp) const noexcept {
        ssize_t ret = std::fputc('[', fp) > 0;
        for(size_t i = 0; i < core_.size() - 1; ++i)
            ret += std::fprintf(fp, "%d, ", int(core_[i]));
        return ret += std::fprintf(fp, "%d]", int(core_.back()));
    }
    std::string sprintf() const noexcept {
        std::fprintf(stderr, "Core size: %zu. to string: %s\n", core_.size(), to_string().data());
        std::string ret = "[";
        for(size_t i = 0; i < core_.size() - 1; ++i)
            ret += std::to_string(core_[i]), ret += ", ";
        ret += std::to_string(core_.back());
        ret += ']';
        return ret;
    }
    hllbase_t<HashStruct> compress(size_t new_np) const {
        // See Algorithm 3 in https://arxiv.org/abs/1702.01284
        // This is not very optimized.
        // I might later add support for doubling, c/o https://research.neustar.biz/2013/04/30/doubling-the-size-of-an-hll-dynamically-extra-bits/
        if(new_np == np_) return hllbase_t(*this);
        if(new_np > np_)
            throw std::runtime_error(std::string("Can't compress to a larger size. Current: ") + std::to_string(np_) + ". Requested new size: " + std::to_string(new_np));
        hllbase_t<HashStruct> ret(new_np, get_estim(), get_jestim());
        unsigned diff = np_ - new_np;
        size_t ratio = static_cast<size_t>(1) << (diff);
        size_t new_size = 1ull << new_np;
        size_t b = 0;
        for(size_t i(0); i < new_size; ++i) {
            size_t j(0);
            while(j < ratio && core_[j + b] == 0) ++j;
            if(j != ratio)
                ret.core_[i] = std::min(ret.q() + 1, j ? clz(j)+1: core_[b] + diff);
            // Otherwise left at 0
            b += ratio;
        }
        return ret;
    }
    // Reset.
    void clear() noexcept {
        std::memset(core_.data(), 0, core_.size() * sizeof(core_[0]));
        value_ = -1.;
    }
    hllbase_t(hllbase_t&&o): value_(-1.), np_(0), estim_(ERTL_MLE), jestim_(static_cast<JointEstimationMethod>(ERTL_MLE)), hf_(std::move(o.hf_)) {
        std::swap_ranges(reinterpret_cast<uint8_t *>(this),
                         reinterpret_cast<uint8_t *>(this) + sizeof(*this),
                         reinterpret_cast<uint8_t *>(std::addressof(o)));
    }
    hllbase_t(const hllbase_t &other): core_(other.core_), value_(other.value_), np_(other.np_),
        estim_(other.estim_), jestim_(other.jestim_), hf_(other.hf_)
    {
#if LZ_COUNTER
        for(size_t i = 0; i < clz_counts_.size(); ++i)
            clz_counts_[i].store(other.clz_counts_[i].load());
#endif
    }
    hllbase_t& operator=(const hllbase_t &other) {
        // Explicitly define to make sure we don't do unnecessary reallocation.
        if(core_.size() != other.core_.size()) core_.resize(other.core_.size());
        std::memcpy(core_.data(), other.core_.data(), core_.size()); // TODO: consider SIMD copy
        np_ = other.np_;
        value_ = other.value_;
        estim_ = other.estim_;
        hf_ = other.hf_;
        return *this;
    }
    hllbase_t& operator=(hllbase_t&&) = default;
    hllbase_t clone() const {
        return hllbase_t(np_, estim_, jestim_);
    }

    hllbase_t &operator+=(const hllbase_t &other) noexcept {
        PREC_REQ(np_ == other.np_, "mismatched sketch sizes.");
#ifndef VEC_DISABLED__
        unsigned i;
#if HAS_AVX_512 || __AVX2__ || __SSE2__
        if(m() >= sizeof(Type)) {
#if HAS_AVX_512 && __AVX512BW__
            __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
            const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
            for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]); // mm512_max_epu8 is available on with AVX512BW :(
#elif __AVX2__
            __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
            const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
            for(i = 0; i < m() * sizeof(uint8_t) / sizeof(__m256i); ++i) {
                assert(reinterpret_cast<const char *>(&els[i]) < reinterpret_cast<const char *>(&core_[core_.size()]));
                els[i] = _mm256_max_epu8(els[i], oels[i]);
            }
#else // __SSE2__
            __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
            const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
            for(i = 0; i < m() >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
#endif /* #if (HAS_AVX_512 && __AVX512BW__) || __AVX2__ || true */

            if(m() < sizeof(Type)) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
        } else {
#endif /* #if HAS_AVX_512 || __AVX2__ || __SSE2__ */
            uint64_t *els(reinterpret_cast<uint64_t *>(core_.data()));
            const uint64_t *oels(reinterpret_cast<const uint64_t *>(other.core_.data()));
            while(els < reinterpret_cast<const uint64_t *>(core_.data() + core_.size()))
                *els = std::max(*els, *oels), ++els, ++oels;
#if HAS_AVX_512 || __AVX2__ || __SSE2__
        }
#endif
        std::transform(core_.begin(), core_.end(), other.core_.begin(), core_.begin(), [](auto x, auto y) {return std::max(x, y);});
#else /*ifndef VEC_DISABLED__ */
#endif
        not_ready();
        return *this;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        if(new_size & (new_size - 1)) new_size = roundup(new_size);
        clear();
        core_.resize(new_size);
        np_ = ilog2(new_size);
    }
    EstimationMethod get_estim()       const {return  estim_;}
    JointEstimationMethod get_jestim() const {return jestim_;}
    void set_estim(EstimationMethod val) noexcept {
        estim_ = std::min(val, ERTL_MLE);
    }
    void set_jestim(JointEstimationMethod val) noexcept {
        jestim_ = val;
    }
    void set_jestim(uint16_t val) {set_jestim(static_cast<JointEstimationMethod>(val));}
    void set_estim(uint16_t val)  {estim_  = static_cast<EstimationMethod>(val);}
    bool get_is_ready() const {return value_ >= 0.;}
    void not_ready() {value_ = -1.;}
    void set_is_ready() {}
    bool may_contain(uint64_t hashval) const {
        // This returns false positives, but never a false negative.
        return core_[hashval >> q()] >= clz(hashval << np_) + 1;
    }

    bool within_bounds(uint64_t actual_size) const noexcept {
        return std::abs(actual_size - creport()) < relative_error() * actual_size;
    }
#if 0
    bool within_bounds(uint64_t actual_size) const noexcept {
        return std::abs(actual_size - report()) < est_err();
    }
#endif
    const auto &core()    const {return core_;}
    auto &mutable_core()        {return core_;}
    const uint8_t *data() const {return core_.data();}

    uint32_t p() const {return np_;}
    uint32_t q() const {return (sizeof(uint64_t) * CHAR_BIT) - np_;}
    void free() noexcept {
        decltype(core_) tmp{};
        std::swap(core_, tmp);
    }
    void write(FILE *fp) const {write(fileno(fp));}
    bool is_calculated() const {return value_ >= 0.;}
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t bf[]{is_calculated(), estim_, jestim_, 1};
        CW(fp, bf, sizeof(bf));
        CW(fp, &np_, sizeof(np_));
        CW(fp, &value_, sizeof(value_));
        CW(fp, core_.data(), core_.size() * sizeof(core_[0]));
#undef CW
    }
    void write(const char *path, bool write_gz=true) const {
        if(write_gz) {
            gzFile fp(gzopen(path, "wb"));
            if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at '") + path + "' for writing");
            write(fp);
            gzclose(fp);
        } else {
            std::FILE *fp(std::fopen(path, "wb"));
            if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for writing");
            write(fileno(fp));
            std::fclose(fp);
        }
    }
    void write(const std::string &path, bool write_gz=true) const {write(path.data(), write_gz);}
    void read(gzFile fp) {
#define CR(fp, dst, len) \
    do {\
        if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) {\
            throw ZlibError(std::string("[E:") + __FILE__ + ':' + std::to_string(__LINE__) + ':' + __PRETTY_FUNCTION__ + "] Error reading from file");\
        }\
    } while(0)
        uint32_t bf[4];
        CR(fp, bf, sizeof(bf));
        estim_  = static_cast<EstimationMethod>(bf[1]);
        jestim_ = static_cast<JointEstimationMethod>(bf[2]);
        CR(fp, &np_, sizeof(np_));
        CR(fp, &value_, sizeof(value_));
        core_.resize(m());
        CR(fp, core_.data(), (core_.size() * sizeof(core_[0])));
        csum();
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}
    void write(int fileno) const {
        uint32_t bf[]{is_calculated(), estim_, jestim_, 137};
#define CHWR(fn, obj, sz) \
    do {\
    if(HEDLEY_UNLIKELY(::write(fn, (obj), (sz)) != ssize_t(sz))) \
        throw std::runtime_error( \
            std::string("[") + __PRETTY_FUNCTION__ + std::string("Failed to write to disk at fd ") + std::to_string(fileno)); \
    } while(0)

        CHWR(fileno, bf, sizeof(bf));
        CHWR(fileno, &np_, sizeof(np_));
        CHWR(fileno, &value_, sizeof(value_));
        CHWR(fileno, core_.data(), core_.size());
#undef CHWR
    }
    void read(int fileno) {
        uint32_t bf[4];
#define CHRE(fn, obj, sz) if(HEDLEY_UNLIKELY(::read(fn, (obj), (sz)) != ssize_t(sz))) throw std::runtime_error(std::string("Failed to read from fd in ") + __PRETTY_FUNCTION__)
        CHRE(fileno, bf, sizeof(bf));
        estim_         = static_cast<EstimationMethod>(bf[1]);
        jestim_        = static_cast<JointEstimationMethod>(bf[2]);
#if VERBOSE_AF
        if(bf[3] != 137) {
            std::fprintf(stderr, "Warning: old sketches. Still binary compatible, but FYI.\n");
        }
#endif
        CHRE(fileno, &np_, sizeof(np_));
        CHRE(fileno, &value_, sizeof(value_));
        core_.resize(m());
        CHRE(fileno, core_.data(), core_.size());
#undef CHRE
    }
    hllbase_t operator+(const hllbase_t &other) const {
        hllbase_t ret(*this);
        ret += other;
        return ret;
    }
    double union_size(const hllbase_t &other) const noexcept {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) {
            assert(m() == other.m());
            std::array<uint32_t, 64> counts{0};
            // We can do this because we use an aligned allocator.
            // We also have found that wider vectors than SSE2 don't matter
#if __SSE2__
            const __m128i *p1(reinterpret_cast<const __m128i *>(data())), *p2(reinterpret_cast<const __m128i *>(other.data()));
            const __m128i *const pe(reinterpret_cast<const __m128i *>(&core_[core_.size()]));
            for(__m128i tmp;p1 < pe;) {
                tmp = _mm_max_epu8(*p1++, *p2++);
                for(size_t i = 0; i < sizeof(tmp);++counts[reinterpret_cast<uint8_t *>(&tmp)[i++]]);
            }
#else
            for(size_t i = 0; i < core_.size(); ++i) {
                ++counts[std::max(core_[i], other.core_[i])];
            }
#endif
            return detail::calculate_estimate(counts, get_estim(), m(), p(), alpha());
        }
        const auto full_counts = ertl_joint(*this, other);
        return full_counts[0] + full_counts[1] + full_counts[2];
    }
    // Jaccard index, but returning a bool to indicate whether it was less than expected error for the cardinality/sketch size
    std::pair<double, bool> bjaccard_index(hllbase_t &h2) const noexcept {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) csum(), h2.csum();
        return const_cast<hllbase_t &>(*this).bjaccard_index(const_cast<const hllbase_t &>(h2));
    }
    std::pair<double, bool> bjaccard_index(const hllbase_t &h2) const noexcept {
        if(jestim_ == JointEstimationMethod::ERTL_JOINT_MLE) {
            auto full_cmps = ertl_joint(*this, h2);
            auto ret = full_cmps[2] / (full_cmps[0] + full_cmps[1] + full_cmps[2]);
            return std::make_pair(ret, ret > relative_error());
        }
        const double us = union_size(h2);
        const double ret = std::max(0., creport() + h2.creport() - us) / us;
        return std::make_pair(ret, ret > relative_error());
    }
    double jaccard_index(hllbase_t &h2) const noexcept {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) csum(), h2.csum();
        return const_cast<hllbase_t &>(*this).jaccard_index(const_cast<const hllbase_t &>(h2));
    }
    double containment_index(const hllbase_t &h2) const noexcept {
        auto fsr = full_set_comparison(h2);
        return fsr[2] / (fsr[2] + fsr[0]);
    }
    std::array<double, 3> full_set_comparison(const hllbase_t &h2) const noexcept {
        if(jestim_ == JointEstimationMethod::ERTL_JOINT_MLE) {
            return ertl_joint(*this, h2);
        }
        const double us = union_size(h2), mys = creport(), os = h2.creport(),
                     is = std::max(mys + os - us, 0.),
                     my_only = std::max(mys - is, 0.), o_only = std::max(os - is, 0.);
        return std::array<double, 3>{{my_only, o_only, is}};
    }
    double jaccard_index(const hllbase_t &h2) const noexcept {
        if(jestim_ == JointEstimationMethod::ERTL_JOINT_MLE) {
            auto full_cmps = ertl_joint(*this, h2);
            const auto ret = full_cmps[2] / (full_cmps[0] + full_cmps[1] + full_cmps[2]);
            return ret;
        }
        const double us = union_size(h2);
        const double ret = (creport() + h2.creport() - us) / us;
        return std::max(0., ret);
    }
    size_t size() const {return size_t(m());}
    static constexpr unsigned min_size() {
#ifndef VEC_DISABLED__
        return ilog2(sizeof(detail::SIMDHolder));
#else
        return 1;
#endif
    }
#if LZ_COUNTER
    ~hllbase_t() {
        std::string tmp;
        for(const auto &val: clz_counts_) tmp += std::to_string(val), tmp += ',';
        tmp.pop_back();
        std::fprintf(stderr, "counts: %s\n", tmp.data());
    }
#endif
};
using hll_t = hllbase_t<>;

template<typename T>
struct has_csum: public std::false_type {};
template<typename HS>
struct has_csum<hllbase_t<HS>>: public std::true_type {};
#if __cplusplus >= 201703L
template<typename T>
static constexpr bool has_csum_v = has_csum<T>::value;
#endif

#ifndef NOT_THREADSAFE
static bool warning_emitted = false;
#endif

template<typename HashStruct=WangHash>
class shllbase_t: public hllbase_t<HashStruct> {
    // See Edith Cohen - All-Distances Sketches, Revisited: HIP Estimators for Massive Graphs Analysis
    // and
    // Daniel Ting - Streamed Approximate Counting of Distinct Elements
    using super = hllbase_t<HashStruct>;
    double cest_;
    double s_;
    // Streaming HyperLogLog
    // Note: composition is not supported, at least not in a principled way.
    // Better estimates on original cardinalities should help the union, but it will
    // not enjoy the asymptotic improvements necessarily.
public:
    template<typename...Args>
    shllbase_t(Args &&...args): super(std::forward<Args>(args)...), cest_(0), s_(this->core_.size()) {}
    INLINE void addh(uint64_t element) {
        element = this->hf_(element);
        add(element);
    }
    auto full_set_comparison(const shllbase_t &o) const {
        auto union_est = super::union_size(o);
        auto isz = std::max(this->cest_ + o.cest_ - union_est, 0.);
        return std::array<double, 3>{this->cest_ - isz, o.cest_ - isz, std::max(this->cest_ + o.cest_ - union_est, 0.)};
    }
    auto jaccard_index(const shllbase_t &o) const {
        auto union_est = super::union_size(o);
        auto isz = std::max(this->cest_ + o.cest_ - union_est, 0.);
        return isz / union_est;
    }
    auto containment_index(const shllbase_t &o) const {
        auto union_est = super::union_size(o);
        auto isz = std::max(this->cest_ + o.cest_ - union_est, 0.);
        return isz / this->cest_;
    }
    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        if(!warning_emitted) {
            warning_emitted = true;
            std::fprintf(stderr, "Warning: shllbase_t is not threadsafe\n");
        }
#endif
        const uint32_t index(hashval >> this->q());
        const uint8_t lzt(clz(((hashval << 1)|1) << (this->np_ - 1)) + 1);
        auto oldv = this->core_[index];
        if(lzt > oldv) {
            cest_ += 1. / std::ldexp(s_, -int(this->np_));
            s_ -= std::ldexp(1., -int(oldv)); // replace with lut?
            if(lzt != 64 - this->np_)
                s_ += std::ldexp(1., -int(lzt));
            this->core_[index] = lzt;
            //std::fprintf(stderr, "news: %f. newc: %f\n", s_, cest_);
        } else {
            //std::fprintf(stderr, "newv %u is not more than %u\n", lzt, oldv);
        }
    }
    auto super_report() const {return super::report();}
    auto report() {return cest_;}
    auto report() const {return cest_;}
    auto creport() {return cest_;}
    auto creport() const {return cest_;}
    shllbase_t &merge_UNSTABLE(const shllbase_t &o) {
        PREC_REQ(this->size() == o.size(), "must be same size");
        SK_UNROLL_8
        for(size_t i = 0; i < this->size(); ++i) {
            auto oldv = this->core_[i], ov = o.core_[i];
            if(ov > oldv) {
                cest_ += 1./ std::ldexp(s_, -int(this->np_));
                s_ -= std::ldexp(1., -int(oldv));
                if(ov != 64 - this->np_)
                    s_ += std::ldexp(1., -int(ov));
                this->core_[i] = ov;
            }
        }
    }
#if 0
    double uest_UNSTABLE(const shllbase_t &o) const {
        PREC_REQ(this->size() == o.size(), "must be same size");
        double cest = cest_, s = s_;;
        SK_UNROLL_8
        for(size_t i = 0; i < this->size(); ++i) {
            auto oldv = this->core_[i], ov = o.core_[i];
            if(ov > oldv) {
                cest += 1./ std::ldexp(s_, -int(this->np_));
                s -= std::ldexp(1., -int(oldv));
                if(ov != 64 - this->np_)
                    s += std::ldexp(1., -int(ov));
            }
        }
        return cest;
    }
#endif
    using final_type = shllbase_t;
};

using shll_t = shllbase_t<>;

// Returns the size of the set intersection
template<typename HS>
inline double intersection_size(hllbase_t<HS> &first, hllbase_t<HS> &other) noexcept {
    first.csum(), other.csum();
    return intersection_size(static_cast<const hllbase_t<HS> &>(first), static_cast<const hllbase_t<HS> &>(other));
}

template<typename HllType> inline std::pair<double, bool> bjaccard_index(const HllType &h1, const HllType &h2) {return h1.bjaccard_index(h2);}
template<typename HllType> inline std::pair<double, bool> bjaccard_index(HllType &h1, HllType &h2) {return h1.bjaccard_index(h2);}

// Returns a HyperLogLog union
template<typename HS>
static inline double union_size(const hllbase_t<HS> &h1, const hllbase_t<HS> &h2) {return h1.union_size(h2);}

template<typename HS>
static inline double intersection_size(const hllbase_t<HS> &h1, const hllbase_t<HS> &h2) {
    return std::max(0., h1.creport() + h2.creport() - union_size(h1, h2));
}


template<typename SeedHllType=hll_t>
class hlfbase_t {
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<SeedHllType>                      hlls_;
    std::vector<uint64_t, common::Allocator<uint64_t>>   seeds_;
    std::vector<double>                         values_;
    mutable double                               value_;
    bool                                 is_calculated_;
    using HashType     = typename SeedHllType::HashType;
public:
    template<typename... Args>
    hlfbase_t(size_t size, uint64_t seedseed, Args &&... args): value_(0), is_calculated_(0) {
        auto sfs = detail::seeds_from_seed(seedseed, size);
        assert(sfs.size());
        hlls_.reserve(size);
        for(const auto seed: sfs) seeds_.emplace_back(seed);
        while(hlls_.size() < seeds_.size()) hlls_.emplace_back(std::forward<Args>(args)...);
    }
    hlfbase_t(const hlfbase_t &) = default;
    hlfbase_t(hlfbase_t &&) = default;
    uint64_t size() const {return hlls_.size();}
    auto m() const {return hlls_[0].size();}
    void write(const char *fn) const {
        gzFile fp = gzopen(fn, "wb");
        if(fp == nullptr) throw ZlibError(Z_ERRNO, std::string("Could not open file for reading at ") + fn);
        this->write(fp);
        gzclose(fp);
    }
    void clear() {
        value_ = is_calculated_ = 0;
        for(auto &hll: hlls_) hll.clear();
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        if(fp == nullptr) throw ZlibError(Z_ERRNO, std::string("Could not open file for reading at ") + fn);
        this->read(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint64_t sz = hlls_.size();
        gzwrite(fp, &sz, sizeof(sz));
        for(auto &seed: seeds_) gzwrite(fp, &seed, sizeof(seed));
        for(const auto &hll: hlls_) {
            hll.write(fp);
        }
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t size;
        gzread(fp, &size, sizeof(size));
        seeds_.resize(size);
        for(unsigned i(0); i < size; gzread(fp, seeds_.data() + i++, sizeof(seeds_[0])));
        hlls_.clear();
        while(hlls_.size() < size) hlls_.emplace_back(fp);
        gzclose(fp);
    }

    // This only works for hlls using 64-bit integers.
    // Looking ahead,consider templating so that the double version might be helpful.

#ifndef VEC_DISABLED__
    using Space = vec::SIMDTypes<uint64_t>;
#endif
    bool may_contain(uint64_t element) const {
#ifndef VEC_DISABLED__
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw NotImplementedError("supporting a non-power of two.");
            const Type *sptr = reinterpret_cast<const Type *>(&seeds_[0]);
            const Type *eptr = reinterpret_cast<const Type *>(&seeds_[seeds_.size()]);
            VType key;
            do {
                key = WangHash()(*sptr++ ^ Space::set1(element));
                for(unsigned i(0); i < Space::COUNT;) if(!hlls_[k++].may_contain(key.arr_[i++])) return false;
            } while(sptr < eptr);
            return true;
        }
#endif
        // if size() >= Space::COUNT
        for(unsigned i(0); i < size(); ++i) if(!hlls_[i].may_contain(WangHash()(element ^ seeds_[i]))) return false;
        return true;
    }
    void addh(uint64_t val) {
        unsigned k = 0;
#ifndef VEC_DISABLED__
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw NotImplementedError("supporting a non-power of two.");
            const Type *sptr = reinterpret_cast<const Type *>(&seeds_[0]);
            const Type *eptr = reinterpret_cast<const Type *>(&seeds_[seeds_.size()]);
            const Type element = Space::set1(val);
            VType key;
            do {
                key = WangHash()(*sptr++ ^ element);
                for(unsigned i(0) ; i < Space::COUNT; hlls_[k++].add(key.arr_[i++]));
                assert(k <= size());
            } while(sptr < eptr);
        }
#endif
        while(k < size()) hlls_[k].add(WangHash()(val ^ seeds_[k])), ++k;
    }
    double creport() const {
        if(is_calculated_) return value_;
        double ret(hlls_[0].creport());
        for(size_t i(1); i < size(); ret += hlls_[i++].creport());
        ret /= static_cast<double>(size());
        return value_ = ret;
    }
    hlfbase_t &operator+=(const hlfbase_t &other) {
        if(other.size() != size()) throw std::runtime_error("Wrong number of subsketches.");
        if(other.hlls_[0].p() != hlls_[0].p()) throw std::runtime_error("Wrong size of subsketches.");
        for(unsigned i(0); i < size(); ++i) {
            hlls_[i] += other.hlls_[i];
            hlls_[i].not_ready();
        }
        is_calculated_ = false;
    }
    hlfbase_t operator+(const hlfbase_t &other) const {
        // Could be more directly optimized.
        hlfbase_t ret = *this;
        ret += other;
    }
    double jaccard_index(const hlfbase_t &other) const {
        double est = chunk_report();
        est += other.chunk_report();
        hlfbase_t tmp = *this + other;
        double uest = tmp.chunk_report();
        double olap = est - uest;
        olap /= uest;
        return olap;
    }
    double report() noexcept {
        if(is_calculated_) return value_;
        hlls_[0].csum();
#if DIVIDE_EVERY_TIME
        double ret(hlls_[0].report() / static_cast<double>(size()));
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report() / static_cast<double>(size());
        }
#else
        double ret(hlls_[0].report());
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report();
        }
        ret /= static_cast<double>(size());
#endif
        return value_ = ret;
    }

    double med_report() noexcept {
        double *values = static_cast<double *>(size() < 100000u ? __builtin_alloca(size() * sizeof(double)): std::malloc(size() * sizeof(double))), *p = values;
        for(auto it = hlls_.begin(); it != hlls_.end(); *p++ = it->report(), ++it);
        if(size() < 32) {
            sort::insertion_sort(values, values + size());
            return .5 * (values[size() >> 1] + values[(size() >> 1) - 1]);
        }
        std::nth_element(values, values + (size() >> 1) - 1, values + size());
        double ret = .5 * (values[(size() >> 1) - 1] + values[(size() >> 1)]);
        if(hlls_.size() >= 100000u) std::free(values);
        return ret;
    }
    // Attempt strength borrowing across hlls with different seeds
    double chunk_report() const {
        if(HEDLEY_LIKELY((size() & (size() - 1)) == 0)) {
            std::array<uint32_t, 64> counts{0};
            for(const auto &hll: hlls_) {
#ifndef VEC_DISABLED__
                detail::inc_counts(counts, hll.core());
#else
                for(const uint8_t val: hll.core()) {
                    ++counts[val];
                }
#endif
            }
            const auto diff = (sizeof(uint32_t) * CHAR_BIT - clz(uint32_t(size())) - 1);
            const auto new_p = hlls_[0].p() + diff;
            const auto new_m = (1ull << new_p);
            return detail::calculate_estimate(counts, hlls_[0].get_estim(), new_m,
                                              new_p, make_alpha(new_m)) / (1ull << diff);
        } else {
            std::fprintf(stderr, "chunk_report is currently only supported for powers of two.");
            return creport();
            // Could try weight averaging, but currently I just report default when size is not a power of two.
        }
    }
};
using hlf_t = hlfbase_t<>;

} // namespace hll

#ifndef VEC_DISABLED__
namespace whll {
using common::Allocator;
struct wh119_t {
    std::vector<uint8_t, Allocator<uint8_t>> core_;
    double wh_base_;
    double estimate_;
    wh119_t(std::vector<uint8_t, Allocator<uint8_t>> &s, long double base): core_(std::move(s)), wh_base_(base), estimate_(cardinality_estimate())
    {
        assert(!(core_.size() & (core_.size() - 1)));
    }
    wh119_t(const char *s) {
        read(s);
    }
    wh119_t(gzFile fp) {
        read(fp);
        estimate_ = cardinality_estimate();
    }
    template<typename HS>
    wh119_t(const minhash::WideHyperLogLogHasher<HS> &o): wh119_t(o.make_whll()) {
    }
    wh119_t(std::string s): wh119_t(s.data()) {}
    wh119_t(const std::vector<uint8_t, Allocator<uint8_t>> &s, long double base): core_(s), wh_base_(base), estimate_(cardinality_estimate()) {}
    std::array<double, 3> full_set_comparison(const wh119_t &o) const {
        double ji = jaccard_index(o);
        double is = (estimate_ + o.estimate_) * ji / (1. + ji);
        double me_only = estimate_ > is ? estimate_ - is: 0.,
               o_only  = o.estimate_ > is ? o.estimate_ - is: 0.;
        return std::array<double, 3>{{me_only, o_only, is}};
    }
    size_t size() const {return core_.size();}
    auto m() const {return size();}
    uint8_t p() const {return ilog2(size());}
    void write(gzFile fp) const {
        uint64_t sz = core_.size();
        gzwrite(fp, &sz, sizeof(sz));
        gzwrite(fp, &wh_base_, sizeof(wh_base_));
        gzwrite(fp, core_.data(), core_.size());
    }
    void write(std::string s) const {write(s.data());}
    void write(const char *s) const {
        gzFile fp = gzopen(s, "wb");
        if(fp == nullptr)
            throw ZlibError(Z_ERRNO, std::string("Could not open file for writing at ") + s);
        write(fp);
        gzclose(fp);
    }
    void read(const char *s) {
        gzFile fp = gzopen(s, "rb");
        if(fp == nullptr)
            throw ZlibError(Z_ERRNO, std::string("Could not open file for reading at ") + s);
        read(fp);
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t sz;
        gzread(fp, &sz, sizeof(sz));
        gzread(fp, &wh_base_, sizeof(wh_base_));
        core_.resize(sz);
        gzread(fp, core_.data(), core_.size());
    }
    wh119_t &operator+=(const wh119_t &o) {
        PREC_REQ(size() == o.size(), "mismatched sketch sizes.");
        unsigned i;
#if !defined(VEC_DISABLED__) && (HAS_AVX_512 || __AVX2__ || __SSE2__)
        if(m() >= sizeof(Type)) {
#if HAS_AVX_512 && __AVX512BW__
            __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
            const __m512i *oels(reinterpret_cast<const __m512i *>(o.core_.data()));
            for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]); // mm512_max_epu8 is available on with AVX512BW :(
#elif __AVX2__
            __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
            const __m256i *oels(reinterpret_cast<const __m256i *>(o.core_.data()));
            for(i = 0; i < m() * sizeof(uint8_t) / sizeof(__m256i); ++i) {
                assert(reinterpret_cast<const char *>(&els[i]) < reinterpret_cast<const char *>(&core_[core_.size()]));
                els[i] = _mm256_max_epu8(els[i], oels[i]);
            }
#else // __SSE2__
            __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
            const __m128i *oels(reinterpret_cast<const __m128i *>(o.core_.data()));
            for(i = 0; i < m() >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
#endif /* #if (HAS_AVX_512 && __AVX512BW__) || __AVX2__ || true */

            if(m() < sizeof(Type)) for(;i < m(); ++i) core_[i] = std::max(core_[i], o.core_[i]);
            return *this;
        }
#else /* #if HAS_AVX_512 || __AVX2__ || __SSE2__ */
        std::transform(core_.data(), core_.data() + core_.size(), o.core_.data(), core_.data(), [](auto x, auto y) {return std::max(x, y);});
#endif
        return *this;
    }
    wh119_t operator+(const wh119_t &o) const {
        auto ret = *this;
        ret += o;
        return ret;
    }
    double cardinality_estimate() const {
        std::array<uint32_t, 256> counts{0};
        PREC_REQ(is_pow2(core_.size()), "Size must be a power of two");
        PREC_REQ(core_.size() >= sizeof(hll::detail::SIMDHolder), "Must be at least as large as a SIMD vector");
        const hll::detail::SIMDHolder *ptr = static_cast<const hll::detail::SIMDHolder*>(static_cast<const void *>(core_.data()));
        for(size_t i = 0; i < core_.size() / sizeof(*ptr); ++i) {
            (*ptr++).inc_counts(counts);
        }
        long double sum = counts[0];
        for(ssize_t i = 1; i < ssize_t(counts.size()); ++i) {
            sum += static_cast<long double>(counts[i]) * (std::pow(wh_base_, -i));
        }
        double ret = static_cast<double>(std::pow(core_.size(), 2) / sum) / std::sqrt(wh_base_);
        // low range correction -- fall back to linear counting
        if(ret < 2.5 * core_.size() && counts[0]) {
            double m = core_.size();
            double newv = m * std::log(m / counts[0]);
#ifndef NDEBUG
            //std::fprintf(stderr, "Underfull sketch. Switching to linear counting (bloom filter estimate). Initial est: %g. Corrected: %g. number of zeros: %u. core size: %zu\n", ret, newv, counts[0], core_.size());
#endif
            ret = newv;
        }
        return ret;
    }
    double jaccard_index(const wh119_t &o) const {
        double us = union_size(o);
        return std::max((estimate_ + o.estimate_ - us) / us, 0.);
    }
    double containment_index(const wh119_t &o) const {
        double us = union_size(o);
        return std::max((estimate_ + o.estimate_ - us) / estimate_, 0.);
    }
    double union_size(const wh119_t &o) const {return union_size(o.core_);}
    double union_size(const std::vector<uint8_t, Allocator<uint8_t>> &o) const {
        PREC_REQ(o.size() == size(), "mismatched sizes");
        if(o.size() != size()) throw std::runtime_error("Non-matching parameters for wh119_t");
        std::array<uint32_t, 256> counts;
        std::memset(counts.data(), 0, sizeof(counts));
        size_t i;
        using space = vec::SIMDTypes<uint64_t>;
        const space::Type *p1 = static_cast<const space::Type *>(static_cast<const void *>(core_.data())),
                          *p2 = static_cast<const space::Type *>(static_cast<const void *>(o.data()));
        for(i = 0; i < core_.size() / sizeof(*p1); ++i) {
            hll::detail::SIMDHolder(hll::detail::SIMDHolder::max_fn(*p1++, *p2++)).inc_counts(counts);
        }
        long double tmp = counts[0];
        for(ssize_t i = 1; i < ssize_t(counts.size()); ++i)
            tmp += static_cast<long double>(counts[i]) * (std::pow(wh_base_, -i));
        double ret = (std::pow(core_.size(), 2) / tmp) / std::sqrt(wh_base_);
        if(ret < 2.5 * core_.size() && counts[0]) {
            double m = core_.size();
            ret = m * std::log(m / counts[0]);
        }
        return ret;
    }
    void free() {
        decltype(core_) tmp; std::swap(tmp, core_);
    }
};
} // whll
#endif /*ifndef VEC_DISABLED__ */
} // namespace sketch

#endif // #ifndef HLL_H_
