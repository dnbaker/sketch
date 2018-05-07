#ifndef HLL_H_
#define HLL_H_
#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "sseutil.h"
#include "util.h"
#include "math.h"
#include "unistd.h"
#include "x86intrin.h"
#include "kthread.h"
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif

#define NO_SLEEF
#define NO_BLAZE
#include "vec.h" // Import vec.h, but disable blaze and sleef.

#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)
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



namespace hll {
using namespace std::literals;

/*
 * TODO: calculate distance *directly* without copying to another sketch!
 */

using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::size_t;

enum EstimationMethod: uint8_t {
    ORIGINAL       = 0,
    ERTL_IMPROVED  = 1,
    ERTL_MLE       = 2
};

enum JointEstimationMethod: uint8_t {
    //ORIGINAL       = 0,
    //ERTL_IMPROVED  = 1, // Improved but biased method
    //ERTL_MLE       = 2, // element-wise max, followed by MLE
    ERTL_JOINT_MLE = 3  // Ertl special version
};

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

namespace detail {
template<typename T>
inline double ertl_ml_estimate(const T& c, unsigned p, unsigned q, double relerr=1e-2); // forward declaration
template<typename Container>
inline std::array<uint64_t, 64> sum_counts(const Container &con);
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
std::string counts2str(const std::array<uint64_t, 64> &arr) {
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
    std::array<uint64_t, 64> countsAXBhalf{0}, countsBXAhalf{0};
    countsAXBhalf[q] = countsBXAhalf[q] = h1.m();
    std::array<uint64_t, 64> cg1{0}, cg2{0}, ceq{0};
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
                    __builtin_unreachable();
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
#if !NDEBUG
    std::fprintf(stderr, "cAXBhalf = %lf\n", cAXBhalf);
    std::fprintf(stderr, "cBXAhalf = %lf\n", cBXAhalf);
#endif
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
static double small_range_correction_threshold(uint64_t m) {return 2.5 * m;}

static INLINE uint64_t finalize(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53;
    key ^= key >> 33;
    return key;
}


template<typename CountArrType>
inline double calculate_estimate(const CountArrType &counts,
                                 EstimationMethod estim, uint64_t m, uint32_t p, double alpha, double relerr=1e-2) {
    assert(estim <= 3 && estim >= 0);
    static_assert(std::is_same_v<std::decay_t<decltype(counts[0])>, uint64_t>, "Counts must be a container for uint64_ts.");
    switch(estim) {
        case ORIGINAL: {
            assert(estim != ERTL_MLE);
            double sum = counts[0];
            for(unsigned i = 1; i < 64 - p; ++i) sum += counts[i] * (1. / (1ull << i)); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
            double value(alpha * m * m / sum);
            if(value < detail::small_range_correction_threshold(m)) {
                if(counts[0]) {
#if !NDEBUG
                    std::fprintf(stderr, "[W:%s:%d] Small value correction. Original estimate %lf. New estimate %lf.\n",
                                 __PRETTY_FUNCTION__, __LINE__, value, m * std::log((double)m / counts[0]));
#endif
                    value = m * std::log((double)(m) / counts[0]);
                }
            } else if(value > detail::LARGE_RANGE_CORRECTION_THRESHOLD) {
                // Reuse sum variable to hold correction.
                // I do think I've seen worse accuracy with the large range correction, but I would need to rerun experiments to be sure.
                sum = -std::pow(2.0L, 32) * std::log1p(-std::ldexp(value, -32));
                if(!std::isnan(sum)) value = sum;
#if !NDEBUG
                else std::fprintf(stderr, "[W:%s:%d] Large range correction returned nan. Defaulting to regular calculation.\n", __PRETTY_FUNCTION__, __LINE__);
#endif
            }
            return value;
        }
        case ERTL_IMPROVED: {
            static const double divinv = 1. / (2.L*std::log(2.L));
            double z = m * detail::gen_tau(static_cast<double>((m-counts[64 - p + 1]))/(double)m);
            for(unsigned i = 64-p; i; z += counts[i--], z *= 0.5); // Reuse value variable to avoid an additional allocation.
            z += m * detail::gen_sigma(static_cast<double>(counts[0])/static_cast<double>(m));
            return m * divinv * m / z;
        }
        case ERTL_MLE: return ertl_ml_estimate(counts, p, 64 - p, relerr);
    }
    __builtin_unreachable();
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<uint64_t> *counts_; // Array decayed to pointer.
    const CoreType          &core_;
    const uint64_t              l_;
    const uint64_t             pb_; // Per-batch
};



union SIMDHolder {

public:

#define DEC_MAX(fn) static constexpr decltype(&fn) max_fn = &fn
#define DEC_GT(fn)  static constexpr decltype(&fn) gt_fn  = &fn
#define DEC_EQ(fn)  static constexpr decltype(&fn) eq_fn  = &fn

// vpcmpub has roughly the same latency as
// vpcmpOPub, but twice the throughput, so we use it instead.
// (the *epu8_mask version over the *epu8_mask)
#if HAS_AVX_512
    using SType    = __m512i;
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
#if !NDEBUG
        SIMDHolder ac(a), bc(b);
        for(unsigned i(0); i < sizeof(ret); ++i) {
            assert(ret.vals[i] == std::max(ac.vals[i], bc.vals[i]));
        }
#endif
        return ret;
    }
    static SIMDHolder eq_fn(__m512i a, __m512i b) {
        SIMDHolder ret;
        ret.subs[0] = _mm256_cmpeq_epi8(*(__m256i *)(&a), *(__m256i *)(&b));
        ret.subs[1] = _mm256_cmpeq_epi8(*(__m256i *)(((uint8_t *)&a) + 32), *(__m256i *)(((uint8_t *)&b) + 32));
#if !NDEBUG
        SIMDHolder tmp, ac(a), bc(b);
        ac.val = a; bc.val = b;
        for(unsigned i(0); i < sizeof(ret); ++i) {
            assert(!!ret.vals[i] == !!(ac.vals[i] == bc.vals[i]));
        }
#endif
        return ret;
    }
#  endif
#elif __AVX2__
    using SType = __m256i;
    using MaskType = SIMDHolder;
    DEC_MAX(_mm256_max_epu8);
    DEC_EQ (_mm256_cmpeq_epi8);
    DEC_GT (_mm256_cmpgt_epi8);
    //DEC_GT(_mm256_cmpgt_epu8_mask);
#elif __SSE2__
    using SType = __m128i;
    using MaskType = SIMDHolder;
    DEC_MAX(_mm_max_epu8);
    DEC_GT(_mm_cmpgt_epi8);
    DEC_EQ(_mm_cmpeq_epi8);
#else
#  error("Need at least SSE2")
#endif
#undef DEC_MAX
#undef DEC_GT
#undef DEC_EQ

    SIMDHolder() {} // empty constructor
    SIMDHolder(SType val_) {
        val = val_;
    }
    operator SType &() {
        return val;
    }
    operator const SType &() const {
        return val;
    }
    static constexpr size_t nels  = sizeof(SType) / sizeof(uint8_t);
    static constexpr size_t nbits = sizeof(SType) / sizeof(uint8_t) * CHAR_BIT;
    using u8arr = uint8_t[nels];
    SType val;
    u8arr vals;
    template<typename T>
    void inc_counts(T &arr) const {
        static_assert(std::is_same_v<std::decay_t<decltype(arr[0])>, uint64_t>, "Must container 64-bit integers.");
        unroller<T, 0, nels> ur;
        ur(*this, arr);
    }
    // Worth considering: it's possible that reinterpreting it as a set of 16-bit integers
    // and make a lookup table.
    template<typename T, size_t iternum, size_t niter_left> struct unroller {
        void operator()(const SIMDHolder &ref, T &arr) const {
            ++arr[ref.vals[iternum]];
            unroller<T, iternum+1, niter_left-1>()(ref, arr);
        }
    };
    template<typename T, size_t iternum> struct unroller<T, iternum, 0> {
        void operator()(const SIMDHolder &ref, T &arr) const {}
    };
    static_assert(sizeof(SType) == sizeof(u8arr), "both items in the union must have the same size");
};

struct joint_unroller {
    using MType = SIMDHolder::MaskType;
    using SType = SIMDHolder::SType;
    // Woof....
#if defined(__AVX512BW__)
    static_assert(sizeof(MType) == sizeof(uint64_t), "Must be 64 bits");
#endif
    template<typename T, size_t iternum, size_t niter_left> struct ju_impl {
        void operator()(const SIMDHolder &ref1, const SIMDHolder &ref2, const SIMDHolder &u, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq, MType gtmask1, MType gtmask2, MType eqmask) const {
            ++arrh1[ref1.vals[iternum]];
            ++arrh2[ref2.vals[iternum]];
            ++arru[u.vals[iternum]];
#if __AVX512BW__
            arrg1[ref1.vals[iternum]] += gtmask1 & 1;
            arreq[ref1.vals[iternum]] += eqmask  & 1;
            arrg2[ref2.vals[iternum]] += gtmask2 & 1;
            gtmask1 >>= 1;
            gtmask2 >>= 1;
            eqmask  >>= 1; // Consider packing these into an SIMD type and shifting them as a set.
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
        INLINE void operator()(const SIMDHolder &ref1, const SIMDHolder &ref2, const SIMDHolder &u, T &arrh1, T &arrh2, T &arru, T &arrg1, T &arrg2, T &arreq, MType gtmask1, MType gtmask2, MType eqmask) const {}
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
        static_assert(std::is_same_v<MType, std::decay_t<decltype(g1)>>, "g1 should be the same time as MType");
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
        assert(c1.size() == c2.size());
        assert((c1.size() & (SIMDHolder::nels - 1)) == 0);
        sum_arrays(reinterpret_cast<const SType *>(&c1[0]), reinterpret_cast<const SType *>(&c2[0]), reinterpret_cast<const SType *>(&*c1.cend()), arrh1, arrh2, arru, arrg1, arrg2, arreq);
    }
};

template<typename T>
inline void inc_counts(T &counts, const SIMDHolder *p, const SIMDHolder *pend) {
    static_assert(std::is_same_v<std::decay_t<decltype(counts[0])>, uint64_t>, "Counts must contain 64-bit integers.");
    SIMDHolder tmp;
    do {
        tmp = *p++;
        tmp.inc_counts(counts);
    } while(p < pend);
}

static inline std::array<uint64_t, 64> sum_counts(const SIMDHolder *p, const SIMDHolder *pend) {
    // Should add Contiguous Container requirement.
    std::array<uint64_t, 64> counts{0};
    inc_counts(counts, p, pend);
    return counts;
}

template<typename Container>
inline std::array<uint64_t, 64> sum_counts(const Container &con) {
    static_assert(std::is_same_v<std::decay_t<decltype(con[0])>, uint8_t>, "Container must contain 8-bit unsigned integers.");
    return sum_counts(reinterpret_cast<const SIMDHolder *>(&*std::cbegin(con)), reinterpret_cast<const SIMDHolder *>(&*std::cend(con)));
}
template<typename T, typename Container>
inline void inc_counts(T &counts, const Container &con) {
    static_assert(std::is_same_v<std::decay_t<decltype(con[0])>, uint8_t>, "Container must contain 8-bit unsigned integers.");
    return inc_counts(counts, reinterpret_cast<const SIMDHolder *>(&*std::cbegin(con)), reinterpret_cast<const SIMDHolder *>(&*std::cend(con)));
}

template<typename CoreType>
void parsum_helper(void *data_, long index, int tid) {
    parsum_data_t<CoreType> &data(*(parsum_data_t<CoreType> *)data_);
    uint64_t local_counts[64]{0};
    SIMDHolder tmp, *p((SIMDHolder *)&data.core_[index * data.pb_]),
                    *pend((SIMDHolder *)&data.core_[std::min(data.l_, (index+1) * data.pb_)]);
    do {
        tmp = *p++;
        tmp.inc_counts(local_counts);
    } while(p < pend);
    for(uint64_t i = 0; i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

inline std::set<uint64_t> seeds_from_seed(uint64_t seed, size_t size) {
    std::mt19937_64 mt(seed);
    std::set<uint64_t> rset;
    while(rset.size() < size) rset.emplace(mt());
    return rset;
}
template<typename T>
inline double ertl_ml_estimate(const T& c, unsigned p, unsigned q, double relerr) {
    const uint64_t m = 1ull << p;
    if (c[q+1] == m) return std::numeric_limits<double>::infinity();

    int kMin, kMax;
    for(kMin=0; c[kMin]==0; ++kMin);
    int kMinPrime = std::max(1, kMin);
    for(kMax=q+1; kMax && c[kMax]==0; --kMax);
    int kMaxPrime = std::min((int)q, kMax);
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
        double xPrime = ldexp(x, -std::max((int)kMaxPrime+1, kappaMinus1+2));
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
    std::array<uint64_t, 64> c1{0}, c2{0}, cu{0}, ceq{0}, cg1{0}, cg2{0};
    detail::joint_unroller ju;
    ju.sum_arrays(h1.core(), h2.core(), c1, c2, cu, cg1, cg2, ceq);
    const double cAX = ertl_ml_estimate(c1, h1.p(), h1.q());
    const double cBX = ertl_ml_estimate(c2, h2.p(), h2.q());
    const double cABX = ertl_ml_estimate(cu, h1.p(), h1.q());
    // std::fprintf(stderr, "Made initials: %lf, %lf, %lf\n", cAX, cBX, cABX);
    std::array<uint64_t, 64> countsAXBhalf{0};
    std::array<uint64_t, 64> countsBXAhalf{0};
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
#if VERIFY_SIMD_JOINT
    auto other = ertl_joint_simple<HllType>(h1, h2);
    std::fprintf(stderr, "Made other\n");
    std::fprintf(stderr, "other: %lf|%lf|%lf. This: %lf|%lf|%lf\n", other[0], other[1], other[2], ret[0], ret[1], ret[2]);
    assert(ret == other);
#endif
    return ret;
}

template<typename HllType>
std::array<double, 3> ertl_joint(HllType &h1, HllType &h2) {
    if(h1.get_jestim() != ERTL_JOINT_MLE) h1.csum(), h2.csum();
    return ertl_joint(static_cast<const HllType &>(h1), static_cast<const HllType &>(h2));
}

// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It has a 1-1 mapping from any one 64-bit integer to another
// and can be inverted with irving_inv_hash.
static INLINE uint64_t wang_hash(uint64_t key) noexcept {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

struct WangHash {
    using Space = vec::SIMDTypes<uint64_t>;
    using Type = typename vec::SIMDTypes<uint64_t>::Type;
    using VType = typename vec::SIMDTypes<uint64_t>::VType;
    auto operator()(uint64_t key) const {
        return wang_hash(key);
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
};

struct MurFinHash {
    using Space = vec::SIMDTypes<uint64_t>;
    using Type = typename vec::SIMDTypes<uint64_t>::Type;
    using VType = typename vec::SIMDTypes<uint64_t>::VType;
    INLINE uint64_t operator()(uint64_t key) const {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccd;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53;
        key ^= key >> 33;
        return key;
    }
    INLINE Type operator()(Type key) const {
        return this->operator()(*((VType *)&key));
    }
    INLINE Type operator()(VType key) const {
#if (HAS_AVX_512)
        static const Type mul1 = Space::set1(0xff51afd7ed558ccd);
        static const Type mul2 = Space::set1(0xc4ceb9fe1a85ec53);
#endif

#if !NDEBUG
        auto save = key.arr_[0];
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xff51afd7ed558ccd;});
#  else
        key = Space::mullo(key.simd_, mul1); // h *= 0xff51afd7ed558ccd;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xc4ceb9fe1a85ec53;});
#  else
        key = Space::mullo(key.simd_, mul2); // h *= 0xc4ceb9fe1a85ec53;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        assert(this->operator()(save) == key.arr_[0]);
        return key.simd_;
    }
};

#ifdef roundup64
#undef roundu64
#endif
static INLINE uint64_t roundup64(size_t x) noexcept {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return ++x;
}

#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
    }} while(0)

constexpr INLINE int clz_manual( uint32_t x )
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}

// Overload
constexpr INLINE int clz_manual( uint64_t x )
{
  int n(0);
  if ((x & 0xFFFFFFFF00000000ull) == 0) {n  = 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (64 - 4));
  return n;
}

// clz wrappers. Apparently, __builtin_clzll is undefined for values of 0.
// However, by modifying our code to set a 1-bit at the end of the shifted
// region, we can guarantee that this does not happen for our use case.

#if __GNUC__ || __clang__
constexpr INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}
constexpr INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
constexpr INLINE unsigned clz(unsigned x) {
    return __builtin_clz(x);
}
constexpr INLINE unsigned ffs(unsigned long long x) {
    return __builtin_ffsll(x);
}
constexpr INLINE unsigned ffs(unsigned long x) {
    return __builtin_ffsl(x);
}
constexpr INLINE unsigned ffs(unsigned x) {
    return __builtin_ffs(x);
}
#else
#pragma message("Using manual clz instead of gcc/clang __builtin_*")
#error("Have not created a manual ffs function. Must be compiled with gcc or clang. (Or a compiler supporting it.)")
#define clz(x) clz_manual(x)
// https://en.wikipedia.org/wiki/Find_first_set#CLZ
// Modified for constexpr, added 64-bit overload.
#endif

static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz failed.");
static_assert(clz(0x0000000000000003ull) == 62, "64-bit clz failed.");
static_assert(clz(0x0000013333000003ull) == 23, "64-bit clz failed.");


constexpr double make_alpha(size_t m) {
    switch(m) {
        case 16: return .673;
        case 32: return .697;
        case 64: return .709;
        default: return 0.7213 / (1 + 1.079/m);
    }
}

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

// TODO: add a compact, 6-bit version
// For now, I think that it's preferable for thread safety,
// considering there's an intrinsic for the atomic load/store, but there would not
// be for bit-packed versions.



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
    std::vector<uint8_t, Allocator<uint8_t>> core_;
    double                 value_;
    uint32_t                  np_;
    uint8_t        is_calculated_:1;
    uint8_t                clamp_:1;
    uint8_t             nthreads_;
    EstimationMethod       estim_;
    JointEstimationMethod jestim_;
public:
    using HashType = HashStruct;
    const HashStruct        hf_;
#if LZ_COUNTER
    std::array<std::atomic<uint64_t>, 64> clz_counts_; // To check for bias in insertion
#endif

    uint64_t m() const {return static_cast<uint64_t>(1) << np_;}
    double alpha()          const {return make_alpha(m());}
    double relative_error() const {return 1.03896 / std::sqrt(static_cast<double>(m()));}
    // Constructor
    explicit hllbase_t(size_t np, EstimationMethod estim=ERTL_MLE, JointEstimationMethod jestim=ERTL_JOINT_MLE, int nthreads=-1, bool clamp=false):
        core_(static_cast<uint64_t>(1) << np),
        value_(0.), np_(np), is_calculated_(0), clamp_(clamp),
        nthreads_(nthreads > 0 ? nthreads: 1),
        estim_(estim), jestim_(jestim),
        hf_{}
#if LZ_COUNTER
        , clz_counts_{0}
#endif
    {
        //std::fprintf(stderr, "p = %u. q = %u. size = %zu\n", np_, q(), core_.size());
    }
    explicit hllbase_t(): hllbase_t(0, EstimationMethod::ERTL_MLE, JointEstimationMethod::ERTL_JOINT_MLE) {}
    hllbase_t(const char *path): hf_{} {read(path);}
    hllbase_t(const std::string &path): hllbase_t(path.data()) {}
    hllbase_t(gzFile fp): hllbase_t() {this->read(fp);}

    // Call sum to recalculate if you have changed contents.
    void sum() {
        const auto counts(detail::sum_counts(core_)); // std::array<uint64_t, 64>
        value_ = detail::calculate_estimate(counts, estim_, m(), np_, alpha());
        is_calculated_ = 1;
    }
    void csum() {
        if(!is_calculated_) sum();
    }

    // Returns cardinality estimate. Sums if not calculated yet.
    double creport() const {
        if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report."
                                                     " Try the report() function.");
        return value_;
    }
    double report() noexcept {
        csum();
        return creport();
    }

    // Returns error estimate
    double cest_err() const {
        if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report.");
        return relative_error() * creport();
    }
    double est_err()  noexcept {
        return cest_err();
    }
    // Returns string representation
    std::string to_string() const {
        std::string params(std::string("p:") + std::to_string(np_) + '|' + EST_STRS[estim_] + ";");
        return (params + (is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                                         : desc_string()));
    }
    // Descriptive string.
    std::string desc_string() const {
        char buf[256];
        std::sprintf(buf, "Size: %u. nb: %llu. error: %lf. Is calculated: %s. value: %lf. Estimation method: %s\n",
                     np_, static_cast<long long unsigned int>(m()), relative_error(), is_calculated_ ? "true": "false", value_, EST_STRS[estim_]);
        return buf;
    }

    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
#if LZ_COUNTER
        ++clz_counts_[clz(((hashval << 1)|1) << (np_ - 1)) + 1];
#endif
    }

    INLINE void addh(uint64_t element) {
        element = hf_(element);
        add(element);
    }
    INLINE void addh(const std::string &element) {
#ifdef ENABLE_CLHASH
        if constexpr(std::is_same<HashStruct, clhasher>::value) {
            add(hf_(element));
        } else {
#endif
            add(std::hash<std::string>{}(element));
#ifdef ENABLE_CLHASH
        }
#endif
    }
    using VectorSpace = vec::SIMDTypes<uint64_t>;
    using VType       = typename vec::SIMDTypes<uint64_t>::VType;
    INLINE void addh(VType element) {
        element = hf_(element.simd_);
        add(element);
    }
    INLINE void add(VType element) {
        element.for_each([&](uint64_t &val) {add(val);});
    }
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(element))>, uint64_t>, "Must return 64-bit hash");
        add(hasher(element));
    }
#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(s, len))>, uint64_t>, "Must return 64-bit hash");
        add(hasher(s, len));
    }
#endif
    void parsum(int nthreads=-1, size_t pb=4096) {
        if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
        std::atomic<uint64_t> acounts[64];
        std::memset(acounts, 0, sizeof acounts);
        detail::parsum_data_t<decltype(core_)> data{acounts, core_, m(), pb};
        const uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
        kt_for(nthreads, detail::parsum_helper<decltype(core_)>, &data, nr);
        uint64_t counts[64];
        std::memcpy(counts, acounts, sizeof(counts));
        value_ = detail::calculate_estimate(counts, estim_, m(), np_, alpha());
        is_calculated_ = 1;
    }
    hllbase_t<HashStruct> compress(size_t new_np) const {
        // See Algorithm 3 in https://arxiv.org/abs/1702.01284
        // This is not very optimized.
        // I might later add support for doubling, c/o https://research.neustar.biz/2013/04/30/doubling-the-size-of-an-hll-dynamically-extra-bits/
        if(new_np == np_) return hllbase_t(*this);
        if(new_np > np_) throw std::runtime_error("Can't compress to a larger size. Current: "s + std::to_string(np_) + ". Requested new size: " + std::to_string(new_np));
        hllbase_t<HashStruct> ret(new_np, get_estim(), get_jestim(), nthreads_, clamp());
        size_t ratio = static_cast<size_t>(1) << (np_ - new_np);
        size_t b = 0;
        for(size_t i(0); i < (1ull << new_np); ++i) {
            size_t j(0);
            while(j < ratio && core_[j + b] == 0) ++j;
            if(j != ratio)
                ret.core_[i] = std::min(ret.q() + 1, j ? clz(j)+1: core_[b]);
            // Otherwise left at 0
            b += ratio;
        }
        return ret;
    }
    // Reset.
    void clear() {
        // Note: this can be accelerated with SIMD.
        std::fill(std::begin(core_), std::end(core_), 0u);
        value_ = is_calculated_ = 0;
    }
    hllbase_t(hllbase_t&&) = default;
    hllbase_t(const hllbase_t &other) = default;
    hllbase_t& operator=(const hllbase_t &other) {
        // Explicitly define to make sure we don't do unnecessary reallocation.
        if(core_.size() != other.core_.size())
            core_.resize(other.core_.size());
        std::memcpy(core_.data(), other.core_.data(), core_.size());
        np_ = other.np_;
        value_ = other.value_;
        is_calculated_ = other.is_calculated_;
        estim_ = other.estim_;
        nthreads_ = other.nthreads_;
        return *this;
    }
    hllbase_t& operator=(hllbase_t&&) = default;
    hllbase_t clone() const {
        return hllbase_t(np_, estim_, jestim_, nthreads_, clamp_);
    }

    hllbase_t &operator+=(const hllbase_t &other) {
        if(other.np_ != np_) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        unsigned i;
#if HAS_AVX_512 && __AVX512BW__
        __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
        const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
        for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]); // mm512_max_epu8 is available on with AVX512BW :(
        if(m() < 64) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __AVX2__
        __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
        const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
        for(i = 0; i < m() >> 5; ++i) els[i] = _mm256_max_epu8(els[i], oels[i]);
        if(m() < 32) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __SSE2__
        __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
        const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
        for(i = 0; i < m() >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
        if(m() < 16) for(; i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#else
        uint64_t *els(reinterpret_cast<__m128i *>(core_.data()));
        const uint64_t *oels(reinterpret_cast<const uint64_t *>(other.core_.data()));
        while(els < oels) *els = std::max(*els, *oels), ++els, ++oels;
#endif
        not_ready();
        return *this;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup64(new_size);
        clear();
        core_.resize(new_size);
        np_ = (std::size_t)std::log2(new_size);
    }
    EstimationMethod get_estim()       const {return  estim_;}
    JointEstimationMethod get_jestim() const {return jestim_;}
    void set_estim(EstimationMethod val)       {estim_  = val;}
    void set_jestim(JointEstimationMethod val) {jestim_ = val;}
    void set_jestim(uint16_t val) {jestim_ = (JointEstimationMethod)val;}
    void set_estim(uint16_t val)  {estim_  = (EstimationMethod)val;}
    // Getter for is_calculated_
    bool get_is_ready() const {return is_calculated_;}
    void not_ready() {is_calculated_ = false;}
    void set_is_ready() {is_calculated_ = true;}
    bool may_contain(uint64_t hashval) const {
        // This returns false positives, but never a false negative.
        return core_[hashval >> q()] >= clz(hashval << np_) + 1;
    }

    bool within_bounds(uint64_t actual_size) const {
        return std::abs(actual_size - creport()) < relative_error() * actual_size;
    }

    bool within_bounds(uint64_t actual_size) {
        return std::abs(actual_size - report()) < est_err();
    }
    const auto &core()    const {return core_;}
    const uint8_t *data() const {return core_.data();}

    uint32_t p() const {return np_;}
    uint32_t q() const {return (sizeof(uint64_t) * CHAR_BIT) - np_;}
    void free() {
        decltype(core_) tmp{};
        std::swap(core_, tmp);
    }
    void write(FILE *fp) const {
        write(fileno(fp));
    }
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t bf[]{is_calculated_, clamp_, estim_, jestim_, nthreads_};
        CW(fp, bf, sizeof(bf));
        CW(fp, &np_, sizeof(np_));
        CW(fp, &value_, sizeof(value_));
        CW(fp, core_.data(), core_.size() * sizeof(core_[0]));
#undef CW
    }
    void write(const char *path, bool write_gz=false) const {
        if(write_gz) {
            gzFile fp(gzopen(path, "wb"));
            if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
            write(fp);
            gzclose(fp);
        } else {
            std::FILE *fp(std::fopen(path, "wb"));
            if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
            write(fileno(fp));
            std::fclose(fp);
        }
    }
    void write(const std::string &path, bool write_gz=false) const {write(path.data(), write_gz);}
    void read(gzFile fp) {
#define CR(fp, dst, len) do {if((uint64_t)gzread(fp, dst, len) != len) throw std::runtime_error("Error reading from file.");} while(0)
        uint32_t bf[5];
        CR(fp, bf, sizeof(bf));
        is_calculated_ = bf[0];
        clamp_  = bf[1];
        estim_  = (EstimationMethod)bf[2];
        jestim_ = (JointEstimationMethod)bf[3];
        nthreads_ = bf[4];
        CR(fp, &np_, sizeof(np_));
        CR(fp, &value_, sizeof(value_));
        core_.resize(m());
        CR(fp, core_.data(), core_.size());
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {
        read(path.data());
    }
    void write(int fileno) const {
        uint32_t bf[]{is_calculated_, clamp_, estim_, jestim_, nthreads_};
        ::write(fileno, bf, sizeof(bf));
        ::write(fileno, &np_, sizeof(np_));
        ::write(fileno, &value_, sizeof(value_));
        ::write(fileno, core_.data(), core_.size() * sizeof(core_[0]));
    }
    void read(int fileno) {
        uint32_t bf[5];
        ::read(fileno, bf, sizeof(bf));
        is_calculated_ = bf[0];
        clamp_         = bf[1];
        estim_         = (EstimationMethod)bf[2];
        jestim_        = (JointEstimationMethod)bf[3];
        nthreads_      = bf[4];
        ::read(fileno, &np_, sizeof(np_));
        ::read(fileno, &value_, sizeof(value_));
        core_.resize(m());
        ::read(fileno, core_.data(), core_.size());
    }
    hllbase_t operator+(const hllbase_t &other) const {
        if(other.p() != p())
            throw std::runtime_error(std::string("p (") + std::to_string(p()) + " != other.p (" + std::to_string(other.p()));
        hllbase_t ret(*this);
        ret += other;
        return ret;
    }
    double union_size(const hllbase_t &other) const {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) {
            using detail::SIMDHolder;
            assert(m() == other.m());
            using SType = typename SIMDHolder::SType;
            std::array<uint64_t, 64> counts{0};
            // We can do this because we use an aligned allocator.
            const SType *p1(reinterpret_cast<const SType *>(data())), *p2(reinterpret_cast<const SType *>(other.data()));
            for(SIMDHolder tmp;p1 < reinterpret_cast<const SType *>(&(*core().cend()));tmp.val = SIMDHolder::max_fn(*p1++, *p2++), tmp.inc_counts(counts));
            return detail::calculate_estimate(counts, get_estim(), m(), p(), alpha());
        }
        const auto full_counts = ertl_joint(*this, other);
        return full_counts[0] + full_counts[1] + full_counts[2];
    }
    double jaccard_index(hllbase_t &h2) {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) csum(), h2.csum();
        return const_cast<hllbase_t &>(*this).jaccard_index(const_cast<const hllbase_t &>(h2));
    }
    std::pair<double, bool> bjaccard_index(hllbase_t &h2) {
        if(jestim_ != JointEstimationMethod::ERTL_JOINT_MLE) csum(), h2.csum();
        return const_cast<hllbase_t &>(*this).bjaccard_index(const_cast<const hllbase_t &>(h2));
    }
    std::pair<double, bool> bjaccard_index(const hllbase_t &h2) const {
        if(jestim_ == JointEstimationMethod::ERTL_JOINT_MLE) {
            auto full_cmps = ertl_joint(*this, h2);
            auto ret = full_cmps[2] / (full_cmps[0] + full_cmps[1] + full_cmps[2]);
            return std::make_pair(ret, ret > relative_error());
        }
        const double us = union_size(h2);
        const double ret = std::max(0., creport() + h2.creport() - us) / us;
        return std::make_pair(ret, ret > relative_error());
    }
    double jaccard_index(const hllbase_t &h2) const {
        if(jestim_ == JointEstimationMethod::ERTL_JOINT_MLE) {
            auto full_cmps = ertl_joint(*this, h2);
            const auto ret = full_cmps[2] / (full_cmps[0] + full_cmps[1] + full_cmps[2]);
            return clamp_ && ret < relative_error() ? 0.: ret;
        }
        const double us = union_size(h2);
        const double ret = (creport() + h2.creport() - us) / us;
        return clamp_ ? ret < relative_error() ? 0.: ret
                      : std::max(0., ret);
    }
    size_t size() const {return size_t(m());}
    bool clamp()  const {return clamp_;}
    void set_clamp(bool val) {clamp_ = val;}
    static constexpr unsigned min_size() {
        return std::log2(sizeof(detail::SIMDHolder));
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

// Returns the size of the set intersection
template<typename HllType>
inline double intersection_size(HllType &first, HllType &other) noexcept {
    first.csum(), other.csum();
    return intersection_size((const HllType &)first, (const HllType &)other);
}

template<typename HllType> inline double jaccard_index(const HllType &h1, const HllType &h2) {return h1.jaccard_index(h2);}
template<typename HllType> inline double jaccard_index(HllType &h1, HllType &h2) {return h1.jaccard_index(h2);}
template<typename HllType> inline std::pair<double, bool> bjaccard_index(const HllType &h1, const HllType &h2) {return h1.bjaccard_index(h2);}
template<typename HllType> inline std::pair<double, bool> bjaccard_index(HllType &h1, HllType &h2) {return h1.bjaccard_index(h2);}

// Returns a HyperLogLog union
template<typename HllType>
static inline double union_size(const HllType &h1, const HllType &h2) {return h1.union_size(h2);}

template<typename HllType>
static inline double intersection_size(const HllType &h1, const HllType &h2) {
    if(h1.clamp()) {
        const auto us = union_size(h1, h2), is = h1.creport() + h2.creport() - us;
        return is < h1.relative_error() * us ? 0.: is;
    } // else
    return std::max(0., h1.creport() + h2.creport() - union_size(h1, h2));
}


template<typename HashFunc=WangHash>
class hlldub_base_t: public hllbase_t<HashFunc> {
    // hlldub_base_t inserts each value twice (forward and reverse)
    // and simply halves cardinality estimates.
public:
    template<typename... Args>
    hlldub_base_t(Args &&...args): hll_t(std::forward<Args>(args)...) {}
    INLINE void add(uint64_t hashval) {
        hllbase_t<HashFunc>::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((this->m()) - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
            this->core_[index] < lzt;
            __sync_bool_compare_and_swap(this->core_.data() + index, this->core_[index], lzt));
#else
        const uint32_t index(hashval & (this->m() - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
        this->core_[index] = std::min(this->core_[index], lzt);
#endif
    }
    double report() {
        this->sum();
        return this->creport();
    }
    double creport() const {
        return hllbase_t<HashFunc>::creport() * 0.5;
    }
    bool may_contain(uint64_t hashval) const {
        return hllbase_t<HashFunc>::may_contain(hashval) && this->core_[hashval & ((this->m()) - 1)] >= ffs(hashval >> this->p());
    }

    INLINE void addh(uint64_t element) {add(this->hf_(element));}
};
using hlldub_t = hlldub_base_t<>;

template<typename HashFunc=WangHash>
class dhllbase_t: public hllbase_t<HashFunc> {
    // dhllbase_t is a bidirectional hll sketch which does not currently support set operations
    // It is based on the idea that the properties of a hll sketch work for both leading and trailing zeros and uses them as independent samples.
    std::vector<uint8_t, Allocator<uint8_t>> dcore_;
    using hll_t = hllbase_t<HashFunc>;
public:
    template<typename... Args>
    dhllbase_t(Args &&...args): hll_t(std::forward<Args>(args)...),
                            dcore_(1ull << hll_t::p()) {
    }
    void sum() {
        uint64_t fcounts[64]{0};
        uint64_t rcounts[64]{0};
        const auto &core(hll_t::core());
        for(size_t i(0); i < core.size(); ++i) {
            // I don't this can be unrolled and LUT'd.
            ++fcounts[core[i]]; ++rcounts[dcore_[i]];
        }
        this->value_  = detail::calculate_estimate(fcounts, this->estim_, this->m(), this->np_, this->alpha());
        this->value_ += detail::calculate_estimate(rcounts, this->estim_, this->m(), this->np_, this->alpha());
        this->value_ *= 0.5;
        this->is_calculated_ = 1;
    }
    void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((this->m()) - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
            dcore_[index] < lzt;
            __sync_bool_compare_and_swap(dcore_.data() + index, dcore_[index], lzt));
#else
        const uint32_t index(hashval & (this->m() - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
        dcore_[index] = std::min(dcore_[index], lzt);
#endif
    }
    void addh(uint64_t element) {add(this->hf_(element));}
    bool may_contain(uint64_t hashval) const {
        return hll_t::may_contain(hashval) && dcore_[hashval & ((this->m()) - 1)] >= ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1));
    }
};
using dhll_t = dhllbase_t<>;


template<typename HashFunc=WangHash>
class seedhllbase_t: public hllbase_t<HashFunc> {
protected:
    uint64_t seed_; // 64-bit integers are xored with this value before passing it to a hash.
                          // This is almost free, in the content of
    using hll_t = hllbase_t<HashFunc>;
public:
    template<typename... Args>
    seedhllbase_t(uint64_t seed, Args &&...args): hll_t(std::forward<Args>(args)...), seed_(seed) {
        if(seed_ == 0) std::fprintf(stderr,
            "[W:%s:%d] Note: seed is set to 0. No more than one of these at a time should have this value, and this is only for the purpose of multiplying hashes."
            " Also, if you are only using one of these at a time, don't use seedhllbase_t, just use hll_t and save yourself an xor per insertion"
            ", not to mention a 64-bit integer in space.", __PRETTY_FUNCTION__, __LINE__);
    }
    seedhllbase_t(gzFile fp): hll_t() {
        this->read(fp);
    }
    void addh(uint64_t element) {
        element ^= seed_;
        this->add(wang_hash(element));
    }
    uint64_t seed() const {return seed_;}
    void write(const char *fn, bool write_gz) {
        if(write_gz) {
            gzFile fp = gzopen(fn, "wb");
            if(fp == nullptr) throw std::runtime_error("Could not open file.");
            this->write(fp);
            gzclose(fp);
        } else {
            std::FILE *fp = std::fopen(fn, "wb");
            if(fp == nullptr) throw std::runtime_error("Could not open file.");
            this->write(fileno(fp));
            std::fclose(fp);
        }
    }
    void write(gzFile fp) const {
        hll_t::write(fp);
        gzwrite(fp, &seed_, sizeof(seed_));
    }
    void read(gzFile fp) {
        hll_t::read(fp);
        gzread(fp, &seed_, sizeof(seed_));
    }
    void write(int fn) const {
        hll_t::write(fn);
        ::write(fn, &seed_, sizeof(seed_));
    }
    void read(int fn) {
        hll_t::read(fn);
        ::read(fn, &seed_, sizeof(seed_));
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        this->read(fp);
        gzclose(fp);
    }
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(element))>, uint64_t>, "Must return 64-bit hash");
        add(detail::finalize(hasher(element) ^ seed_));
    }

#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(s, len))>, uint64_t>, "Must return 64-bit hash");
        add(detail::finalize(hasher(s, len) ^ seed_));
    }
#endif
};
using seedhll_t = seedhllbase_t<>;
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
} // namespace sort

template<typename SeedHllType=hll_t>
class hlfbase_t {
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<SeedHllType>                      hlls_;
    std::vector<uint64_t, Allocator<uint64_t>>   seeds_;
    std::vector<double>                         values_;
    mutable double                               value_;
    bool                                 is_calculated_;
    using HashType     = typename SeedHllType::HashType;
    const HashType                                  hf_;
public:
    template<typename... Args>
    hlfbase_t(size_t size, uint64_t seedseed, Args &&... args): value_(0), is_calculated_(0), hf_{} {
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
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
        this->write(fp);
        gzclose(fp);
    }
    void clear() {
        value_ = is_calculated_ = 0;
        for(auto &hll: hlls_) hll.clear();
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
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

    bool may_contain(uint64_t element) const {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0); i < Space::COUNT;) if(!hlls_[k++].may_contain(key.arr_[i++])) return false;
            } while(sptr < eptr);
            return true;
        } else { // if size() >= Space::COUNT
            for(unsigned i(0); i < size(); ++i) if(!hlls_[i].may_contain(hf_(element ^ seeds_[i]))) return false;
            return true;
        }
    }
    void addh(uint64_t val) {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            const SType element = Space::set1(val);
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0) ; i < Space::COUNT; hlls_[k++].add(key.arr_[i++]));
                assert(k <= size());
            } while(sptr < eptr);
        } else while(k < size()) hlls_[k].add(hf_(val ^ seeds_[k])), ++k;
    }
    double creport() const {
        if(is_calculated_) return value_;
        double ret(hlls_[0].creport());
        for(size_t i(1); i < size(); ret += hlls_[i++].creport());
        ret /= static_cast<double>(size());
        value_ = ret;
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
        if(values_.empty())
            values_.reserve(size());
        values_.clear();
        for(auto &hll: hlls_) values_.emplace_back(hll.report());
        if(size() & 1) {
            if(size() < 32)
                sort::insertion_sort(std::begin(values_), std::end(values_));
            else
                std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) + 1, std::end(values_));
            return values_[size() >> 1];
        }
        if(size() < 32) {
            sort::insertion_sort(std::begin(values_), std::end(values_));
            return .5 * (values_[size() >> 1] + values_[(size() >> 1) - 1]);
        }
        std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) - 1, std::end(values_));
        return .5 * (values_[(values_.size() >> 1) - 1] + *std::min_element(std::cbegin(values_) + (size() >> 1), std::cend(values_)));
    }
    // Attempt strength borrowing across hlls with different seeds
    double chunk_report() const {
        if((size() & (size() - 1)) == 0) {
            std::array<uint64_t, 64> counts{0};
            for(const auto &hll: hlls_) detail::inc_counts(counts, hll.core());
            const auto diff = (sizeof(uint32_t) * CHAR_BIT - clz((uint32_t)size()) - 1);
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
template<typename HashType=WangHash>
class chlf_t { // contiguous hyperlogfilter
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<uint64_t, Allocator<uint64_t>>   seeds_;
    std::vector<double>                         values_;
    EstimationMethod                             estim_;
    JointEstimationMethod                       jestim_;
    const uint32_t                                subp_;
    const uint16_t                                  ns_; // number of subsketches
    const uint16_t                                  np_;
    mutable double                               value_;
    bool                                 is_calculated_;
    const HashType                                  hf_;
    std::vector<uint8_t, Allocator<uint8_t>>      core_;
public:
    chlf_t(size_t l2ss, EstimationMethod estim,
           JointEstimationMethod jestim, unsigned p, uint64_t seedseed=0): estim_(estim), jestim_(jestim), subp_(p - l2ss), ns_(1 << l2ss), np_(p), value_(0), is_calculated_(0), hf_{}, core_(1ull << p) {
        auto sfs = detail::seeds_from_seed(seedseed ? seedseed: ns_ + l2ss * p + 137, ns_);
        seeds_ = std::vector<uint64_t, Allocator<uint64_t>>(std::begin(sfs), std::end(sfs));
        assert(sfs.size());
    }
    auto nbytes()    const {return core_.size();}
    auto nsketches() const {return ns_;}
    auto m() const {return core_.size();}
    auto subp() const {return subp_;}
    auto subq() const {return (sizeof(uint64_t) * CHAR_BIT) - subp_;}
    void add(uint64_t hashval, unsigned subidx) {
#ifndef NOT_THREADSAFE
        // subidx << subp gets us to the subtable, hashval >> subq gets us to the slot within that table.
        for(const uint32_t index((hashval >> subq()) + (subidx << subp())), lzt(clz(((hashval << 1)|1) << (subp() - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index((hashval >> subq()) + (subidx << subp())), lzt(clz(((hashval << 1)|1) << (subp() - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
        
    }
    using Space = vec::SIMDTypes<uint64_t>;
    using SType = typename Space::Type;
    using VType = typename Space::VType;
    INLINE bool may_contain(uint64_t val) const {
        unsigned k = 0;
        if(ns_ >= Space::COUNT) {
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            const SType element = Space::set1(val);
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(const auto val: key.arr_)
                    if((clz(((val << 1)|1) << (subp() - 1)) + 1) > core_[(val >> subq()) + (k++ << subp())])
                        return false;
                assert(k <= ns_);
            } while(sptr < eptr);
        } else while(k < ns_) {
            auto tmp = hf_(val ^ seeds_[k]);
            if((clz(((tmp << 1)|1) << (subp() - 1)) + 1) > core_[(tmp >> subq()) + (k++ << subp())]) return false;
        }
        return true;
    }
    void addh(uint64_t val) {
        unsigned k = 0;
        if(ns_ >= Space::COUNT) {
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            const SType element = Space::set1(val);
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                key.for_each([&](const uint64_t &val) {add(val, k++);});
                assert(k <= ns_);
            } while(sptr < eptr);
        } else for(;k < ns_;add(hf_(val ^ seeds_[k]), k), ++k);
    }
    double chunk_report() const {
#if NON_POW2
        if((size() & (size() - 1)) == 0) {
#endif
        std::array<uint64_t, 64> counts{0};
        detail::inc_counts(counts, core_);
        //const auto diff = (sizeof(uint32_t) * CHAR_BIT - clz((uint32_t)size()) - 1); Maybe do this instead of storing l2ns_?
        return detail::calculate_estimate(counts, estim_, core_.size(),
                                          np_, make_alpha(core_.size())) / ns_;
#if NON_POW2
        } else {
            std::fprintf(stderr, "chunk_report is currently only supported for powers of two.");
            return creport();
            // Could try weight averaging, but currently I just report default when size is not a power of two.
        }
#endif
    }
    size_t size() const {return core_.size();}
    void clear() {
        std::fill(std::begin(core_), std::end(core_), 0);
        is_calculated_ = 0;
        value_         = 0;
    }
};

} // namespace hll

#endif // #ifndef HLL_H_
