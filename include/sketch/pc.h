#ifndef PROBCOUNTING_H__
#define PROBCOUNTING_H__
#include "sketch/integral.h"
#include "sketch/hash.h"

namespace sketch {

namespace pc {

namespace detail {
template<typename T>
INLINE T R(T x) {return ~x & (x + 1);}
template<typename T>
INLINE T r(T x) {
    //return ctz(~x);
    return popcount(R(x) - 1);
}
} // namespace detail

template<typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
class ProbabilisticCounter {
protected:
    T sketch_;
public:
    ProbabilisticCounter(): sketch_(0) {
    }
    void add(uint64_t hv) {
        sketch_ |= detail::R(hv);
    }
    ProbabilisticCounter &operator|=(const ProbabilisticCounter &o) {
        sketch_ |= o.sketch_;
        return *this;
    }
    void addh(uint64_t item) {
        wy::wyhash64_stateless(&item);
        add(item);
    }
    double report() const {
        return detail::R(sketch_) * 1.292808;
    }
    T getregister() const {return sketch_;}
};

template<typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
class PCSA {
    /*
     * Note: See https://arxiv.org/abs/2007.08051
             for recent theory on space/accuracy results.
     */
    std::unique_ptr<T[]> counters_;
    const size_t n_;
public:
    PCSA(size_t n): counters_(new T[n]), n_(n) {
        std::memset(counters_.get(), 0, sizeof(T) * n_);
    }
    PCSA(const PCSA &o): counters_(new T[o.n_]), n_(o.n_) {
        std::memcpy(counters_.get(), o.counters_.get(), sizeof(T) * n_);
    }
    PCSA(PCSA &&o) = default;
    PCSA &operator|=(const PCSA &o) {
        for(unsigned i = 0; i < n_; ++i) counters_[i] |= o.counters_[i];
        return *this;
    }
    void addh(uint64_t value) {
        add(value);
    }
    void add(uint64_t value) {
        auto ind = value % n_;
        value /= n_;
        counters_[ind] |= detail::R(value);
    }
    double report() const {
        CONST_IF(sizeof(T) == 4) {
        /* Notes: this could be accelerated for more cases.
           1. Apply R(x) - 1 using SIMD
           2. Apply popcount using SIMD
           3. convert to floats
           4. Accumulate into a result
        */
#if __AVX2__
            __m256 sums = _mm256_setzero_ps();
            static constexpr size_t nper = sizeof(__m256i) / 4;
            const size_t nsimd = n_ / nper;
            size_t i;
            for(i = 0; i < nsimd; ++i) {
                __m256i vals = _mm256_loadu_si256((const __m256i *)&counters_[i * nper]);
                // x = R(x) - 1 = (~x & (x + 1))
                __m256i vx = _mm256_andnot_si256(vals, _mm256_add_epi32(vals, _mm256_set1_epi32(1)));
                // x = popcount(x)
                auto start = (uint32_t *)&vx;
                SK_UNROLL_8
                for(unsigned i = 0; i < nper; ++i) {
                    start[i] = popcount(start[i]);
                }
                // x = vector of floats via popcount(x)
                // Acccumulate
                sums = _mm256_add_ps(sums, _mm256_cvtepi32_ps(vx));
            }
            // Reduce
            double sum = 0.;
            for(unsigned i = 0; i < nper; ++i) sum += ((float *)&sums)[i];
            for(i *= nper; i < n_; ++i) {
                sum += detail::r(counters_[i]);
            }
            sum /= n_;
            return n_ * 1.292808 * sum * sum;
#endif
        }
        double mean =
            double(std::accumulate(counters_.get(), counters_.get() + n_, 0u, [](auto x, auto y) {
                return detail::r(y) + x;
            })) / n_;
        return n_ * 1.292808 * std::pow(2, mean);
    }
};

} // pc
using pc::PCSA;
using pc::ProbabilisticCounter;

} // sketch

#endif /* PROBCOUNTING_H__ */
