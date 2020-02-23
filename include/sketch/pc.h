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
    void addh(uint64_t item) {
        wy::wyhash64_stateless(&item);
        add(item);
    }
    double report() const {
        return detail::R(sketch_) * 1.292808;
    }
    T getregister() const {return sketch_;}
};

template<typename T>
class PCSA {
    std::unique_ptr<T[]> counters_;
    const size_t n_;
public:
    PCSA(size_t n): counters_(new T[n]), n_(n) {
        std::memset(counters_.get(), 0, sizeof(T) * n_);
    }
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
        return n_ * 1.292808 * std::pow(2, double(std::accumulate(counters_.get(), counters_.get() + n_, 0u, [](auto x, auto y) {return detail::r(y) + x;})) / n_);
    }
};

} // pc
using pc::PCSA;
using pc::ProbabilisticCounter;

} // sketch

#endif /* PROBCOUNTING_H__ */
