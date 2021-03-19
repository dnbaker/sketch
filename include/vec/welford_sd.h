#ifndef WELFORD_ONLINE_STDEV_H__
#define WELFORD_ONLINE_STDEV_H__

// based on John D. Cook's blog https://www.johndcook.com/blog/standard_deviation/
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "blaze/Math.h"

namespace stats {

template<typename T, typename SizeType=std::uint64_t,
         typename=typename std::enable_if<std::is_floating_point<T>::value>::type,
         typename=typename std::enable_if<std::is_unsigned<SizeType>::value>::type>
class OnlineSD {
    T old_mean_, new_mean_, olds_, news_;
    SizeType n_;
public:
    OnlineSD() {std::memset(this, 0, sizeof(*this));}

    void add(T x)
    {
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (__builtin_expect(++n_ == 1, 0)) old_mean_ = new_mean_ = x, olds_ = 0.;
        else
        {
            new_mean_ = old_mean_ + (x - old_mean_)/n_;
            news_ = olds_ + (x - old_mean_)*(x - new_mean_);
            // set up for next iteration
            old_mean_ = new_mean_, olds_ = news_;
        }
    }
    size_t n()   const {return n_;}
    T mean()     const {return n_ ? new_mean_: 0.0;}
    T variance() const {return n_ > 1 ? news_ / (n_ - 1): 0.0;}
    T stdev()    const {return std::sqrt(variance());}
};

template<typename VecType=::blaze::DynamicVector<double, ::blaze::rowVector>, typename SizeType=std::uint64_t>
class OnlineVectorSD {
    VecType old_mean_, new_mean_, olds_, news_;
    SizeType n_;
public:

    template<typename VType2>
    OnlineVectorSD(const VType2 &vec): OnlineVectorSD(vec.size()) {
        add(vec);
    }
    OnlineVectorSD(size_t d): old_mean_(d, 0), new_mean_(d, 0), olds_(d, 0), news_(d, 0), n_(0) {}

    template<typename VType2>
    void add(const VType2 &x)
    {
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (__builtin_expect(++n_ == 1, 0)) old_mean_ = new_mean_ = x, olds_ = 0.;
        else
        {
            new_mean_ = old_mean_ + (x - old_mean_)* (1./n_);
            news_ = olds_ + (x - old_mean_)*(x - new_mean_);
            // set up for next iteration
            old_mean_ = new_mean_, olds_ = news_;
        }
    }
#ifndef NDEBUG
#define ASSERTFULL() do {if(!n_) {throw std::runtime_error("Cannot calculate stats on an empty stream.");}} while(0)
#endif
    size_t n()   const {return n_;}
    const VecType &mean()     const {ASSERTFULL(); return new_mean_;}
    const VecType &variance() const {ASSERTFULL(); return news_ / (n_ - 1);}
    const VecType &stdev()    const {ASSERTFULL(); return blaze::sqrt(variance());}
};

template<typename T=float, typename SizeType=std::int64_t,
         typename=typename std::enable_if<std::is_floating_point<T>::value>::type,
         typename=typename std::enable_if<std::is_integral<SizeType>::value>::type>
class OnlineStatistics
{
public:
    OnlineStatistics() {clear();}
    void clear() {std::memset(this, 0, sizeof(*this));}
    void add(T x) {
        T delta, delta_n, delta_n2, term1;

        SizeType n1 = n_++;
        delta = x - m1_;
        delta_n = delta / n_;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        m1_ += delta_n;
        m4_ += term1 * delta_n2 * (n_*n_ - 3*n_ + static_cast<SizeType>(3)) + \
              6. * delta_n2 * m2_ - 4. * delta_n * m3_;
        m3_ += term1 * delta_n * (n_ - 2) - 3 * delta_n * m2_;
        m2_ += term1;
    }
    SizeType n() const {return n_;}
    T mean() const {return m1_;}
    T variance() const {return m2_/(n_- (n_ > 1.0));}
    T stdev() const    {return std::sqrt(variance());}
    T skewness() const {
        assert(m2_ >= 0.);
        return std::sqrt(static_cast<T>(n_)) * m3_/ std::pow(m2_, 1.5);
    }
    T kurtosis() const {return static_cast<T>(n_)*m4_ / (m2_*m2_) - 3.0;}

    template<typename T1, typename SizeType2>
    OnlineStatistics& operator+=(const OnlineStatistics<T1, SizeType2>& b)
    {
        auto newn = this->n + b.n;

        const T delta = b.m1_ - this->m1_;
        const T delta2 = delta*delta;
        const T delta3 = delta*delta2;
        const T delta4 = delta2*delta2;

        auto newm1 = (this->n*this->m1_ + b.n*b.m1_) / newn;
        auto newm2 = this->m2_ + b.m2_ +
                      delta2 * this->n * b.n / newn;

        auto newm3 = this->m3_ + b.m3_ +
                     delta3 * this->n * b.n * (this->n - b.n)/(newn*newn);
        newm3 += 3.0*delta * (this->n*b.m2_ - b.n*this->m2_) / newn;

        auto newm4 = this->m4_ + b.m4_ + delta4*this->n*b.n * (this->n*this->n - this->n*b.n + b.n*b.n) /
                      (newn*newn*newn);
        newm4 += 6.0*delta2 * (this->n*this->n*b.m2_ + b.n*b.n*this->m2_)/(newn*newn) +
                 4.0*delta*(this->n*b.m3_ - b.n*this->m3_) / newn;
        this->n = newn;
        this->m4_ = newm4;
        this->m3_ = newm3;
        this->m2_ = newm2;
        this->m1_ = newm1;
        return *this;
    }

private:
    T m1_, m2_, m3_, m4_;
    SizeType n_;
};

template<typename T, typename SizeType>
auto operator+(const OnlineStatistics<T, SizeType> &a, const OnlineStatistics<T, SizeType> &b) {
    auto ret(a); // Copy a
    ret += b;
    return ret;
}

} // namespace stats

#endif /* #ifndef WELFORD_ONLINE_STDEV_H__ */
