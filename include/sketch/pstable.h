#pragma once
#ifndef SKETCH_PSTABLE_H__
#define SKETCH_PSTABLE_H__
/*
 * Canonical p-stable random variate generators, shared by the p-stable LSH
 * implementations (rnla.h's PStableSketcher and the Poisson-LSH machinery).
 *
 * cms_distribution is the Chambers-Mallows-Stuck generator for skewed
 * stable laws with stability parameter a in (0, 2] and skewness b in
 * [-1, 1]:
 *
 *     Chambers, J. M., Mallows, C. L., & Stuck, B. W. (1976). "A method
 *     for simulating stable random variables." JASA 71(354), 340-344.
 *
 * Special cases: a=2 is Gaussian with variance 2 (independent of b),
 * a=1 is Cauchy with location/skew b.
 *
 * This header is the tracked canonical copy; include/sketch/poissonlsh.h
 * (untracked) carries a duplicate and should be migrated to include this.
 */
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace sketch {
inline namespace pstable {

static constexpr double PSTABLE_PI = 3.14159265358979323846264338327950288;
static constexpr double PSTABLE_PI_2 = 1.57079632679489661923132169163975144;

template<typename FT>
struct clipped_cauchy_distribution {
    // Cauchy clipped to [-max, max]; keeps operator() bounded, at the cost
    // of slightly lighter tails than the true Cauchy law.
    const FT max_;
    mutable std::cauchy_distribution<FT> dist_;
    FT min() const {return -max_;}
    FT max() const {return max_;}
    clipped_cauchy_distribution(FT max=1000.): max_(std::abs(max)) {}
    template<typename RNG>
    FT operator()(RNG &x) const {
        FT ret = dist_(x);
        if(ret > max_) ret = max_;
        if(ret < -max_) ret = -max_;
        return ret;
    }
    void reset() {}
};

template<typename FT>
struct cms_distribution {
    // p-stable sampler for stability parameter a in (0, 2] and skewness
    // b in [-1, 1], via the Chambers-Mallows-Stuck method.
public:
    const FT a_, b_;
private:
    FT t1_, t2_, t2i_, t1sp1_;
    mutable std::uniform_real_distribution<FT> urd_;
    mutable std::exponential_distribution<FT> expd_;
public:
    cms_distribution(double a=2., double b=0): a_(a), b_(b)
    {
        if(a <= 0 || a > 2) throw std::out_of_range("[cms_distribution] stability parameter a must be in (0, 2]");
        if(b < -1 || b > 1) throw std::out_of_range("[cms_distribution] skewness b must be in [-1, 1]");
        t1_  = -b_ * std::tan(PSTABLE_PI_2 * a_);
        t2_  = a_ == 1. ? FT(PSTABLE_PI_2) : std::atan(-t1_) / a_;
        t2i_ = 1. / t2_;
        t1sp1_ = std::pow(1. + t1_ * t1_, .5 / a_);
    }
    template<typename RNG>
    FT operator()(RNG &rng) const {
        const FT U = (urd_(rng) - FT(.5)) * PSTABLE_PI, W = expd_(rng);
        FT s;
        if(a_ == 1.) {
            const FT bu = b_ * U;
            s = t2i_ * ((PSTABLE_PI_2 + bu) * std::tan(U) - b_ * std::log(PSTABLE_PI_2 * W * std::cos(U) / (PSTABLE_PI_2 + bu)));
        } else {
            const FT ainv = 1. / a_;
            const FT at2u = a_ * (t2_ + U);
            s = t1sp1_
                * std::sin(at2u) / std::pow(std::cos(U), ainv)
                * std::pow(std::cos(U - at2u) / W, (1. - a_) * ainv);
        }
        return s;
    }
};

} // namespace pstable
} // namespace sketch

#endif // SKETCH_PSTABLE_H__
