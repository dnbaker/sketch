#pragma once
#ifndef SKETCH_MLE_H__
#define SKETCH_MLE_H__
/*
 * Shared numerical routines for Ertl-style maximum-likelihood (ML)
 * cardinality estimation, used by the register-compressed HLL successors
 * (UltraLogLog, ExaLogLog).
 *
 * All of these sketches lead to the same shaped ML equation
 *
 *     maximize  e^{-x*a} * prod_{k=0}^{n} (1 - e^{-x/2^k})^{b[k]}
 *
 * which is solved with the robust secant-method-based algorithm 8 from
 *   O. Ertl, "New cardinality estimation algorithms for HyperLogLog
 *   sketches", arXiv:1702.01284, 2017.
 *
 * Reference behavior is cross-checked against the Hash4j implementation
 * (github.com/dynatrace-oss/hash4j).
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace sketch {
inline namespace mle {
namespace detail {

// Exact uint32_t 64-bit integer -> double conversion.
inline double u64_as_double(uint64_t l) {
    double d = double(l & 0x7fffffffffffffffULL);
    if(l & 0x8000000000000000ULL) d += 0x1p63;
    return d;
}

/*
 * Solves the ML equation above for x, where a and b[k] must be nonnegative
 * and at least one of them positive. n is the largest index that will be
 * read from b (b must have n + 1 entries). The result is returned with the
 * caller's own scaling (the sketch-specific factor).
 */
inline double solve_maximum_likelihood_equation(double a, const uint32_t *b, uint32_t n, double relerr) {
    if(a == 0.) return std::numeric_limits<double>::infinity();
    int kmax = int(n);
    while(kmax >= 0 && b[kmax] == 0) --kmax;
    if(kmax < 0) return 0.; // all elements in b are 0
    int kmin = kmax;
    uint32_t t = b[kmax];
    uint64_t s1 = t;
    double s2 = std::ldexp(double(t), kmax);
    for(int k = kmax - 1; k >= 0; --k) {
        t = b[k];
        if(t > 0) {
            s1 += t;
            s2 += std::ldexp(double(t), k);
            kmin = k;
        }
    }
    static constexpr double C0 = -1. / 3.;
    static constexpr double C1 = 1. / 45.;
    static constexpr double C2 = 1. / 472.5;

    double gprev = 0.;
    double x;
    if(s2 <= 1.5 * a) x = double(s1) / (0.5 * s2 + a);
    else              x = std::log1p(s2 / a) * (double(s1) / s2);
    double deltaX = x;
    const double s1d = double(s1);
    while(deltaX > x * relerr) {
        int kappaminus1;
        std::frexp(x, &kappaminus1);
        double xprime = std::ldexp(x, -std::max(kmax + 1, kappaminus1 + 2)); // in [0, 0.25]
        const double xprime2 = xprime * xprime;
        double h = xprime + xprime2 * (C0 + xprime2 * (C1 - xprime2 * C2));
        for(int k = kappaminus1; k >= kmax; --k) {
            const double hprime = 1. - h;
            h = (xprime + h * hprime) / (xprime + hprime);
            xprime += xprime;
        }
        double g = b[kmax] * h;
        for(int k = kmax - 1; k >= kmin; --k) {
            const double hprime = 1. - h;
            h = (xprime + h * hprime) / (xprime + hprime);
            xprime += xprime;
            g += b[k] * h;
        }
        g += x * a;
        if(gprev < g && g <= s1d) deltaX *= (g - s1d) / (gprev - g);
        else                        deltaX = 0.;
        x += deltaX;
        gprev = g;
    }
    return x;
}

/*
 * Martingale (HIP) estimator, following D. Ting (2014) and E. Cohen (2015)
 * as implemented in Hash4j's MartingaleEstimator. Feeding it the state-change
 * probability decrements reported by a sketch on every state-changing insert
 * yields an estimate with lower variance than register-based estimation,
 * at the cost of mergeability (the sketch may no longer be merged after
 * martingale estimation has been used for reporting; merging is still sound
 * for the sketch state itself, the martingale counter must be reset).
 */
class MartingaleEstimator {
    double estimate_;
    double state_change_probability_;
public:
    MartingaleEstimator(): estimate_(0.), state_change_probability_(1.) {}
    MartingaleEstimator(double estimate, double state_change_probability):
        estimate_(estimate), state_change_probability_(state_change_probability) {
        if(estimate < 0.) throw std::invalid_argument("[MartingaleEstimator] estimate must be non-negative");
        if(state_change_probability < 0. || state_change_probability > 1.)
            throw std::invalid_argument("[MartingaleEstimator] state change probability must be in [0, 1]");
    }
    void reset() {
        estimate_ = 0.;
        state_change_probability_ = 1.;
    }
    double estimate() const {return estimate_;}
    double get_estimate() const {return estimate_;}
    double state_change_probability() const {return state_change_probability_;}
    // To be called whenever the sketch state changed, with the decrement of
    // the sketch's overall state-change probability.
    void state_changed(double probability_decrement) {
        estimate_ += 1. / state_change_probability_;
        state_change_probability_ -= probability_decrement;
        if(state_change_probability_ <= 0.) state_change_probability_ = 0.; // numerical errors
    }
};

} // namespace detail
} // namespace mle
} // namespace sketch

#endif // SKETCH_MLE_H__
