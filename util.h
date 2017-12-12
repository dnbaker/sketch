#ifndef _HLL_UTIL_H__
#define _HLL_UTIL_H__
#include <limits>
#include "logutil.h"

namespace hll {
namespace detail {

// Based off https://github.com/oertl/hyperloglog-sketch-estimation-paper/blob/master/c%2B%2B/cardinality_estimation.hpp
template<typename FloatType>
static constexpr FloatType gen_sigma(FloatType x, FloatType eps=1e-10) {
    if(x == 1.) return std::numeric_limits<FloatType>::infinity();
    FloatType z(x);
    for(FloatType zp(0.), y(1.); std::abs(z-zp) > eps;) {
        x *= x; zp = z; z += x * y; y += y;
        if(std::isnan(z)) {
            LOG_WARNING("Reached nan. Returning the last usable number.\n")
            return zp;
        }
    }
    return z;
}

template<typename FloatType>
static constexpr FloatType gen_tau(FloatType x) {
    if (x == 0. || x == 1.) {
        std::fprintf(stderr, "x is %f\n", (float)x);
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

} // detail

} // hll

#endif // #ifndef _HLL_UTIL_H__
