#ifndef KAHAN_SUMMATION_H__
#define KAHAN_SUMMATION_H__
#include "sketch/macros.h"

namespace sketch { namespace kahan {

template<typename T>
INLINE T update(T &sum, T &carry, T increment) {
    increment -= carry;
    T tmp = sum + increment;
    carry = (tmp - sum) - increment;
    return sum = tmp;
}

}} // sketch::kahan

#endif
