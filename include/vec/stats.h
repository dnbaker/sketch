#ifndef VEC_STATS_H
#define VEC_STATS_H
#include "vec.h"
#include <stdexcept>
#include <limits>

#ifdef _BLAZE_CONFIG_CONFIG_H_
#define HAS_BLAZE 1
#else
#define HAS_BLAZE 0
#endif

namespace stats {
using namespace std::literals;

template<typename Container>
auto sum(const Container &c) {
    using Type = std::decay_t<decltype(c[0])>;
    using Space = vec::SIMDTypes<Type>;
    using VType = typename Space::VType;
    if(__builtin_expect(c.size() == 0, 0)) return static_cast<Type>(0);
    // If this saturates the type, this result will be wrong.
    VType tmp, tsum = 0;
    const VType *ptr = (const VType *)&*std::cbegin(c);
    auto eptr = &*std::end(c);
    if(Space::aligned(ptr)) {
        do {
            tmp.simd_ = Space::load((const Type *)ptr++);
            tsum.simd_ = Space::add(tsum.simd_, tmp.simd_);
        } while(ptr < (const VType *)eptr);
    } else {
        do {
            tmp.simd_ = Space::loadu((const Type *)ptr++);
            tsum.simd_ = Space::add(tsum.simd_, tmp.simd_);
        } while(ptr < (const VType *)eptr);
    }
    Type ret = tmp.sum();
    auto lptr = (const Type *)ptr;
    while(lptr < eptr) ret += *lptr++;
    return ret;
}

template<typename Container, typename FloatType=double>
FloatType mean(const Container &c) {
    return c.size() ? static_cast<FloatType>(sum(c)) / c.size(): std::numeric_limits<FloatType>::quiet_NaN();
}


template<typename Container>
auto pearsonr(const Container &c1, const Container &c2) {
    using FType = std::decay_t<decltype(c1[0])>;
    static_assert(std::is_floating_point_v<FType>, "Containers must hold floating points.");
    if(c1.size() != c2.size())
        throw std::runtime_error("Wrong sizes. size1: "s + std::to_string(c1.size()) + ", " + std::to_string(c2.size()));
    using Space = vec::SIMDTypes<FType>;
    using VType = typename Space::VType;
    auto m1 = mean(c1), m2 = mean(c2);
    VType v1, v2;
    VType sum1sq(0), sum2sq(0), sumdot(0);
    const VType mb1 = m1, mb2 = m2;
    const VType *p1((const VType *)&c1[0]), *p2((const VType *)&c2[0]);
    if(Space::aligned(p1) && Space::aligned(p2)) {
        do { // aligned loads
            v1.simd_ = Space::sub(Space::load((const FType *)p1++), mb1.simd_);
            v2.simd_ = Space::sub(Space::load((const FType *)p2++), mb2.simd_);
            sum1sq = Space::add(sum1sq.simd_, Space::mul(v1.simd_, v1.simd_));
            sum2sq = Space::add(sum2sq.simd_, Space::mul(v2.simd_, v2.simd_));
            sumdot = Space::add(sumdot.simd_, Space::mul(v1.simd_, v2.simd_));
        } while(p1 < (const VType *)&c1[c1.size()]);
    } else { // unaligned loads
        do {
            v1.simd_ = Space::sub(Space::loadu((const FType *)p1++), mb1.simd_);
            v2.simd_ = Space::sub(Space::loadu((const FType *)p2++), mb2.simd_);
            sum1sq = Space::add(sum1sq.simd_, Space::mul(v1.simd_, v1.simd_));
            sum2sq = Space::add(sum2sq.simd_, Space::mul(v2.simd_, v2.simd_));
            sumdot = Space::add(sumdot.simd_, Space::mul(v1.simd_, v2.simd_));
        } while(p1 < (const VType *)&c1[c1.size()]);
    }
    auto sd = sumdot.sum();
    auto s1s = sum1sq.sum();
    auto s2s = sum2sq.sum();
    const FType *fp1 = (FType *)p1, *fp2 = (FType *)p2;
    while(fp1 < &*(std::cend(c1))) {
        auto v1 = (*fp1++) - m1;
        auto v2 = (*fp2++) - m2;
        sd += v1 * v2;
        s1s += v1 * v1;
        s2s += v2 * v2;
    }
    auto rden = std::sqrt(s1s) * std::sqrt(s2s); // two square roots for better floating-point accuracy.
    return rden ? std::min(std::max(sd / rden, FType(-1.)), FType(1.)): std::numeric_limits<FType>::quiet_NAN();
}


} // stats
#endif
