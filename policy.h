#ifndef SKETCH_POLICY_H
#define SKETCH_POLICY_H
#include "integral.h"
namespace sketch {
namespace policy {

template<typename T>
struct SizePow2Policy {
    T mask_;
    SizePow2Policy(size_t n): mask_((1ull << nelem2arg(n)) - 1) {
    }
    static size_t nelem2arg(size_t nelem) {
        return ilog2(roundup(nelem));
    }
    size_t nelem() const {return size_t(mask_) + 1;}
    static size_t arg2vecsize(size_t arg) {return size_t(1) << nelem2arg(arg);}
    T mod(T rv) const {
        return rv & mask_;
    }
};

template<typename T>
struct SizeDivPolicy {
    schism::Schismatic<T> div_;
    static size_t nelem2arg(size_t nelem) {
        return nelem;
    }
    size_t nelem() const {return div_.d();}
    static size_t arg2vecsize(size_t arg) {return arg;}
    T mod(T rv) const {return div_.mod(rv);}
    SizeDivPolicy(T div): div_(div) {}
};

} // policy
} // sketch

#endif /* SKETCH_POLICY_H */
