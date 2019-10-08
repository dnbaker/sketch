#ifndef SKETCH_POLICY_H
#define SKETCH_POLICY_H
#include "integral.h"
#include "div.h"
namespace sketch {
namespace policy {

template<typename T>
struct SizePow2Policy {
    T mask_;
    T shift_;
    SizePow2Policy(size_t n): mask_((1ull << nelem2arg(n)) - 1), shift_(ilog2(mask_ + 1)) {
        auto n2a = nelem2arg(n);
        std::fprintf(stderr, "n (%zu) becomes argument %zu\n", n, n2a);
        mask_ = (1ull << nelem2arg(n));
        shift_ = ilog2(mask_);
        --mask_;
    }
    static size_t nelem2arg(size_t nelem) {
        // Return the floor of nelem, but increment by one if it wasn't a power of two.
        return ilog2(nelem) + ((nelem & (nelem - 1)) != 0);
    }
    size_t nelem() const {return size_t(mask_) + 1;}
    static size_t arg2vecsize(size_t arg) {return size_t(1) << nelem2arg(arg);}
    T mod(T rv) const {
        return rv & mask_;
    }
    T div(T rv) const {
        return rv >> shift_;
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
    T div(T rv) const {return div_.div(rv);}
    SizeDivPolicy(T div): div_(div) {}
};

} // policy
} // sketch

#endif /* SKETCH_POLICY_H */
