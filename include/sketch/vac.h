#ifndef VAC_SKETCH_H__
#define VAC_SKETCH_H__
#include "./mult.h"
#include "./hll.h"
#include "./fixed_vector.h"
#include "aesctr/wy.h"
#include "./exception"
#include "./tsg.h"

namespace sketch {

namespace vac {

using tsg::ThreadSeededGen;

template<typename BaseSketch,
         template<typename...> class Container=std::vector,
         typename RNG=wy::WyHash<uint64_t, 8>,
         typename...VectorArgs>
struct VACSketch {
    using base = BaseSketch;

    // Members
    Container<base, VectorArgs...> sketches_;
    const unsigned n_;

    // Construction
    template<typename...Args>
    VACSketch(size_t n, Args &&...args): n_(n > 1? n: size_t(1)) {
        if(n <= 1)
            std::fputs((std::string(__PRETTY_FUNCTION__) + " requires n >= 2. Provided: " + std::to_string(n)).data(), stderr);

        sketches_.reserve(n);
        for(size_t i = n; i--; sketches_.emplace_back(std::forward<Args>(args)...));
    }
    // Addition
    void addh(uint64_t x) {
        thread_local static ThreadSeededGen<RNG> gen;
        const auto end = std::min(ctz(gen()) + 1, n_);
        unsigned i = 0;
        do sketches_[i++].addh(x); while(i < end);
    }
    // Composition
    VACSketch &operator+=(const VACSketch &o) {
        if(n_ != o.n_) throw std::runtime_error("Mismatched vacsketch counts");
        auto i1 = sketches_.begin();
        auto i2 = o.sketches_.begin();
        auto e1 = sketches_.end();
        while(i1 != e1) (*i1++ += *i2++);
        return *this;
    }
    VACSketch operator+(const VACSketch &o) const {
        auto tmp = *this;
        tmp += o;
        return tmp;
    }
};

static fixed::vector<uint64_t> construct_power_table(double base, size_t n) {
    if(base <= 1.) throw std::runtime_error(std::to_string(base) + " is forbidden. Must be > 1.");
    fixed::vector<uint64_t> ret(n - 1);
    std::vector<double> mem(n);
    auto p = mem.data();
    p[0] = 1.;
    for(size_t i = 1; i < n; ++i) {
        auto tmp = base * p[i];
        ret[i] = std::numeric_limits<uint64_t>::max() / tmp;
        p[i + 1] = tmp;
    }
    return ret;
}

template<typename BaseSketch,
         template<typename...> class Container=std::vector,
         typename RNG=wy::WyHash<uint64_t, 8>,
         typename...VectorArgs>
struct PowerVACSketch: public VACSketch<BaseSketch, Container, RNG, VectorArgs...> {
    using super = VACSketch<BaseSketch, Container, RNG, VectorArgs...>;

    const fixed::vector<uint64_t> lut_;
    const double base_;

    template<typename...Args>
    PowerVACSketch(double base, size_t n, Args &&... args):
        super(n, std::forward<Args>(args)...),
        lut_(construct_power_table(base, n)),
        base_(base)
    {
        std::fprintf(stderr, "base: %f. n: %zu\n", base, n);
    }
    // Addition
    void addh(uint64_t x) {
        thread_local static ThreadSeededGen<RNG> gen;
        auto v = gen();
        unsigned i = 0;
        do {
            this->sketches_[i++].addh(x);
        } while(i < this->n_ && v < lut_[i]);
    }
    // Composition
    PowerVACSketch &operator+=(const PowerVACSketch &o) {
        PREC_REQ(this->n_ == o.n_, "Must be same n");
        PREC_REQ(this->base_ == o.base_, "Must be same base");
        auto i1 = this->sketches_.begin();
        auto i2 = o.sketches_.begin();
        auto e1 = this->sketches_.end();
        while(i1 != e1) (*i1++ += *i2++);
        return *this;
    }
    PowerVACSketch operator+(const PowerVACSketch &o) const {
        auto tmp = *this;
        tmp += o;
        return tmp;
    }
};

using HVAC = VACSketch<hll::hll_t>;
using PowerHVAC = PowerVACSketch<hll::hll_t>;

} // vac
using namespace vac;

}

#endif /* VAC_SKETCH_H__ */
