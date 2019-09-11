#ifndef VAC_SKETCH_H__
#define VAC_SKETCH_H__
#include "mult.h"
#include "hll.h"
#include "fixed_vector.h"
#include "aesctr/wy.h"

namespace sketch {

namespace vac {

// For seeding each thread's generator separately
template<typename RNG>
struct ThreadSeededGen: public RNG {
    template<typename...Args>
    ThreadSeededGen(Args &&...args): RNG(std::forward<Args>(args)...) {
        this->seed(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) {return RNG::operator()(std::forward<Args>(args)...);}
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) const {return RNG::operator()(std::forward<Args>(args)...);}
};


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
    VACSketch(size_t n, Args &&...args): n_(n) {
        if(n <= 1) {
            std::fputs((std::string(__PRETTY_FUNCTION__) + " requires n >= 2. Provided: " + std::to_string(n)).data(), stderr);
            n_ = 1;
        }
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
        for(auto i1 = sketches_.begin(), i2 = o.sketches_.begin(), e1 = sketches_.end();
            i1 != e1; *i1++ += *i2++);
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
    detail::alloca_wrap<double> mem(n);
    auto p = mem.get();
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
        lut_(construct_power_table(base, n)), base_(base),
        super(n, std::forward<Args>(args)...)
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
        if(this->n_ != this->o.n_) throw std::runtime_error("Mismatched vacsketch counts");
        if(base_ != o.base_) throw std::runtime_error("Mismatched vacsketch base parameter counts");
        for(auto i1 = this->sketches_.begin(), i2 = o.sketches_.begin(), e1 = this->sketches_.end();
            i1 != e1; *i1++ += *i2++);
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
