#ifndef VAC_SKETCH_H__
#define VAC_SKETCH_H__
#include "mult.h"
#include "hll.h"
#include "aesctr/wy.h"

namespace sketch {

namespace vac {

template<typename BaseSketch,
         template<typename...> class Container=std::vector,
         typename RNG=wy::WyHash<uint64_t, 8>,
         typename...VectorArgs>
struct VACSketch {
    using base = BaseSketch;

    // Members
    Container<base, VectorArgs...> sketches_;
    const unsigned n_;

    // For seeding each thread's generator separately
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

    // Construction
    template<typename...Args>
    VACSketch(size_t n, Args &&...args): n_(n) {
        if(n <= 1) throw std::runtime_error(std::string(__PRETTY_FUNCTION__) + " requires n >= 2");
        sketches_.reserve(n);
        for(size_t i = n; i--; sketches_.emplace_back(std::forward<Args>(args)...));
    }
    // Addition
    void addh(uint64_t x) {
        thread_local ThreadSeededGen gen;
        const auto end = std::min(ctz(~gen()) + 1, n_);
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

using HVAC = VACSketch<hll::hll_t>;

} // vac
using namespace vac;

}

#endif /* VAC_SKETCH_H__ */
