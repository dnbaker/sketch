#ifndef DNB_SKETCH_MULTIPLICITY_H__
#define DNB_SKETCH_MULTIPLICITY_H__

#include "blaze/Math.h"
#include <random>
#include "ccm.h" // Count-min sketch

namespace sketch {
using namespace common;

namespace cws {

template<typename FType=float>
struct CWSamples {
    using MType = blaze::DynamicMatrix<float>;
    MType r_, c_, b_;
    CWSamples(size_t nhist, size_t histsz, uint64_t seed=0xB0BAFe77C001D00D): r_(nhist, histsz), c_(nhist, histsz), b_(nhist, histsz) {
        std::mt19937_64 mt(seed);
        std::gamma_distribution<FType> dist(2, 1);
        std::uniform_real_distribution<FType> rdist;
        for(size_t i = 0; i < nhist; ++i) {
            auto rr = row(r_, i);
            auto cr = row(c_, i);
            auto br = row(b_, i);
            for(size_t j = 0; j < histsz; ++j)
                rr[j] = dist(mt), cr[j] = dist(mt), br[j] = rdist(mt);
        }
    }
};

template<typename FType=float, typename HashStruct=common::WangHash>
class realccm_t: public cm::ccmbase_t<cm::update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,false> {
    using super = cm::ccmbase_t<cm::update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,false>;
    using FSpace = vec::SIMDTypes<FType>;
    FType scale_;
public:
    FType decay_rate() const {return scale_;}
    void addh(uint64_t val) {this->add(val);}
    void add(uint64_t val) {
        if(scale_) {
            rescale();
        }
        super::add(val);
    }
    template<typename...Args>
    realccm_t(FType scale_prod=0., Args &&...args): scale_(scale_prod), super(std::forward<Args>(args)...) {
        assert(scale_ >= 0. && scale_ <= 1.);
    }
    void rescale() {
        auto ptr = reinterpret_cast<typename FSpace::VType *>(this->data_.data());
        auto eptr = reinterpret_cast<typename FSpace::VType *>(this->data_.data() + this->data_.size());
        auto mul = FSpace::set1(scale_);
        while(eptr > ptr) {
            *ptr = Space::mul(ptr->simd_, mul);
            ++ptr;
        }
        FType *rptr = reinterpret_cast<FType *>(ptr);
        while(rptr < this->data_.data() + this->data_.size())
            *rptr++ *= scale_;
    }
};

} // namespace cws
} // namespace sketch

#endif /* DNB_SKETCH_MULTIPLICITY_H__ */
