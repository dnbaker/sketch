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

template<typename FType=float, typename HashStruct=common::WangHash, bool decay=false, bool conservative=false>
class realccm_t: public cm::ccmbase_t<cm::update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,conservative> {
    using super = cm::ccmbase_t<cm::update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,conservative>;
    using FSpace = vec::SIMDTypes<FType>;
    FType scale_;
public:
    FType decay_rate() const {return scale_;}
    void addh(uint64_t val) {this->add(val);}
    void add(uint64_t val) {
        CONST_IF(decay)
            rescale();
        super::add(val);
    }
    template<typename...Args>
    realccm_t(FType scale_prod=1., Args &&...args): scale_(scale_prod), super(std::forward<Args>(args)...) {
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
namespace nt {
template<typename Container=std::vector<uint32_t, Allocator<uint32_t>>, typename HashStruct=WangHash, bool filter=true>
struct Card {
    // Ref: https://www.ncbi.nlm.nih.gov/pubmed/28453674
    Container core_;
    const uint16_t p_, r_, pshift_;
    HashStruct hf_;
    template<typename...Args>
    Card(unsigned r, unsigned p, Args &&... args):
        core_(std::forward<Args>(args)...), p_(p), r_(r), pshift_(64 - p) {}
    void addh(uint64_t v) {
        v = hf_(v);
        add(v);
    }
    void add(uint64_t v) {
        CONST_IF(filter) {
            if(v >> pshift_)
                return;
        }
        v <<= (64 - r_);
        v >>= (64 - r_);
#ifndef NOT_THREADSAFE
        __sync_fetch_and_add(&core_[v], 1);
#else
        ++core_[v];
#endif
    }
};
template<typename CType, typename HashStruct=WangHash, bool filter=true>
struct VecCard: public Card<std::vector<CType, Allocator<CType>>, HashStruct, filter> {
    using super = Card<std::vector<CType, Allocator<CType>>, HashStruct, filter>;
    static_assert(std::is_integral<CType>::value, "Must be integral.");
    VecCard(unsigned p, unsigned r): super(p, r, 1ull << r) {}
};

} // namespace nt
} // namespace sketch

#endif /* DNB_SKETCH_MULTIPLICITY_H__ */
