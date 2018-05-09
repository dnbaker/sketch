#ifndef FILTER_HLL_H__
#define FILTER_HLL_H__
#include "common.h"
#include "hll.h"
#include "cbf.h"

namespace fhll {

using namespace ::common;

template<typename HashType=WangHash>
class fhllbase_t {
    using cbf_t = bf::cbfbase_t<HashType>;
    using hll_t = hll::hllbase_t<HashType>;
    cbf_t cbf_;
    hll_t hll_;
    unsigned threshold_;
public:
    fhllbase_t(unsigned np_, size_t nbfs, size_t l2sz, unsigned nhashes, uint64_t seedseedseedval,
               unsigned threshold, hll::EstimationMethod estim=hll::ERTL_MLE, hll::JointEstimationMethod jestim=hll::ERTL_JOINT_MLE, bool clamp=true):
        cbf_(nbfs, l2sz, nhashes, seedseedseedval), hll_(np_, estim, jestim, -1, clamp), threshold_(threshold) {
        if(threshold > (1u << (nbfs - 1))) throw std::runtime_error("Count threshold must be countable-to");
    }
    void addh(uint64_t val) {
        cbf_.addh(val); // This wastes one check in bf1. TODO: elide this.
        if(cbf_.est_count(val) >= threshold_) hll_.addh(val);
    }
    void addh(VType val) {
        cbf_.addh(val); // This wastes one check in bf1. TODO: elide this.
        val.for_each([&](uint64_t val){if(cbf_.est_count(val) >= threshold_) hll_.addh(val);});
    }
    void clear() {
        hll_.clear();
        cbf_.clear();
    }
    void set_threshold(unsigned threshold) {threshold_ = threshold;}
    void resize_bloom(unsigned newnp) {
        cbf_.resize_sketches(newnp);
    }
    auto threshold() const {return threshold_;}
    void not_ready() {hll_.not_ready();}
    hll_t       &hll()       {return hll_;}
    const hll_t &hll() const {return hll_;}
    void free_cbf() {cbf_.free();}
    void free_hll() {hll_.free();}
    fhllbase_t clone(uint64_t seed=0) const {
        return fhllbase_t(hll_.p(), cbf_.size(), cbf_.filter_size(), cbf_.nhashes(), seed ? seed: ((uint64_t)std::rand() << 32) | std::rand(), threshold_, hll_.get_estim(), hll_.get_jestim(), hll_.clamp());
    }
};
using fhll_t = fhllbase_t<>;
template<typename HashType=WangHash>
using filterhll_t = fhllbase_t<HashType>;

} // namespace fhll

namespace cbf {

using namespace ::common;

template<typename HashStruct=WangHash, typename RngType=aes::AesCtr<uint64_t, 8>>
class pcbfbase_t {
    // Probabilistic bloom filter counting.
protected:
    using bf_t  = bf::bfbase_t<HashStruct>;
    using hll_t = hll::seedhllbase_t<HashStruct>;

    std::vector<hll_t> hlls_;
    std::vector<bf_t>   bfs_;
    RngType             rng_;
    uint64_t            gen_;
    uint8_t           nbits_;
public:
    explicit pcbfbase_t(size_t nbfs, size_t l2sz, unsigned nhashes,
                        uint64_t seedseedseedval, unsigned hllp=0, hll::EstimationMethod estim=hll::ERTL_MLE,
                        hll::JointEstimationMethod jestim=hll::ERTL_JOINT_MLE, bool shrinkpow2=false):
        rng_{seedseedseedval}, gen_(rng_()), nbits_(64)
    {
        bfs_.reserve(nbfs);
        hlls_.reserve(nbfs);
        while(bfs_.size() < nbfs) bfs_.emplace_back(l2sz, nhashes, rng_());
        if(shrinkpow2) for(int hllstart = hllp ? hllp: l2sz > 12 ? l2sz - 4: 8; hlls_.size() < nbfs; hlls_.emplace_back(rng_(), std::max(hllstart--, static_cast<int>(hll::hllbase_t<HashStruct>::min_size())), estim, jestim, 1, false));
        else while(hlls_.size() < nbfs) hlls_.emplace_back(rng_(), hllp ? hllp: l2sz > 12 ? l2sz - 4: 8, estim, jestim, 1, false);
    }
    const std::vector<bf_t>  &bfs()  const {return bfs_;}
    const std::vector<hll_t> &hlls() const {return hlls_;}
    void resize_bloom(unsigned newsize) {
        for(auto &bf: bfs_) bf.resize(newsize);
    }
    size_t size() const {return bfs_.size();}
    INLINE void addh(uint64_t val) {
        unsigned i(0);
        if(!bfs_[i].may_contain(val) || !hlls_[i].may_contain(val)) {
            bfs_[i].addh(val);
            hlls_[i].addh(val);
            return;
        }
        while(++i != bfs_.size() && bfs_[i].may_contain(val) && hlls_[i].may_contain(val));
        if(i == bfs_.size()) return;
        if(__builtin_expect(nbits_ < i, 0)) gen_ = rng_(), nbits_ = 64;
        if((gen_ & (UINT64_C(-1) >> (64 - i))) == 0) bfs_[i].addh(val), hlls_[i].addh(val);
        gen_ >>= i, nbits_ -= i;
    }
    INLINE void addh(VType val) {
        val.for_each([&](uint64_t val) {this->addh(val);}); // Could be further accelerated with SIMD. I'm including this for interface compatibility.
    }
    bool may_contain(uint64_t val) const {
        for(unsigned i(0); i < bfs_.size(); ++i)
            if(!bfs_[i].may_contain(val) || !hlls_[i].may_contain) return false;
        return true;
    }
    void clear() {
        for(auto &h: hlls_) h.clear();
        for(auto &b: bfs_) b.clear();
        gen_ = nbits_ = 0;
    }
    unsigned naive_est_count(uint64_t val) const {
        unsigned i(0);
        if(!bfs_[i].may_contain(val) || !hlls_[i].may_contain(val)) return 0;
        while(++i != bfs_.size() && bfs_[i].may_contain(val) && hlls_[i].may_contain(val));
        return 1u << (i - 1);
    }
    unsigned est_count(uint64_t val) const {
        // TODO: better estimate using the hlls to provide error rate estimates.
        return naive_est_count(val);
    }
};

using pcbf_t = pcbfbase_t<hll::WangHash>;
template<typename HashType=hll::WangHash>
class pcbfhllbase_t {
    using hll_t = hll::hllbase_t<HashType>;
    pcbfbase_t<HashType>    pcb_;
    hll_t                   hll_;
    unsigned          threshold_;
    uint64_t    seedseedseedval_;
public:
    pcbfhllbase_t(unsigned filternp_, unsigned subnp_, size_t nbfs, size_t l2sz, unsigned nhashes, uint64_t seedseedseedval,
                  unsigned threshold, hll::EstimationMethod estim=hll::ERTL_MLE, hll::JointEstimationMethod jestim=hll::ERTL_JOINT_MLE, bool clamp=true):
            pcb_(nbfs, l2sz, nhashes, seedseedseedval, subnp_, estim, jestim, clamp),
            hll_(filternp_, estim, jestim, -1, clamp), threshold_{threshold}, seedseedseedval_(seedseedseedval)
    {
        if(threshold > (1u << (pcb_.size() - 1))) throw std::runtime_error("Count threshold must be countable-to");
    }
    void addh(uint64_t val) {
        pcb_.addh(val); // This wastes a check. TODO: elide this.
        if(pcb_.est_count(val) >= threshold_) hll_.addh(val);
    }
    void addh(VType val) {
        pcb_.addh(val); // This wastes a check. TODO: elide this.
        val.for_each([&](uint64_t val){if(pcb_.est_count(val) >= threshold_) hll_.addh(val);});
    }
    void clear() {
        hll_.clear();
        pcb_.clear();
    }
    void resize_bloom(unsigned newsize) {
        pcb_.resize_bloom(newsize);
    }
    void set_threshold(unsigned threshold) {threshold_ = threshold;}
    auto threshold() const {return threshold_;}
    void not_ready() {hll_.not_ready();}
    hll_t       &hll()       {return hll_;}
    const hll_t &hll() const {return hll_;}
    pcbfhllbase_t clone() const {
        return pcbfhllbase_t(hll_.p(), pcb_.hlls()[0].p(), pcb_.size(), std::log2(pcb_.bfs()[0].size()),
                             pcb_.bfs()[0].nhashes(), seedseedseedval_, threshold_, pcb_.hlls()[0].get_estim(), pcb_.hlls()[0].get_jestim(), pcb_.hlls()[0].clamp());
    }
};
using pcfhll_t = pcbfhllbase_t<hll::WangHash>;

} // namespace cbf

#endif // #ifndef FILTER_HLL_H__
