#ifndef FILTER_HLL_H__
#define FILTER_HLL_H__
#include "common.h"
#include "hll.h"
#include "cbf.h"

namespace sketch {
inline namespace fhll {

template<typename HashType=WangHash>
class fhllbase_t {
    using cbf_t = bf::cbfbase_t<HashType>;
    using hll_t = hll::hllbase_t<HashType>;
    cbf_t cbf_;
    hll_t hll_;
    unsigned threshold_;
public:
    fhllbase_t(unsigned np_, size_t nbfs, size_t l2sz, unsigned nhashes, uint64_t seedseedseedval,
               unsigned threshold, hll::EstimationMethod estim=hll::ERTL_MLE, hll::JointEstimationMethod jestim=hll::ERTL_JOINT_MLE):
        cbf_(nbfs, l2sz, nhashes, seedseedseedval), hll_(np_, estim, jestim), threshold_(threshold) {
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
    void reseed(uint64_t seed) {
        cbf_.reseed(seed);
    }
    fhllbase_t(const fhllbase_t&) = default;
    fhllbase_t clone(uint64_t seed=0) const {
        auto ret = fhllbase_t(*this);
        ret.clear();
        ret.reseed(seed ? seed: (uint64_t(std::rand()) << 32) | std::rand());
    }
};
using fhll_t = fhllbase_t<>;


template<typename HashType=hll::WangHash>
class pcbfhllbase_t {
    using hll_t = hll::hllbase_t<HashType>;
    bf::pcbfbase_t<HashType> pcb_;
    hll_t                    hll_;
    unsigned           threshold_;
    uint64_t     seedseedseedval_;
public:
    pcbfhllbase_t(unsigned filternp_, unsigned subnp_, size_t nbfs, size_t l2sz, unsigned nhashes, uint64_t seedseedseedval,
                  unsigned threshold, hll::EstimationMethod estim=hll::ERTL_MLE, hll::JointEstimationMethod jestim=hll::ERTL_JOINT_MLE, bool shrinkpow2=true):
            pcb_(nbfs, l2sz, nhashes, seedseedseedval, subnp_, estim, jestim, shrinkpow2),
            hll_(filternp_, estim, jestim), threshold_{threshold}, seedseedseedval_(seedseedseedval)
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
    void reseed(uint64_t newseed) {
        seedseedseedval_ = newseed;
        pcb_.reseed(newseed);
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
    pcbfhllbase_t(const pcbfhllbase_t &other) = default;
    pcbfhllbase_t clone(uint64_t seed=0) const {
        auto ret = pcbfhllbase_t(*this);
        ret.clear();
        ret.reseed(seed ? seed: (uint64_t(std::rand())<<32)|std::rand());
        return ret;
    }
    void free_filters() {
        pcb_.free();
    }
};
using pcfhll_t = pcbfhllbase_t<hll::WangHash>;

} // inline namespace fhll
} // namespace sketch

#endif // #ifndef FILTER_HLL_H__
