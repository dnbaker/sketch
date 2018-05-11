#ifndef FILTER_HLL_H__
#define FILTER_HLL_H__
#include "common.h"
#include "hll.h"
#include "cbf.h"

namespace sketch {
namespace fhll {

using namespace common;

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


template<typename HashType=hll::WangHash>
class pcbfhllbase_t {
    using hll_t = hll::hllbase_t<HashType>;
    bf::pcbfbase_t<HashType>    pcb_;
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

} // namespace fhll
} // namespace sketch

#endif // #ifndef FILTER_HLL_H__
