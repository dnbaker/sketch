#ifndef HLL_DEV_H__
#define HLL_DEV_H__

namespace hll {

#ifndef HLL_H_
#  error("Please include hll.h first. Defining -DENABLE_HLL_DEVELOP is recommended instead, but so long as you include this after, no harm is done.")
#endif

class hlldub_t: public hll_t {
    // hlldub_t inserts each value twice (forward and reverse)
    // and simply halves cardinality estimates.
public:
    template<typename... Args>
    hlldub_t(Args &&...args): hll_t(std::forward<Args>(args)...) {}
    INLINE void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(hashval >> p()) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(hashval >> p()) + 1);
        if(core_[index] < lzt) core_[index] = lzt;
#endif
    }
    double report() {
        sum();
        return this->creport();
    }
    double creport() {
        return hll_t::creport() * 0.5;
    }
    bool may_contain(uint64_t hashval) {
        return hll_t::may_contain(hashval) && core_[hashval & ((m()) - 1)] >= ctz(hashval >> p()) + 1;
    }

    INLINE void addh(uint64_t element) {add(wang_hash(element));}
};

class dhll_t: public hll_t {
    // dhll_t is a bidirectional hll sketch which does not currently support set operations
    // It is based on the idea that the properties of a hll sketch work for both leading and trailing zeros and uses them as independent samples.
    std::vector<uint8_t, Allocator> dcore_;
public:
    template<typename... Args>
    dhll_t(Args &&...args): hll_t(std::forward<Args>(args)...),
                            dcore_(1ull << hll_t::p()) {
    }
    void sum() {
        uint32_t fcounts[65]{0};
        uint32_t rcounts[65]{0};
        const auto &core(hll_t::core());
        for(size_t i(0); i < core.size(); ++i) {
            // I don't this can be unrolled and LUT'd.
            ++fcounts[core[i]]; ++rcounts[dcore_[i]];
        }
        value_  = detail::calculate_estimate(fcounts, use_ertl_, m(), np_, alpha());
        value_ += detail::calculate_estimate(rcounts, use_ertl_, m(), np_, alpha());
        value_ *= 0.5;
        is_calculated_ = 1;
    }
    void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(hashval >> p()) + 1);
            dcore_[index] < lzt;
            __sync_bool_compare_and_swap(dcore_.data() + index, dcore_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(hashval >> p()) + 1);
        if(dcore_[index] < lzt) dcore_[index] = lzt;
#endif
    }
    void addh(uint64_t element) {add(wang_hash(element));}
    bool may_contain(uint64_t hashval) {
        return hll_t::may_contain(hashval) && dcore_[hashval & ((m()) - 1)] >= ctz(hashval >> p()) + 1;
    }
};


} // namespace hll

#endif // #ifndef HLL_DEV_H__
