#ifndef HLL_DEV_H__
#define HLL_DEV_H__

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
        value_  = calculate_estimate(fcounts, use_ertl_, m(), np_, alpha());
        value_ += calculate_estimate(rcounts, use_ertl_, m(), np_, alpha());
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

union SIMDHolder {
public:
#if HAS_AVX_512
    using SType = __m512i;
#  define MAX_FN(x, y) _mm512_max_epu8(x, y)
#elif __AVX2__
    using SType = __m256i;
#  define MAX_FN(x, y) _mm256_max_epu8(x, y)
#elif __SSE2__
    using SType = __m128i;
#  define MAX_FN(x, y) _mm_max_epu8(x, y)
#else
#  error("Need at least SSE2")
#endif
    static constexpr size_t nels = sizeof(SType) / sizeof(char);
    SType val;
    uint8_t vals[nels];
};

static inline double union_size(const hll_t &h1, const hll_t &h2) {
    assert(h1.m() == h2.m());
    using SType = typename SIMDHolder::SType;
    uint32_t counts[65];
    const SType *p1((const SType *)(h1.data())), *p2((const SType *)(h2.data()));
    const SType *pend(reinterpret_cast<const SType *>(h1.data() + h1.m()));
    SIMDHolder tmp;
    tmp.val = MAX_FN(*p1++, *p2++);
    for(const auto el: tmp.vals) ++counts[el];
    while(p1 < pend) {
        tmp.val = MAX_FN(*p1++, *p2++);
        for(const auto el: tmp.vals) ++counts[el];
    }
    return calculate_estimate(counts, h1.get_use_ertl(), h1.m(), h1.p(), h1.alpha());
}

#undef MAX_FN

#endif // #ifndef HLL_DEV_H__
