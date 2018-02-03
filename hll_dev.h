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

static inline uint64_t finalize(uint64_t h) {
    // Murmurhash3 finalizer, for multiplying hash functions for seedhll_t and hllfilter_t.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
};

class seedhll_t: public hll_t {
protected:
    const uint64_t seed_; // 64-bit integers are xored with this value before passing it to a hash.
                          // This is almost free, in the content of 
public:
    template<typename... Args>
    seedhll_t(uint64_t seed, Args &&...args): hll_t(std::forward<Args>(args)...), seed_(seed) {
        if(seed_ == 0) LOG_WARNING("Note: seed is set to 0. No more than one of these at a time should have this value, and this is only for the purpose of multiplying hashes."
                                   " Also, if you are only using one of these at a time, don't use seedhll_t, just use hll_t and save yourself an xor per insertion"
                                   ", not to mention a 64-bit integer in space.");
    }
    void addh(uint64_t element) {
        element ^= seed_;
        add(wang_hash(element));
    }
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(element))>, uint64_t>, "Must return 64-bit hash");
        add(finalize(hasher(element) ^ seed_));
    }
#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(s, len))>, uint64_t>, "Must return 64-bit hash");
        add(finalize(hasher(s, len) ^ seed_));
    }
#endif
};

class hllfilter_t {
protected:
    // Consider templating this to extend to hlldub_ts as well.
    std::vector<seedhll_t> hlls_;
public:
    template<typename SeedContainer, typename... Args>
    hllfilter_t(const SeedContainer &con, Args &&... args) {
        hlls_.reserve(std::size(con));
        using SeedType = std::decay_t<decltype(*std::begin(con))>;
        static_assert(std::is_integral_v<SeedType>, "seeds must be integers....");
        std::vector<SeedType> seedset;
        for(const auto seed: con) {
            hlls_.emplace_back(seed, std::forward<Args>(args)...);
            if(std::find(std::begin(seedset), std::end(seedset), seed) != std::end(seedset))
                throw std::runtime_error("Error: hllfilter_t requires distinct seeds for each subhll. Otherwise you don't improve your power.");
            seedset.push_back(seed);
        }
        if(seedset.size() != std::size(con)) throw std::runtime_error("Error: hllfilter_t requires distinct seeds for each subhll. Otherwise you don't improve your power.");
    }
    auto size() const {return hlls_.size();}
    auto m() const {return hlls_[0].size();}
#if 0

    This will require more work, as we cannot just check for hashvals. We have to allow each hll to perform its hash, and then
    test for membership. Look ahead, considering making it so that the double version is helpful.

    auto may_contain(uint64_t element) {
        return std::accumulate(hlls_.begin(), hlls_.end(), true, [hashval](auto a, auto b) {return a && b.may_contain(hashval);});
    }
#endif
};

} // namespace hll

#endif // #ifndef HLL_DEV_H__
