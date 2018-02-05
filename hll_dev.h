#ifndef HLL_DEV_H__
#define HLL_DEV_H__

#include "hll.h"
#include <random>

namespace hll {

class hlldub_t: public hll_t {
    // hlldub_t inserts each value twice (forward and reverse)
    // and simply halves cardinality estimates.
public:
    template<typename... Args>
    hlldub_t(Args &&...args): hll_t(std::forward<Args>(args)...) {}
    INLINE void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (p() - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (p() - 1)) + 1);
        core_[index] = std::min(core_[index], lzt);
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
        uint64_t fcounts[64]{0};
        uint64_t rcounts[64]{0};
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
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (p() - 1)) + 1);
            dcore_[index] < lzt;
            __sync_bool_compare_and_swap(dcore_.data() + index, dcore_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (p() - 1)) + 1);
        dcore_[index] = std::min(dcore_[index], lzt);
#endif
    }
    void addh(uint64_t element) {add(wang_hash(element));}
    bool may_contain(uint64_t hashval) {
        return hll_t::may_contain(hashval) && dcore_[hashval & ((m()) - 1)] >= ctz(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (p() - 1)) + 1;
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
    uint64_t seed() const {return seed_;}

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

namespace detail {
inline std::vector<uint64_t> seeds_from_seed(uint64_t seed, size_t size) {
    LOG_DEBUG("Initializing a vector of seeds of size %zu with a seed-seed of %" PRIu64 "\n", size, seed);
    std::mt19937_64 mt(seed);
    std::vector<uint64_t> ret;
    while(ret.size() < size) ret.emplace_back(mt());
    return ret;
}
}

class hlf_t {
protected:
    // Consider templating this to extend to hlldub_ts as well.
    std::vector<seedhll_t> hlls_;
public:
    template<typename SeedContainer, typename... Args>
    hlf_t(const SeedContainer &con, Args &&... args) {
        if(std::size(con) == 0) throw std::runtime_error("%s requires are least a size of 1.\n", __func__);
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
    template<typename SeedContainer, typename... Args>
    hlf_t(size_t size, uint64_t seedseed, Args &&... args): hlf_t(detail::seeds_from_seed(seedseed, size), std::forward<Args>(args)...) {}
    auto size() const {return hlls_.size();}
    auto m() const {return hlls_[0].size();}

    // This only works for hlls using 64-bit integers.
    // Looking ahead,consider templating so that the double version might be helpful.

    auto may_contain(uint64_t element) const {
#pragma message("Note: may_contain only works for the HyperLogFilter in the case of 64-bit integer insertions. One must hash a string to a 64-bit integer first in order to use it for this purpose.")
        return std::accumulate(hlls_.begin() + 1, hlls_.end(), hlls_.front().may_contain(wang_hash(element ^ hlls_.front().seed())),
                               [element](auto a, auto b) {
            return a && b.may_contain(wang_hash(element ^ b.seed()));
        });
    }
    void add(uint64_t val) {
        for(auto &hll: hlls_) hll.addh(val);
    }
    double creport() const {
        double ret(hlls_[0].creport());
        for(size_t i(1); i < size(); ret += hlls_[i++].creport());
        ret /= static_cast<double>(size());
        return ret;
    }
    double report() noexcept {
        hlls_[0].sum();
        double ret(hlls_[0].report());
        for(size_t i(1); i < size(); ++i)
            hlls_[i].sum(), ret += hlls_[i].report();
        ret /= static_cast<double>(size());
        return ret;
    }
};

} // namespace hll

#endif // #ifndef HLL_DEV_H__
