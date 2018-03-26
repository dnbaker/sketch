#pragma once


namespace hll {

#ifdef ENABLE_HLL_DEVELOP
#  if !NDEBUG
#    pragma message("hll develop enabled (-DENABLE_HLL_DEVELOP)")
#  endif
#else
namespace dev {
#endif

template<typename HashFunc=WangHash>
class hlldub_base_t: public hllbase_t<HashFunc> {
    // hlldub_base_t inserts each value twice (forward and reverse)
    // and simply halves cardinality estimates.
public:
    template<typename... Args>
    hlldub_base_t(Args &&...args): hll_t(std::forward<Args>(args)...) {}
    INLINE void add(uint64_t hashval) {
        hllbase_t<HashFunc>::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((this->m()) - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
            this->core_[index] < lzt;
            __sync_bool_compare_and_swap(this->core_.data() + index, this->core_[index], lzt));
#else
        const uint32_t index(hashval & (this->m() - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
        this->core_[index] = std::min(this->core_[index], lzt);
#endif
    }
    double report() {
        this->sum();
        return this->creport();
    }
    double creport() const {
        return hllbase_t<HashFunc>::creport() * 0.5;
    }
    bool may_contain(uint64_t hashval) const {
        return hllbase_t<HashFunc>::may_contain(hashval) && this->core_[hashval & ((this->m()) - 1)] >= ffs(hashval >> this->p());
    }

    INLINE void addh(uint64_t element) {add(this->hf_(element));}
};
using hlldub_t = hlldub_base_t<>;

template<typename HashFunc=WangHash>
class dhllbase_t: public hllbase_t<HashFunc> {
    // dhllbase_t is a bidirectional hll sketch which does not currently support set operations
    // It is based on the idea that the properties of a hll sketch work for both leading and trailing zeros and uses them as independent samples.
    std::vector<uint8_t, Allocator<uint8_t>> dcore_;
    using hll_t = hllbase_t<HashFunc>;
public:
    template<typename... Args>
    dhllbase_t(Args &&...args): hll_t(std::forward<Args>(args)...),
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
        this->value_  = detail::calculate_estimate(fcounts, this->estim_, this->m(), this->np_, this->alpha());
        this->value_ += detail::calculate_estimate(rcounts, this->estim_, this->m(), this->np_, this->alpha());
        this->value_ *= 0.5;
        this->is_calculated_ = 1;
    }
    void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((this->m()) - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
            dcore_[index] < lzt;
            __sync_bool_compare_and_swap(dcore_.data() + index, dcore_[index], lzt));
#else
        const uint32_t index(hashval & (this->m() - 1)), lzt(ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1)));
        dcore_[index] = std::min(dcore_[index], lzt);
#endif
    }
    void addh(uint64_t element) {add(this->hf_(element));}
    bool may_contain(uint64_t hashval) const {
        return hll_t::may_contain(hashval) && dcore_[hashval & ((this->m()) - 1)] >= ffs(((hashval >> 1)|UINT64_C(0x8000000000000000)) >> (this->p() - 1));
    }
};
using dhll_t = dhllbase_t<>;


template<typename HashFunc=WangHash>
class seedhllbase_t: public hllbase_t<HashFunc> {
protected:
    uint64_t seed_; // 64-bit integers are xored with this value before passing it to a hash.
                          // This is almost free, in the content of
    using hll_t = hllbase_t<HashFunc>;
public:
    template<typename... Args>
    seedhllbase_t(uint64_t seed, Args &&...args): hll_t(std::forward<Args>(args)...), seed_(seed) {
        if(seed_ == 0) std::fprintf(stderr,
            "[W:%s:%d] Note: seed is set to 0. No more than one of these at a time should have this value, and this is only for the purpose of multiplying hashes."
            " Also, if you are only using one of these at a time, don't use seedhllbase_t, just use hll_t and save yourself an xor per insertion"
            ", not to mention a 64-bit integer in space.", __PRETTY_FUNCTION__, __LINE__);
    }
    seedhllbase_t(gzFile fp): hll_t() {
        this->read(fp);
    }
    void addh(uint64_t element) {
        element ^= seed_;
        this->add(wang_hash(element));
    }
    uint64_t seed() const {return seed_;}
    void write(const char *fn, bool write_gz) {
        if(write_gz) {
            gzFile fp = gzopen(fn, "wb");
            if(fp == nullptr) throw std::runtime_error("Could not open file.");
            this->write(fp);
            gzclose(fp);
        } else {
            std::FILE *fp = std::fopen(fn, "wb");
            if(fp == nullptr) throw std::runtime_error("Could not open file.");
            this->write(fileno(fp));
            std::fclose(fp);
        }
    }
    void write(gzFile fp) const {
        hll_t::write(fp);
        gzwrite(fp, &seed_, sizeof(seed_));
    }
    void read(gzFile fp) {
        hll_t::read(fp);
        gzread(fp, &seed_, sizeof(seed_));
    }
    void write(int fn) const {
        hll_t::write(fn);
        ::write(fn, &seed_, sizeof(seed_));
    }
    void read(int fn) {
        hll_t::read(fn);
        ::read(fn, &seed_, sizeof(seed_));
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        this->read(fp);
        gzclose(fp);
    }
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(element))>, uint64_t>, "Must return 64-bit hash");
        add(detail::finalize(hasher(element) ^ seed_));
    }

#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(s, len))>, uint64_t>, "Must return 64-bit hash");
        add(detail::finalize(hasher(s, len) ^ seed_));
    }
#endif
};
using seedhll_t = seedhllbase_t<>;

namespace sort {
// insertion_sort from https://github.com/orlp/pdqsort
// Slightly modified stylistically.
template<class Iter, class Compare>
inline void insertion_sort(Iter begin, Iter end, Compare comp) {
    using T = typename std::iterator_traits<Iter>::value_type;

    for (Iter cur = begin + 1; cur < end; ++cur) {
        Iter sift = cur;
        Iter sift_1 = cur - 1;

        // Compare first so we can avoid 2 moves for an element already positioned correctly.
        if (comp(*sift, *sift_1)) {
            T tmp = std::move(*sift);

            do { *sift-- = std::move(*sift_1); }
            while (sift != begin && comp(tmp, *--sift_1));

            *sift = std::move(tmp);
        }
    }
}
template<class Iter>
inline void insertion_sort(Iter begin, Iter end) {
    insertion_sort(begin, end, std::less<std::decay_t<decltype(*begin)>>());
}
} // namespace sort

template<typename SeedHllType=hll_t>
class hlfbase_t {
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<SeedHllType>                      hlls_;
    std::vector<uint64_t, Allocator<uint64_t>>   seeds_;
    std::vector<double>                         values_;
    mutable double                               value_;
    bool                                 is_calculated_;
    using HashType     = typename SeedHllType::HashType;
    const HashType                                  hf_;
public:
    template<typename... Args>
    hlfbase_t(size_t size, uint64_t seedseed, Args &&... args): value_(0), is_calculated_(0), hf_{} {
        auto sfs = detail::seeds_from_seed(seedseed, size);
        assert(sfs.size());
        hlls_.reserve(size);
        for(const auto seed: sfs)
            hlls_.emplace_back(std::forward<Args>(args)...), seeds_.emplace_back(seed);
    }
    hlfbase_t(const hlfbase_t &) = default;
    hlfbase_t(hlfbase_t &&) = default;
    uint64_t size() const {return hlls_.size();}
    auto m() const {return hlls_[0].size();}
    void write(const char *fn) const {
        gzFile fp = gzopen(fn, "wb");
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
        this->write(fp);
        gzclose(fp);
    }
    void clear() {
        value_ = is_calculated_ = 0;
        for(auto &hll: hlls_) hll.clear();
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
        this->read(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint64_t sz = hlls_.size();
        gzwrite(fp, &sz, sizeof(sz));
        for(auto &seed: seeds_) gzwrite(fp, &seed, sizeof(seed));
        for(const auto &hll: hlls_) {
            hll.write(fp);
        }
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t size;
        gzread(fp, &size, sizeof(size));
        seeds_.resize(size);
        for(unsigned i(0); i < size; gzread(fp, seeds_.data() + i++, sizeof(seeds_[0])));
        hlls_.clear();
        while(hlls_.size() < size) hlls_.emplace_back(fp);
        gzclose(fp);
    }

    // This only works for hlls using 64-bit integers.
    // Looking ahead,consider templating so that the double version might be helpful.

    bool may_contain(uint64_t element) const {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0); i < Space::COUNT;) if(!hlls_[k++].may_contain(key.arr_[i++])) return false;
            } while(sptr < eptr);
            return true;
        } else { // if size() >= Space::COUNT
            for(unsigned i(0); i < size(); ++i) if(!hlls_[i].may_contain(hf_(element ^ seeds_[i]))) return false;
            return true;
        }
    }
    void addh(uint64_t val) {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            const SType element = Space::set1(val);
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0) ; i < Space::COUNT; hlls_[k++].add(key.arr_[i++]));
                assert(k <= size());
            } while(sptr < eptr);
        }
    }
    double creport() const {
        if(is_calculated_) return value_;
        double ret(hlls_[0].creport());
        for(size_t i(1); i < size(); ret += hlls_[i++].creport());
        ret /= static_cast<double>(size());
        value_ = ret;
        return value_ = ret;
    }
    hlfbase_t &operator+=(const hlfbase_t &other) {
        if(other.size() != size()) throw std::runtime_error("Wrong number of subsketches.");
        if(other.hlls_[0].p() != hlls_[0].p()) throw std::runtime_error("Wrong size of subsketches.");
        for(unsigned i(0); i < size(); ++i) {
            hlls_[i] += other.hlls_[i];
            hlls_[i].not_ready();
        }
        is_calculated_ = false;
    }
    hlfbase_t operator+(const hlfbase_t &other) const {
        // Could be more directly optimized.
        hlfbase_t ret = *this;
        ret += other;
    }
    double report() noexcept {
        if(is_calculated_) return value_;
        hlls_[0].csum();
#if DIVIDE_EVERY_TIME
        double ret(hlls_[0].report() / static_cast<double>(size()));
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report() / static_cast<double>(size());
        }
#else
        double ret(hlls_[0].report());
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report();
        }
        ret /= static_cast<double>(size());
#endif
        return value_ = ret;
    }
    double med_report() noexcept {
        if(values_.empty())
            values_.reserve(size());
        values_.clear();
        for(auto &hll: hlls_) values_.emplace_back(hll.report());
        if(size() & 1) {
            if(size() < 32)
                sort::insertion_sort(std::begin(values_), std::end(values_));
            else
                std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) + 1, std::end(values_));
            return values_[size() >> 1];
        }
        if(size() < 32) {
            sort::insertion_sort(std::begin(values_), std::end(values_));
            return .5 * (values_[size() >> 1] + values_[(size() >> 1) - 1]);
        }
        std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) - 1, std::end(values_));
        return .5 * (values_[(values_.size() >> 1) - 1] + *std::min_element(std::cbegin(values_) + (size() >> 1), std::cend(values_)));
    }
    // Attempt strength borrowing across hlls with different seeds
    double chunk_report() const {
        if((size() & (size() - 1)) == 0) {
            std::array<uint64_t, 64> counts{0};
            for(const auto &hll: hlls_) detail::inc_counts(counts, hll.core());
            const auto diff = (sizeof(uint32_t) * CHAR_BIT - clz((uint32_t)size()) - 1);
            const auto new_p = hlls_[0].p() + diff;
            const auto new_m = (1ull << new_p);
            return detail::calculate_estimate(counts, hlls_[0].get_estim(), new_m,
                                              new_p, make_alpha(new_m)) / (1ull << diff);
        } else {
            std::fprintf(stderr, "chunk_report is currently only supported for powers of two.");
            return creport();
            // Could try weight averaging, but currently I just report default when size is not a power of two.
        }
    }
};
template<typename HashType>
class chlf_t { // contiguous hyperlogfilter
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<uint8_t, Allocator<uint8_t>>      core_;
    std::vector<uint64_t, Allocator<uint64_t>>   seeds_;
    std::vector<double>                         values_;
    const uint32_t                                subp_;
    const uint32_t                                  ns_; // log 2 number of subsketches
    EstimationMethod                             estim_;
    JointEstimationMethod                       jestim_;
    mutable double                               value_;
    bool                                 is_calculated_;
    const HashType                                  hf_;
public:
    template<typename... Args>
    chlf_t(size_t size, uint64_t seedseed, Args &&... args): value_(0), is_calculated_(0), hf_{} {
        auto sfs = detail::seeds_from_seed(seedseed, size);
        assert(sfs.size());
    }
    auto nbytes()    const {return core_.size();}
    auto nsketches() const {return ns_;}
    auto m() const {return core_.size();}
    auto subp() const {return subp_;}
    auto subq() const {return (sizeof(uint64_t) * CHAR_BIT) - subp_;}
    void add(uint64_t hashval, unsigned subidx) {
#ifndef NOT_THREADSAFE
        // subidx << subp gets us to the subtable, hashval >> subq gets us to the slot within that table.
        for(const uint32_t index((hashval >> subq()) + (subidx << subp())), lzt(clz(((hashval << 1)|1) << (subp() - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index((hashval >> subq()) + (subidx << subp())), lzt(clz(((hashval << 1)|1) << (subp() - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
        
    }
    void addh(uint64_t val) {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(nsketches() >= Space::COUNT) {
            if(nsketches() & (nsketches() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            const SType element = Space::set1(val);
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0); i < Space::COUNT;add(key.arr_[i++], k++));
                assert(k <= nsketches);
            } while(sptr < eptr);
        }
        else for(unsigned i(0); i < nsketches();add(hf_(val ^ seeds_[i]), i), ++i);
    }
#if 0
    void write(const char *fn) const {
        gzFile fp = gzopen(fn, "wb");
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
        this->write(fp);
        gzclose(fp);
    }
    void clear() {
        value_ = is_calculated_ = 0;
        for(auto &hll: hlls_) hll.clear();
    }
    void read(const char *fn) {
        gzFile fp = gzopen(fn, "rb");
        if(fp == nullptr) throw std::runtime_error("Could not open file.");
        this->read(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        gzwrite(fp, &ns_, sizeof(ns_));
        for(auto &seed: seeds_) gzwrite(fp, &seed, sizeof(seed));
        for(const auto &hll: hlls_) {
            hll.write(fp);
        }
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t size;
        gzread(fp, &size, sizeof(size));
        seeds_.resize(size);
        for(unsigned i(0); i < size; gzread(fp, seeds_.data() + i++, sizeof(seeds_[0])));
        hlls_.clear();
        while(hlls_.size() < size) hlls_.emplace_back(fp);
        gzclose(fp);
    }

    // This only works for hlls using 64-bit integers.
    // Looking ahead,consider templating so that the double version might be helpful.

    bool may_contain(uint64_t element) const {
        using Space = vec::SIMDTypes<uint64_t>;
        using SType = typename Space::Type;
        using VType = typename Space::VType;
        unsigned k = 0;
        if(size() >= Space::COUNT) {
            if(size() & (size() - 1)) throw std::runtime_error("NotImplemented: supporting a non-power of two.");
            const SType *sptr = (const SType *)&seeds_[0];
            const SType *eptr = (const SType *)&seeds_.back();
            VType key;
            do {
                key = hf_(*sptr++ ^ element);
                for(unsigned i(0); i < Space::COUNT;) if(!hlls_[k++].may_contain(key.arr_[i++])) return false;
            } while(sptr < eptr);
            return true;
        } else { // if size() >= Space::COUNT
            for(unsigned i(0); i < size(); ++i) if(!hlls_[i].may_contain(hf_(element ^ seeds_[i]))) return false;
            return true;
        }
    }
    double creport() const {
        if(is_calculated_) return value_;
        double ret(hlls_[0].creport());
        for(size_t i(1); i < size(); ret += hlls_[i++].creport());
        ret /= static_cast<double>(size());
        value_ = ret;
        return value_ = ret;
    }
    double report() noexcept {
        if(is_calculated_) return value_;
        hlls_[0].csum();
#if DIVIDE_EVERY_TIME
        double ret(hlls_[0].report() / static_cast<double>(size()));
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report() / static_cast<double>(size());
        }
#else
        double ret(hlls_[0].report());
        for(size_t i(1); i < size(); ++i) {
            hlls_[i].csum();
            ret += hlls_[i].report();
        }
        ret /= static_cast<double>(size());
#endif
        return value_ = ret;
    }
    double med_report() noexcept {
        if(values_.empty())
            values_.reserve(size());
        values_.clear();
        for(auto &hll: hlls_) values_.emplace_back(hll.report());
        if(size() & 1) {
            if(size() < 32)
                sort::insertion_sort(std::begin(values_), std::end(values_));
            else
                std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) + 1, std::end(values_));
            return values_[size() >> 1];
        }
        if(size() < 32) {
            sort::insertion_sort(std::begin(values_), std::end(values_));
            return .5 * (values_[size() >> 1] + values_[(size() >> 1) - 1]);
        }
        std::nth_element(std::begin(values_), std::begin(values_) + (size() >> 1) - 1, std::end(values_));
        return .5 * (values_[(values_.size() >> 1) - 1] + *std::min_element(std::cbegin(values_) + (size() >> 1), std::cend(values_)));
    }
    // Attempt strength borrowing across hlls with different seeds
    double chunk_report() const {
        if((size() & (size() - 1)) == 0) {
            std::array<uint64_t, 64> counts{0};
            for(const auto &hll: hlls_) detail::inc_counts(counts, hll.core());
            const auto diff = (sizeof(uint32_t) * CHAR_BIT - clz((uint32_t)size()) - 1);
            const auto new_p = hlls_[0].p() + diff;
            const auto new_m = (1ull << new_p);
            return detail::calculate_estimate(counts, hlls_[0].get_estim(), new_m,
                                              new_p, make_alpha(new_m)) / (1ull << diff);
        } else {
            std::fprintf(stderr, "chunk_report is currently only supported for powers of two.");
            return creport();
            // Could try weight averaging, but currently I just report default when size is not a power of two.
        }
    }
#endif
};
using hlf_t = hlfbase_t<>;



#ifdef ENABLE_HLL_DEVELOP
#else
} // namespace dev
#endif
} // namespace hll
