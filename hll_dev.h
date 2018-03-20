#pragma once


namespace hll {

#ifdef ENABLE_HLL_DEVELOP
#pragma message("hll develop enabled (-DENABLE_HLL_DEVELOP)")
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
    std::vector<uint8_t, Allocator> dcore_;
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

template<typename SeedHllType=seedhll_t>
class hlfbase_t {
protected:
    // Note: Consider using a shared buffer and then do a weighted average
    // of estimates from subhlls of power of 2 sizes.
    std::vector<SeedHllType> hlls_;
    mutable double value_;
    bool is_calculated_;
public:
    template<typename... Args>
    hlfbase_t(size_t size, uint64_t seedseed, Args &&... args): value_(0), is_calculated_(0) {
        auto sfs = detail::seeds_from_seed(seedseed, size);
        assert(sfs.size());
        hlls_.reserve(size);
        for(const auto seed: sfs)
            hlls_.emplace_back(seed, std::forward<Args>(args)...);
    }
    auto size() const {return hlls_.size();}
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
        for(const auto &hll: hlls_) {
            hll.write(fp);
        }
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t size;
        gzread(fp, &size, sizeof(size));
        hlls_.clear();
        while(hlls_.size() < size) hlls_.emplace_back(fp);
        gzclose(fp);
    }

    // This only works for hlls using 64-bit integers.
    // Looking ahead,consider templating so that the double version might be helpful.

    auto may_contain(uint64_t element) const {
#pragma message("Note: may_contain only works for the HyperLogFilter in the case of 64-bit integer insertions. One must hash a string to a 64-bit integer first in order to use it for this purpose.")
        return std::accumulate(hlls_.begin() + 1, hlls_.end(), hlls_.front().may_contain(wang_hash(element ^ hlls_.front().seed())),
                               [element](auto a, auto b) {
            return a && b.may_contain(wang_hash(element ^ b.seed()));
        });
    }
    void addh(uint64_t val) {
        val = hlls_[0].hf_(val);
        for(auto &hll: hlls_) hll.add(val);
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
        if(!hlls_[0].is_ready()) hlls_[0].sum();
        double ret(hlls_[0].report() / static_cast<double>(size()));
        for(size_t i(1); i < size(); ++i) {
            if(!hlls_[i].is_ready()) hlls_[i].sum();
            ret += hlls_[i].report() / static_cast<double>(size());
        }
        return value_ = ret;
    }
    double med_report() noexcept {
        std::vector<double> values;
        values.reserve(size());
        for(auto &hll: hlls_) values.emplace_back(hll.report());
        std::sort(values.begin(), values.end());
        return size() & 1 ? values[size() >> 1]
                          : 0.5 * (values[size() >> 1] + values[(size() >> 1) - 1]);
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
using hlf_t = hlfbase_t<>;


template<typename HllType>
std::array<double, 3> ertl_joint(const HllType &h1, const HllType &h2) {
    std::array<double, 3> ret;
    auto p = h1.p();
    auto q = h1.q();
    auto c1 = detail::sum_counts(h1.core());
    auto c2 = detail::sum_counts(h2.core());
    const double cAX = ertl_ml_estimate(c1, h1.p(), h1.q());
    const double cBX = ertl_ml_estimate(c2, h2.p(), h2.q());
    std::fprintf(stderr, "cAX ml est: %lf. cBX ml els: %lf\n", cAX, cBX);
    //const double cBX = hl2.creport();
    //const double cBX = hl2.creport();
    const double cABX = union_size(h1, h2);
    std::array<uint64_t, 64> countsAXBhalf{0}; // 
    std::array<uint64_t, 64> countsBXAhalf{0}; // 
    countsAXBhalf[q] = h1.m();
    countsBXAhalf[q] = h1.m();
    std::array<uint64_t, 64> countsG1{0};
    std::array<uint64_t, 64> countsG2{0};
    std::array<uint64_t, 64> countsL1{0};
    std::array<uint64_t, 64> countsL2{0};
    std::array<uint64_t, 64> countsEq{0};
    {
        const auto &core1(h1.core()), &core2(h2.core());
        for(uint64_t i(0); i < core1.size(); ++i) {
            switch((core1[i] > core2[i]) << 1 | (core2[i] > core1[i])) {
                case 0:
                    ++countsEq[core1[i]]; break;
                case 1:
                    ++countsG2[core2[i]];
                    ++countsL1[core1[i]];
                    break;
                case 2: 
                    ++countsG1[core1[i]];
                    ++countsL2[core2[i]];
                    break;
            }
        }
    }
    for(unsigned _q = 0; _q < q; ++_q) {
        // Handle AXBhalf
        countsAXBhalf[_q] = countsG1[_q] + countsEq[_q] + countsG2[_q + 1];
        assert(countsAXBhalf[q] >= countsAXBhalf[_q]);
        countsAXBhalf[q] -= countsAXBhalf[_q];

        // Handle BXAhalf
        countsBXAhalf[_q] = countsG2[_q] + countsEq[_q] + countsG1[_q + 1];
        assert(countsBXAhalf[q] >= countsBXAhalf[_q]);
        countsBXAhalf[q] -= countsBXAhalf[_q];
    }
    double cAXBhalf = ertl_ml_estimate(countsAXBhalf, p, q - 1);
    double cBXAhalf = ertl_ml_estimate(countsBXAhalf, p, q - 1);
    std::fprintf(stderr, "cAXBhalf = %lf\n", cAXBhalf);
    std::fprintf(stderr, "cBXAhalf = %lf\n", cBXAhalf);
    ret[0] = cABX - cBX;
    ret[1] = cABX - cAX;
    double cX1 = (1.5 * cBX + 1.5*cAX - cBXAhalf - cAXBhalf);
    double cX2 = 2.*(cBXAhalf + cAXBhalf) - 3.*cABX;
    std::fprintf(stderr, "Halves of contribution: %lf, %lf. Initial est: %lf. Result: %lf\n", cX1, cX2, cABX, std::max(0., 0.5 * (cX1 + cX2)));
    ret[2] = std::max(0., 0.5 * (cX1 + cX2));
    return ret;
}

template<typename HllType>
std::array<double, 3> ertl_joint_union(HllType &h1, HllType &h2) {
    if(h1.not_ready()) h1.sum();
    if(h2.not_ready()) h2.sum();
    return ertl_joint_union(static_cast<const HllType &>(h1), static_cast<const HllType &>(h2));
}

#ifdef ENABLE_HLL_DEVELOP
#else
} // namespace dev
#endif
} // namespace hll
