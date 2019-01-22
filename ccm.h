#pragma once
#include "common.h"
#include "aesctr/aesctr.h"
#include <ctime>

namespace sketch {

namespace cm {

namespace detail {
    template<typename IntType, typename=typename std::enable_if<std::is_signed<IntType>::value>::type>
    static constexpr IntType signarr []{static_cast<IntType>(-1), static_cast<IntType>(1)};

    template<typename T>
    struct IndexedValue {
        using Type = typename std::decay_t<decltype(*(T(100, 10).cbegin()))>;
    };
    template<typename T>
    static constexpr int range_check(unsigned nbits, T val) {
        CONST_IF(std::is_signed<T>::value) {
            const int64_t v = val;
            return v < -int64_t(1ull << (nbits - 1)) ? -1
				                     : v > int64_t((1ull << (nbits - 1)) - 1);
        } else {
            return val >= (1ull << nbits);
        }
    }
}

namespace update {

struct Increment {
    // Saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) const {
        ref = ref + (ref < maxval);
        //ref += (ref < maxval);
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType nbits) const {
            unsigned count = con[ref[0]];
            if(detail::range_check<typename detail::IndexedValue<Container>::Type>(nbits, ++count) == 0)
                for(const auto el: ref)
                    con[el] = count;
    }
    template<typename... Args>
    Increment(Args &&... args) {}
    static uint64_t est_count(uint64_t val) {
        return val;
    }
    template<typename T1, typename T2>
    static auto combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        return RetType(i) + RetType(j);
    }
};


struct CountSketch {
    // Saturates
    template<typename T, typename IntType, typename IntType2>
    void operator()(T &ref, IntType maxval, IntType2 hash) const {
        ref = int64_t(ref) + detail::signarr<IntType2>[hash&1];
    }
    template<typename T, typename Container, typename IntType>
    ssize_t operator()(std::vector<T> &ref, std::vector<T> &hashes, Container &con, IntType nbits) const {
        using IDX = typename detail::IndexedValue<Container>::Type;
        IDX newval;
        std::vector<IDX> s;
        assert(ref.size() == hashes.size());
        for(size_t i(0); i < ref.size(); ++i) {
#if !NDEBUG
            auto inc = detail::signarr<::std::int64_t>[hashes[i]&1];
            assert(inc == 1 || inc == -1);
            newval = con[ref[i]] + inc;
#else
            newval = con[ref[i]] + detail::signarr<::std::int64_t>[hashes[i]&1];
#endif
            s.push_back(newval);
            if(detail::range_check<IDX>(nbits, newval) == 0)
                con[ref[i]] = newval;
        }
        if(s.size()) {
            common::sort::insertion_sort(s.begin(), s.end());
            return (s[s.size()>>1] + s[(s.size()-1)>>1]) >> 1;
        }
        return 0;
    }
    template<typename... Args>
    static void Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
        return val;
    }
    template<typename T1, typename T2>
    static uint64_t combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        std::fprintf(stderr, "[%s:%d:%s] I'm not sure this is actually right; this is essentially a placeholder.\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
        return RetType(i) + RetType(j);
    }
    template<typename... Args>
    CountSketch(Args &&... args) {}
};

struct PowerOfTwo {
    aes::AesCtr<uint64_t, 2> rng_;
    uint64_t  gen_;
    uint8_t nbits_;
    // Also saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
#if !NDEBUG
        std::fprintf(stderr, "maxval: %zu. ref: %zu\n", size_t(maxval), size_t(ref));
#endif
        if(unsigned(ref) == 0) ref = 1;
        else {
            if(ref >= maxval) return;
            if(__builtin_expect(nbits_ < ref, 0)) gen_ = rng_(), nbits_ = 64;
            const unsigned oldref = ref;
            ref = ref + ((gen_ & (UINT64_C(-1) >> (64 - unsigned(ref)))) == 0);
            gen_ >>= oldref, nbits_ -= oldref;
        }
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType nbits) {
        uint64_t val = con[ref[0]];
#if !NDEBUG
        //std::fprintf(stderr, "Value before incrementing: %zu\n", size_t(val));
        //for(const auto i: ref) std::fprintf(stderr, "These should all be the same: %u\n", unsigned(con[i]));
#endif
        if(val == 0) {
            for(const auto el: ref)
                con[el] = 1;
        } else {
            if(__builtin_expect(nbits_ < val, 0)) gen_ = rng_(), nbits_ = 64;
#if !NDEBUG
            //std::fprintf(stderr, "bitmasked gen: %u. val: %u\n", unsigned((gen_ & (UINT64_C(-1) >> (64 - val)))), unsigned(val));
#endif
            auto oldval = val;
            if((gen_ & (UINT64_C(-1) >> (64 - val))) == 0) {
#if !NDEBUG
            //std::fprintf(stderr, "We incremented a second time!\n");
#endif
                ++val;
                if(detail::range_check(nbits, val) == 0)
                    for(const auto el: ref)
                        con[el] = val;
            }
            gen_ >>= oldval;
            nbits_ -= oldval;
        }
    }
    template<typename T1, typename T2>
    static auto combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        RetType i_(i), j_(j);
        return std::max(i_, j_) + (i == j);
    }
    PowerOfTwo(uint64_t seed=0): rng_(seed), gen_(rng_()), nbits_(64) {}
    static uint64_t est_count(uint64_t val) {
        return uint64_t(1) << (val - 1);
    }
};

} // namespace update

using namespace common;

template<typename UpdateStrategy=update::Increment,
         typename VectorType=DefaultCompactVectorType,
         typename HashStruct=common::WangHash,
         bool conservative_update=true>
class ccmbase_t {
    static_assert(!std::is_same<UpdateStrategy, update::CountSketch>::value || std::is_signed<typename detail::IndexedValue<VectorType>::Type>::value,
                  "If CountSketch is used, value must be signed.");

protected:
    VectorType        data_;
    UpdateStrategy updater_;
    const unsigned nhashes_;
    const unsigned l2sz_:16;
    const unsigned nbits_:16;
    const HashStruct hf_;
    const uint64_t mask_;
    const uint64_t subtbl_sz_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;

public:
    static constexpr bool supports_deletion() {
        return !conservative_update;
    }
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this),
                              seeds_.size() * sizeof(seeds_[0]) + data_.bytes());
    }
    void clear() {
        common::detail::zero_memory(data_, std::log2(subtbl_sz_));
    }
    template<typename... Args>
    ccmbase_t(int nbits, int l2sz, int nhashes=4, uint64_t seed=0, Args &&... args):
            data_(nbits, nhashes << l2sz),
            updater_(seed + l2sz * nbits * nhashes),
            nhashes_(nhashes), l2sz_(l2sz),
            nbits_(nbits), hf_(std::forward<Args>(args)...),
            mask_((1ull << l2sz) - 1),
            subtbl_sz_(1ull << l2sz)
    {
        if(__builtin_expect(nbits < 0, 0)) throw std::runtime_error("Number of bits cannot be negative.");
        if(__builtin_expect(l2sz < 0, 0)) throw std::runtime_error("l2sz cannot be negative.");
        if(__builtin_expect(nhashes < 0, 0)) throw std::runtime_error("nhashes cannot be negative.");
        std::mt19937_64 mt(seed + 4);
        auto nperhash64 = lut::nhashesper64bitword[l2sz];
        while(seeds_.size() * nperhash64 < static_cast<unsigned>(nhashes)) seeds_.emplace_back(mt());
        clear();
#if !NDEBUG
        std::fprintf(stderr, "%i bits for each number, %i is log2 size of each table, %i is the number of subtables. %zu is the number of 64-bit hashes with %u nhashesper64bitword\n", nbits, l2sz, nhashes, seeds_.size(), nperhash64);
        std::fprintf(stderr, "Size of updater: %zu\n", sizeof(updater_));
#endif
    }
    VectorType &ref() {return data_;}
    auto addh(uint64_t val) {return add(val);}
    template<typename T>
    T hash(T val) const {
        return hf_(val);
    }
    uint64_t subhash(uint64_t val, uint64_t seedind) const {
        return hash(((val ^ seeds_[seedind]) & mask_) + (val & mask_));
    }
    uint64_t mask() const {return mask_;}
    auto np() const {return l2sz_;}
    auto &at_pos(uint64_t hv, uint64_t seedind) {
        return data_[(hv & mask_) + (seedind << np())];
    }
    const auto &at_pos(uint64_t hv, uint64_t seedind) const {
        return data_[(hv & mask_) + (seedind << np())];
    }
    bool may_contain(uint64_t val) const {
        throw std::runtime_error("This needs to be rewritten after subhash refactoring.");
        unsigned nhdone = 0;
        Space::VType v;
        const Space::Type *seeds(reinterpret_cast<Space::Type *>(&seeds_[0]));
        assert(data_.size() == subtbl_sz_ * nhashes_);
        while(nhashes_ - nhdone >= Space::COUNT) {
            v = hash(Space::xor_fn(Space::set1(val), *seeds++));
            for(unsigned i = 0; i < Space::COUNT; ++i) {
                if(at_pos(v.arr_[i], nhdone++) == 0) return false;
                v >>= np();
            }
        }
        while(nhdone < nhashes_) {
            if(at_pos(v, nhdone++) == 0)
                return false;
            v >>= np();
        }
        return true;
    }
    uint32_t may_contain(Space::VType val) const {
        throw std::runtime_error("This needs to be rewritten after subhash refactoring.");
        Space::VType tmp, and_val;
        unsigned nhdone = 0;
        const Space::Type *seeds(reinterpret_cast<const Space::Type *>(seeds_.data()));
        and_val = Space::set1(mask_);
        uint32_t ret = static_cast<uint32_t>(-1) >> ((sizeof(ret) * CHAR_BIT) - Space::COUNT);
        //uint32_t bitmask;
        while(nhdone + Space::COUNT < nhashes_) {
            for(unsigned i = 0; i < Space::COUNT; ++i) {
                tmp = Space::set1(val.arr_[i]);
                tmp = hash(Space::xor_fn(tmp.simd_, Space::load(seeds++)));
                for(unsigned j = 0; j < Space::COUNT; ++i) {
                    ret &= ~(uint32_t(data_[tmp.arr_[j] + (nhdone++ << np())] == 0) << i);
                }
                nhdone -= Space::COUNT;
            }
            nhdone += Space::COUNT;
        }
        while(nhdone < nhashes_) {
            for(auto ptr(reinterpret_cast<const uint64_t *>(seeds)); ptr < &seeds_[seeds_.size()];) {
                tmp = Space::xor_fn(val.simd_, Space::set1(*ptr++));
                for(unsigned j = 0; j < Space::COUNT; ++j) {
                    ret &= ~(uint32_t(data_[tmp.arr_[j] + (nhdone << np())] == 0) << j);
                }
            }
            ++nhdone;
        }
        return ret;
    }
    ssize_t sub(const uint64_t val) {
        CONST_IF(!std::is_same<UpdateStrategy, update::Increment>::value) {
            std::fprintf(stderr, "Can't delete from an approximate counting sketch.");
            return std::numeric_limits<ssize_t>::min();
        }
        CONST_IF(unlikely(!supports_deletion())) {
            std::fprintf(stderr, "Can't delete from a conservative update scheme sketch.");
            return std::numeric_limits<ssize_t>::min();
        }
        unsigned nhdone = 0, seedind = 0;
        const auto nperhash64 = lut::nhashesper64bitword[l2sz_];
        const auto nbitsperhash = l2sz_;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp;
        ssize_t ret = std::numeric_limits<decltype(ret)>::max();
        while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(Space::COUNT * nperhash64)) {
            Space::VType(hash(Space::xor_fn(vb.simd_, Space::load(sptr++)))).for_each([&](uint64_t subval) {
                for(unsigned k(0); k < nperhash64;) {
                    auto ref = data_[((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                    ref = ref - 1;
                    ret = std::min(ret, ssize_t(ref));
                }
            });
            seedind += Space::COUNT;
        }
        while(nhdone < nhashes_) {
            uint64_t hv = hash(val ^ seeds_[seedind]);
            for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone);) {
                auto ref = data_[((hv >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                ref = ref - 1;
                ret = std::min(ret, ssize_t(ref));
            }
            ++seedind;
        }
        return ret;
    }
    ssize_t add(const uint64_t val) {
        unsigned nhdone = 0, seedind = 0;
        const auto nperhash64 = lut::nhashesper64bitword[l2sz_];
        const auto nbitsperhash = l2sz_;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp;
        ssize_t ret;
        CONST_IF(conservative_update) {
            std::vector<uint64_t> indices, best_indices;
            indices.reserve(nhashes_);
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(Space::COUNT * nperhash64)) {
                tmp = hash(Space::xor_fn(vb.simd_, Space::load(sptr++)));
                tmp.for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64; indices.push_back(((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_));
                });
                seedind += Space::COUNT;
            }
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[seedind]);
                for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); indices.push_back(((hv >> (k++ * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++));
                ++seedind;
            }
            best_indices.push_back(indices[0]);
            ssize_t minval = data_[indices[0]];
            unsigned score;
            for(size_t i(1); i < indices.size(); ++i) {
                // This will change with
                if((score = data_[indices[i]]) == minval) {
                    best_indices.push_back(indices[i]);
                } else if(score < minval) {
                    best_indices.clear();
                    best_indices.push_back(indices[i]);
                    minval = score;
                }
            }
            updater_(best_indices, data_, nbits_);
            ret = minval;
        } else { // not conservative update. This means we support deletions
            ret = std::numeric_limits<decltype(ret)>::max();
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(Space::COUNT * nperhash64)) {
                Space::VType(hash(Space::xor_fn(vb.simd_, Space::load(sptr++)))).for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64;) {
                        auto ref = data_[((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                        updater_(ref, 1u << nbits_);
                        ret = std::min(ret, ssize_t(ref));
                    }
                });
                seedind += Space::COUNT;
            }
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[seedind++]);
                for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone);) {
                    auto ref = data_[((hv >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                    updater_(ref, 1u << nbits_);
                    ret = std::min(ret, ssize_t(ref));
                }
            }
        }
        return ret;
    }
    uint64_t est_count(uint64_t val) const {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType tmp;
        //const Space::VType and_val = Space::set1(mask_);
        const Space::VType vb = Space::set1(val);
        unsigned nhdone = 0, seedind = 0, k;
        const auto nperhash64 = lut::nhashesper64bitword[l2sz_];
        const auto nbitsperhash = l2sz_;
        uint64_t count = std::numeric_limits<uint64_t>::max();
        while(nhashes_ - nhdone > Space::COUNT * nperhash64) {
            tmp = hash(Space::xor_fn(vb.simd_, Space::load(sptr++)));
            tmp.for_each([&](const uint64_t subval){
                for(k = 0; k < nperhash64; count = std::min(count, uint64_t(data_[((subval >> (k++ * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++])));
            });
            seedind += Space::COUNT;
        }
        while(nhdone < nhashes_) {
            uint64_t hv = hash(val ^ seeds_[seedind++]);
            for(k = 0; k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); ++k) {
                count = std::min(count, uint64_t(data_[(hv >> (k * nbitsperhash)) & mask_]) + nhdone++ * subtbl_sz_);
            }
        }
        return updater_.est_count(count);
#if 0
        // Now this must be a count sketch.
        std::vector<int64_t> estimates;
        estimates.reserve(nhashes_);
        while(nhashes_ - nhdone > Space::COUNT * nperhash64) {
            tmp = hash(Space::xor_fn(vb.simd_, Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval) {
                for(k = 0; k < nperhash64; ++k) {
                    estimates.push_back(data_[((subval >> (k * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++] * ((subval >> ((nperhash64 - k - 1) * nbitsperhash)) & 1 ? 1: -1) );
                }
                ++seedind;
            });
        }
        while(nhdone < nhashes_) {
            uint64_t hv = hash(val ^ seeds_[seedind]);
            for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); ++k) {
                estimates.push_back(data_[((hv >> (k * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++] * detail::signarr<int64_t>[(hv >> ((nperhash64 - k - 1) * nbitsperhash)) & 1]);
            }
            ++seedind;
        }
        sort::insertion_sort(std::begin(estimates), std::end(estimates));
        return std::max(static_cast<int64_t>(0),
                        nhashes_ & 1 ? estimates[nhashes_>>1]
                                     : (estimates[nhashes_>>1] + estimates[(nhashes_>>1) - 1]) / 2);
#endif
    }
    ccmbase_t operator+(const ccmbase_t &other) const {
        ccmbase_t cpy = *this;
        cpy += other;
        return cpy;
    }
    ccmbase_t operator&(const ccmbase_t &other) const {
        ccmbase_t cpy = *this;
        cpy &= other;
        return cpy;
    }
    ccmbase_t &operator&=(const ccmbase_t &other) {
        if(seeds_.size() != other.seeds_.size() || !std::equal(seeds_.cbegin(), seeds_.cend(), other.seeds_.cbegin()))
            throw std::runtime_error("Could not add sketches together with different hash functions.");
        for(size_t i(0), e(data_.size()); i < e; ++i) {
            data_[i] = std::min(static_cast<unsigned>(data_[i]), static_cast<unsigned>(other.data_[i]));
        }
        return *this;
    }
    ccmbase_t &operator+=(const ccmbase_t &other) {
        if(seeds_.size() != other.seeds_.size() || !std::equal(seeds_.cbegin(), seeds_.cend(), other.seeds_.cbegin()))
            throw std::runtime_error("Could not add sketches together with different hash functions.");
        for(size_t i(0), e(data_.size()); i < e; ++i) {
            data_[i] = updater_.combine(data_[i], other.data_[i]);
        }
        return *this;
    }
};
template<typename HashStruct=common::WangHash, typename CounterType=int32_t, typename=typename std::enable_if<std::is_signed<CounterType>::value>::type>
class csbase_t {
    /*
     * Commentary: because of chance, one can end up with a negative number as an estimate.
     * Either the item collided with another item which was quite large and it was outweighed
     * or it and others in the bucket were not heavy enough and by chance it did
     * not weigh over the other items with the opposite sign. Treat these as 0s.
    */
    std::vector<CounterType, Allocator<CounterType>> core_;
    uint32_t np_;
    uint32_t nh_;
    uint32_t nph_;
    const HashStruct hf_;
    uint64_t mask_;
    std::vector<CounterType, Allocator<CounterType>> seeds_;
public:
    template<typename...Args>
    csbase_t(unsigned np, unsigned nh=1, unsigned seedseed=137, Args &&...args):
        core_(uint64_t(nh) << np), np_(np), nh_(nh), nph_(64 / (np + 1)), hf_(std::forward<Args>(args)...),
        mask_((1ull << np_) - 1), 
        seeds_((nh_ + (nph_ - 1)) / nph_ - 1)
    {
        aes::AesCtr<uint64_t> gen(np + nh + seedseed);
        for(auto &el: seeds_) el = gen();
        // Just to make sure that simd addh is always accessing owned memory.
        while(seeds_.size() < sizeof(Space::Type) / sizeof(uint64_t)) seeds_.emplace_back(gen());
    }
    void addh(uint64_t val) {
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1), add(v, added++));
        auto it = seeds_.begin();
        while(added < nh_) {
            v = hf_(*it++ ^ val);
            for(unsigned k = nph_; k--; v >>= (np_ + 1)) {
                add(v, added++);
                if(added == nh_) break; // this could be optimized by pre-scanning, I think.
            }
        }
    }
    void subh(uint64_t val) {
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1), add(v, added++));
        auto it = seeds_.begin();
        while(added < nh_) {
            v = hf_(*it++ ^ val);
            for(unsigned k = nph_; k--; v >>= (np_ + 1)) {
                sub(v, added++);
                if(added == nh_) break; // this could be optimized by pre-scanning, I think.
            }
        }
    }
    INLINE void add(uint64_t hv, unsigned subidx) noexcept {
        at_pos(hv, subidx) += sign(hv);
    }
    INLINE void sub(uint64_t hv, unsigned subidx) noexcept {
        at_pos(hv, subidx) -= sign(hv);
    }
    INLINE auto &at_pos(uint64_t hv, unsigned subidx) noexcept {
        assert((hv & mask_) + (subidx << np_) < core_.size() || !std::fprintf(stderr, "hv & mask_: %zu. subidx %d. np: %d. nh: %d. size: %zu\n", size_t(hv&mask_), subidx, np_, nh_, core_.size()));
        return core_[(hv & mask_) + (subidx << np_)];
    }
    INLINE int sign(uint64_t hv) const noexcept {return hv & (1ul << np_) ? 1: -1;}
    INLINE auto at_pos(uint64_t hv, unsigned subidx) const noexcept {
        assert((hv & mask_) + (subidx << np_) < core_.size());
        return core_[(hv & mask_) + (subidx << np_)];
    }
    INLINE void subh(Space::VType hv) noexcept {
        Space::VType tmp = hf_(hv);
        unsigned gadded = 0;
        tmp.for_each([&](uint64_t v) {
            for(uint32_t added = 0; added++ < std::min(nph_, nh_) && gadded < nh_;v >>= (np_ + 1))
                sub(v, gadded++);
        });
        for(auto it = seeds_.begin(); gadded < nh_;) {
            tmp = hf_(Space::xor_fn(Space::set1(*it++), hv));
            unsigned lastgadded = gadded;
            tmp.for_each([&](uint64_t v) {
                unsigned added;
                for(added = lastgadded; added < std::min(lastgadded + nph_, nh_); sub(v, added++), v >>= (np_ + 1));
                gadded = added;
            });
        }
    }
    INLINE void addh(Space::VType hv) noexcept {
        Space::VType tmp = hf_(hv);
        unsigned gadded = 0;
        tmp.for_each([&](uint64_t v) {
            for(uint32_t added = 0; added++ < std::min(nph_, nh_) && gadded < nh_;v >>= (np_ + 1))
                add(v, gadded++);
        });
        for(auto it = seeds_.begin(); gadded < nh_;) {
            tmp = hf_(Space::xor_fn(Space::set1(*it++), hv));
            unsigned lastgadded = gadded;
            tmp.for_each([&](uint64_t v) {
                unsigned added;
                for(added = lastgadded; added < std::min(lastgadded + nph_, nh_); add(v, added++), v >>= (np_ + 1));
                gadded = added;
            });
        }
    }
    CounterType est_count(uint64_t val) const {
#if AVOID_ALLOCA
        CounterType *ptr = static_cast<CounterType *>(std::malloc(nh_ * sizeof(CounterType))), *p = ptr;
#else
        CounterType *ptr = static_cast<CounterType *>(__builtin_alloca(nh_ * sizeof(CounterType))), *p = ptr;
#endif
        if(__builtin_expect(ptr == nullptr, 0)) throw std::bad_alloc();
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1)) {
            *p++ = at_pos(v, added++) * sign(v);
        }
        for(auto it = seeds_.begin();;) {
            v = hf_(*it++ ^ val);
            for(unsigned k = 0; k < nph_; ++k, v >>= (np_ + 1)) {
                *p++ = at_pos(v, added++) * sign(v);
                if(added == nh_) goto end;
            }
        }
        end:
        sort::insertion_sort(ptr, ptr + nh_);
#if AVOID_ALLOCA
        std::free(ptr);
#endif
        return (ptr[nh_>>1] + ptr[(nh_-1)>>1]) >> 1;
    }
};

template<typename VectorType=DefaultCompactVectorType,
         typename HashStruct=common::WangHash>
class cmmbase_t: protected ccmbase_t<update::Increment, VectorType, HashStruct> {
    uint64_t stream_size_;
    using BaseType = ccmbase_t<update::Increment, VectorType, HashStruct>;
    cmmbase_t(int nbits, int l2sz, int nhashes=4, uint64_t seed=0): BaseType(nbits, l2sz, nhashes, seed), stream_size_(0) {
        throw NotImplementedError("count min mean sketch not completed.");
    }
    void add(uint64_t val) {this->addh(val);}
    void addh(uint64_t val) {
        ++stream_size_;
        BaseType::addh(val);
    }
    uint64_t est_count(uint64_t val) const {
        return BaseType::est_count(val); // TODO: this (This is just
    }
};

using ccm_t = ccmbase_t<>;
using cmm_t = cmmbase_t<>;
using cs_t = csbase_t<>;
using pccm_t = ccmbase_t<update::PowerOfTwo>;

} // namespace cm
} // namespace sketch
