#pragma once
#ifndef HEAVY_KEEPER_H__
#define HEAVY_KEEPER_H__
#include "./common.h"
#include "aesctr/wy.h"
#include "tsg.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include "hash.h"
#if SKETCH_THREADSAFE
#include <mutex>
#endif

namespace sketch {

inline namespace hk {
template<typename HKType, typename ValueType, typename Hasher, typename Allocator, typename HashSetFingerprint, typename, typename>
class HeavyKeeperHeap;

template<size_t fpsize, size_t ctrsize=64-fpsize, typename Hasher=hash::WangHash, typename Policy=policy::SizePow2Policy<uint64_t>, typename RNG=wy::WyHash<uint64_t>, typename Allocator=common::Allocator<uint64_t>>
class HeavyKeeper {

    static_assert(fpsize > 0 && ctrsize > 0 && 64 % (fpsize + ctrsize) == 0, "fpsize and ctrsize must be evenly divisible into our 64-bit words and at least one must be nonzero.");

    struct decoded_register: std::pair<uint64_t, uint64_t> {
        uint64_t &count() {return this->first;}
        uint64_t &fp() {return this->second;}
    };
    template<typename HK, typename VT, typename H, typename A, typename HS, typename, typename>
    friend class HeavyKeeperHeap;


    // Members
    Policy pol_;
    size_t nh_;
    std::vector<uint64_t, Allocator> data_;
    Hasher hasher_;
    double b_;
    uint64_t n_updates_;
#if SKETCH_THREADSAFE
    std::unique_ptr<std::mutex[]> mutexes_;
#endif
public:
    static constexpr size_t VAL_PER_REGISTER = 64 / (fpsize + ctrsize);
    using hash_type = Hasher;
    // Constructor
    // Note: HeavyKeeper paper suggests 1.08, but that seems extreme.
    HeavyKeeper(size_t requested_size, size_t subtables): HeavyKeeper(requested_size, subtables, 1.08) {}
    template<typename...Args>
    HeavyKeeper(size_t requested_size, size_t subtables, double pdec, Args &&...args):
        pol_(requested_size), nh_(subtables),
        data_((pol_.nelem() * subtables + (VAL_PER_REGISTER - 1)) / VAL_PER_REGISTER),
        hasher_(std::forward<Args>(args)...),
        b_(pdec), n_updates_(0)
    {
        assert(subtables);
#if SKETCH_THREADSAFE
        mutexes_.reset(new std::mutex[subtables]);
#endif
        PREC_REQ(pdec >= 1., std::string("pdec is not valid (>= 1.). Value: ") + std::to_string(pdec));
        PREC_REQ(data_.size() > 0, "HeavyKeeper must be greater than 0 in size");
    }

    HeavyKeeper(const HeavyKeeper &o): pol_(o.pol_), nh_(o.nh_), data_(o.data_), hasher_(o.hasher_), b_(o.b_), n_updates_(o.n_updates_)
#if SKETCH_THREADSAFE
        , mutexes_(new std::mutex[o.nh_])
#endif
    {
    }
    HeavyKeeper& operator=(const HeavyKeeper &o) {
        pol_ = o.pol_;
        nh_ = o.nh_;
        data_ = o.data_;
        hasher_ = o.hasher_;
        b_ = o.b_;
        n_updates_ = o.n_updates_;
        return *this;
    }
    HeavyKeeper(HeavyKeeper &&o)      = default;
    HeavyKeeper& operator=(HeavyKeeper &&o)      = default;


    // utilities
    template<typename T, typename=std::enable_if_t<!std::is_same<T, uint64_t>::value>>
    uint64_t hash(const T &x) const {return hasher_(x);}
    uint64_t hash(uint64_t x) const {return hasher_(x);}
    uint64_t hash(uint32_t x) const {return hasher_(x);}
    uint64_t hash(int64_t x) const {return hasher_(uint64_t(x));}
    uint64_t hash(int32_t x) const {return hasher_(uint32_t(x));}

    void clear() {
        std::memset(data_.data(), 0, sizeof(data_[0]) * data_.size());
    }

    static constexpr uint64_t sig_size = fpsize + ctrsize;
    static constexpr uint64_t sig_mask         = bitmask(sig_size);
    static constexpr uint64_t sig_mask_at_pos(size_t i) {
        return sig_mask << (i % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER);
    }
    static_assert(sig_size <= 64, "For sig_mask to be greater than 64, we'd need to use a larger value type.");
    static constexpr uint64_t fingerprint_mask = bitmask(fpsize) << ctrsize;
    static constexpr uint64_t count_mask       = bitmask(ctrsize);

    decoded_register decode(uint64_t x) const {
        decoded_register ret;
        ret.count() = x & count_mask;
        ret.fp()    = (x & fingerprint_mask) >> ctrsize;
        return ret;
    }
    uint64_t encode(uint64_t count, uint64_t fp) const {
        assert(fp < (1ull << fpsize));
        assert(count < (1ull << ctrsize));
        return count | (fp << ctrsize);
    }
    uint64_t encode(decoded_register reg) const {
        return encode(reg.count(), reg.fp());
    }

    uint64_t from_index(size_t i, size_t subidx) const {
        auto dataptr = data_.data() + (subidx * pol_.nelem() / VAL_PER_REGISTER);
        assert(dataptr < &data_[data_.size()] || !std::fprintf(stderr, "subidx: %zu. nelem: %zu. data size: %zu\n", subidx, pol_.nelem(), data_.size()));
        auto pos = pol_.mod(i);
        uint64_t value = dataptr[pos / VAL_PER_REGISTER];
        CONST_IF(VAL_PER_REGISTER > 1) {
            value = (value >> ((pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER))) & sig_mask;
        }
        return value;
    }

    void store(size_t pos, size_t subidx, uint64_t fp, uint64_t count) {
#if SKETCH_THREADSAFE
        std::unique_lock<std::mutex> lock(mutexes_[subidx]);
#endif
        assert(subidx < nh_);
        auto dataptr = data_.data() + (subidx * pol_.nelem() / VAL_PER_REGISTER);
        assert(dataptr < data_.data() + data_.size());
        uint64_t to_insert = encode(count, fp);
        CONST_IF(VAL_PER_REGISTER == 1) {
            dataptr[pos] = to_insert;
            return;
        }
        size_t shift = ((pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER));
        to_insert <<= shift;
        auto &r = dataptr[pos / VAL_PER_REGISTER];
        r &= ~(bitmask(64 / VAL_PER_REGISTER) << shift);
        r |= to_insert;
    }
    bool random_sample(size_t count) {
        static thread_local std::uniform_real_distribution<double> gen;
        static thread_local tsg::ThreadSeededGen<RNG> rng;
        auto limit = std::pow(b_, -ssize_t(count));
        auto val = gen(rng);
#if VERBOSE_AF
        std::fprintf(stderr, "limit: %f. count: %zu. val: %f, pass: %s\n", limit, count, val, val <= limit ? "true": "false");
#endif
        return count && val <= limit;
    }
    static constexpr uint64_t max_fp() {return ((1ull << fpsize) - 1);}
    template<typename T2>
    void divmod(uint64_t x, T2 &pos, T2 &fp) const {
        pos = pol_.mod(x);
        fp = pol_.div(x) & max_fp();
    }
    void display_encoded_value(uint64_t x) {
        auto dec = decode(x);
        std::fprintf(stderr, "dec.count() = %zu. dec.fp() (hash) = %zu\n", dec.count(), dec.fp());
    }
    template<typename T>
    uint64_t addh(const T &x) {
        return add(hash(x));
    }
//protected:
    uint64_t add(uint64_t x) {
        __sync_fetch_and_add(&n_updates_, 1);
        uint64_t maxv = 0;
        unsigned i = 0;
        FOREVER {
            size_t pos, newfp;
            divmod(x, pos, newfp);
            decoded_register vals = decode(from_index(pos, i));
            auto count = vals.count();
            auto fp = vals.fp();
            assert(encode(count, fp) == from_index(pos, i));
            assert(decode(encode(count, fp)).first == count);
            assert(decode(encode(count, fp)).second == fp);
            if(count == 0) {
                store(pos, i, newfp, 1);
                maxv += maxv == 0;
            } else if(fp == newfp) {
                count += count < count_mask;
                store(pos, i, newfp, count);
                maxv = std::max(maxv, uint64_t(count));
            } else {
                if(random_sample(count)) {
                    if(--count == 0) {
                        store(pos, i, newfp, 1);
                        maxv = std::max(maxv, uint64_t(1));
                    } else {
                        store(pos, i, fp, count);
                    }
                }
                assert(decode(from_index(pos, i)).first != 0);
            }
            if(++i == nh_) break;
            wy::wyhash64_stateless(&x);
        }
        return maxv;
    }
//public:
    template<typename T>
    uint64_t queryh(const T &x) const {
        return query(hash(x));
    }
    uint64_t query(uint64_t x) const {
        uint64_t ret = 0;
        unsigned i = 0;
        FOREVER {
            size_t pos, newfp;
            divmod(x, pos, newfp);
            auto p = decode(from_index(pos, i));
            auto count = p.count(), fp = p.fp();
            if(fp == newfp) ret = std::max(ret, count);
            if(++i == nh_) break;
            wy::wyhash64_stateless(&x);
        }
        return ret;
    }
    template<typename T>
    uint64_t est_count(const T &x) {
        return queryh(x);
    }
    HeavyKeeper &operator|=(const HeavyKeeper &o) {
        size_t data_index = 0;
        for(size_t subidx = 0; subidx < nh_; ++subidx) {
            for(size_t i = 0; i < o.size(); ++i) {
                uint64_t lv = data_[i], rv = o.data_[i];
                for(size_t j = 0; j < VAL_PER_REGISTER; ++j, ++data_index) {
                    auto mask = sig_mask_at_pos(j);
                    uint64_t llv = lv & mask, lrv = rv & mask;
                    auto llp = decode(llv), lrp = decode(lrv);
                    auto lc = llp.first, lpf = llp.second,
                         rc = lrp.first, rf = lrp.seccond;
                    if(lpf == rf) { // Easily predicted branch first
                        auto newc = lc + rc;
                        if(VAL_PER_REGISTER != 1) store(data_index, subidx, lpf, newc);
                        else {
                            data_[data_index] = (lpf << ctrsize) | (lc + rc);
                            assert(newc >= lc && newc >= rc);
                        }
                    } else {
                        auto newc = std::max(lc, rc), newsub = std::min(lc, rc);
                        newc -= newsub; // Not rigorous, but an exact formula would be effort/perhaps costly.
                        if(newc == 0)
                            store(data_index, subidx, 0, 0);
                        else
                            store(data_index, subidx, newc == lc ? lpf: rf, newc);
                    }
                }
            }
        }
        return *this;
    }
    HeavyKeeper &operator+=(const HeavyKeeper &o) {return *this |= o;}
    HeavyKeeper operator+(const HeavyKeeper &x) const {
        HeavyKeeper cpy(*this);
        cpy += x;
        return cpy;
    }
    uint64_t n_updates() const {return n_updates_;}
    size_t size() const {return data_.size();}
};
template<typename T>
struct is_hk: std::false_type {};

template<size_t fpsize, size_t ctrsize, typename Hasher, typename Policy, typename RNG, typename Allocator>
struct is_hk<HeavyKeeper<fpsize,ctrsize,Hasher,Policy,RNG,Allocator>>: std::true_type {};

#if CXX20_CONCEPTS
template<typename T>
concept BigGoalie = is_hk<T>::value;
template<typename T>
concept Integral = std::is_integral<T>::value;
#endif

template<
         typename HKType,
         typename ValueType,
         typename Hasher=std::hash<ValueType>,
         typename Allocator=sketch::Allocator<ValueType>,
         typename HashSetFingerprint=uint64_t,
         typename=typename std::enable_if_t<is_hk<HKType>::value>,
         typename=typename std::enable_if_t<std::is_integral<HashSetFingerprint>::value>
>
class HeavyKeeperHeap {
protected:
    HKType hk_;
    std::vector<ValueType, Allocator> heap_;
    ska::flat_hash_set<HashSetFingerprint> hashes_;
    size_t max_heap_size_;
    struct Comparator {
        const HKType &hk_;
        Comparator(const HKType &h): hk_(h) {}
        bool operator()(const ValueType &x, const ValueType &y) const {
            auto xy = hk_.hash(x), yh = hk_.hash(y);
            auto v1 = hk_.query(xy), v2 = hk_.query(yh);
            return std::tie(v1, yh) > std::tie(v2, xy);
            // Inverted: selects maximum counts with minimum hash values
        }
    };

public:
    using hashfp_t = HashSetFingerprint;
    using keeper_t = HKType;
    using hasher_t = Hasher;
    using value_type = ValueType;
    HeavyKeeperHeap(size_t heap_size, HKType &&hvk): hk_(std::move(hvk)), max_heap_size_(heap_size) {
        heap_.reserve(heap_size);
    }
    auto &top() {return heap_.front();}
    const auto &top() const {return heap_.front();}
    auto hash(const value_type &x) const {return hk_.hash(x);}
    void addh(const value_type &x) {
        value_type t = x;
        addh(std::move(t));
    }
    auto est_count(const value_type &x) {return hk_.queryh(x);}
    uint64_t addh(value_type &&x) {
        const auto hv = hk_.hash(x);
        auto old_count = hk_.query(hv);
        if(hashes_.find(hv) != hashes_.end()) {
            hk_.add(hv);
        } else {
            if(heap_.size() < max_heap_size_) {
                heap_.emplace_back(std::move(x));
                hk_.add(hv);
                hashes_.emplace(hv);
                std::push_heap(heap_.begin(), heap_.end(), Comparator(hk_));
            } else {
                auto yhv = hk_.hash(top());
                const auto cmpcount = hk_.query(yhv);
                if(std::tie(old_count, yhv) > std::tie(cmpcount, hv)) {
                    assert(old_count >= cmpcount || hash(top()) > hash(x));

                    // 3.4:Optimization 1 -- detecting fingerprint collisions
                    if(old_count > cmpcount + 1) {
                        old_count = 0;
                    } else if(old_count == cmpcount + 1 || old_count == cmpcount) {
                        // 3.4:Optimization 2 -- selective increment
                        // Note: we will replace the top of the heap even if
                        // cmpcount is nmin if the new hashvalue is smaller,
                        // as the items themselves are equivalent
                        hk_.add(hv);

                        std::pop_heap(heap_.begin(), heap_.end(), Comparator(hk_));
                        hashes_.erase(hash(heap_.back()));
                        hashes_.emplace(hv);
                        heap_.back() = std::move(x);
                        assert(hashes_.size() == heap_.size());
                        std::push_heap(heap_.begin(), heap_.end(), Comparator(hk_));
                    }
                }
            }
        }
        return old_count;
    }
    auto begin() {return heap_.begin();}
    auto begin() const {return heap_.begin();}
    auto end() {return heap_.end();}
    auto end() const {return heap_.end();}
    auto to_container() const {
        std::vector<value_type, Allocator> ret = heap_;
        sort::default_sort(ret.begin(), ret.end(), Comparator(hk_));
        std::vector<hashfp_t, common::Allocator<hashfp_t>> hashfps;
        std::vector<size_t, common::Allocator<size_t>> counts;
        hashfps.reserve(heap_.size());
        counts.reserve(heap_.size());
        for(const auto &x: ret) {
            auto hv = hash(x);
            hashfps.push_back(hv);
            counts.push_back(hk_.query(hv));
        }
        return std::make_tuple(std::move(ret), std::move(counts), std::move(hashfps));
    }
    auto finalize() const & {
        auto ret = to_container();
        return ret;
    }
    auto finalize() && { // Consumes and destroys
        sort::default_sort(heap_.begin(), heap_.end(), Comparator(hk_));
        std::vector<hashfp_t, common::Allocator<hashfp_t>> hashfps;
        std::vector<size_t, common::Allocator<size_t>> counts;
        hashfps.reserve(heap_.size());
        counts.reserve(heap_.size());
        for(const auto &x: heap_) {
            auto hv = hash(x);
            hashfps.push_back(hv);
            counts.push_back(hk_.query(hv));
        }
        hashes_.clear();
        hk_.clear();
        return std::make_tuple(std::move(heap_), std::move(counts), std::move(hashfps));
    }
};

template<typename...Args>
struct HeavyKeeperHeavyHitters: public HeavyKeeperHeap<Args...> {
    using super = HeavyKeeperHeap<Args...>;
    using Comparator = typename super::Comparator;
    double theta_;
    HeavyKeeperHeavyHitters(double theta, size_t maxheapsize, typename super::keeper_t &&hk): super(maxheapsize, std::move(hk)), theta_(theta) {
        PREC_REQ(this->hk_.size(), "Must have non-empty heavykeeper");
        PREC_REQ(theta_ < 1. && theta_ > 0., "theta must be [0, 1)");
    }
    auto to_container() const {
        size_t minsize = theta_ * this->n_updates();
        std::vector<typename super::value_type, Allocator<typename super::value_type>> ret = this->heap_;

        std::vector<typename super::hashfp_t, common::Allocator<typename super::hashfp_t>> hashfps;
        std::vector<size_t, common::Allocator<size_t>> counts;
        for(const auto &x: ret) {
            auto hv = super::hash(x);
            auto qest = this->hk_.query(hv);
            if(qest < minsize) break;
            hashfps.push_back(hv);
            counts.push_back(qest);
        }
        if(counts.size() != ret.size())
            ret.resize(counts.size());
        return std::make_tuple(std::move(ret), std::move(counts), std::move(hashfps));
    }
    uint64_t addh(const typename super::value_type &x) {
        auto tmp(x);
        return addh(std::move(tmp));
    }
    uint64_t addh(typename super::value_type &&x) {
        const auto hv = this->hk_.hash(x);
        auto old_count = this->hk_.query(hv);
        if(this->hashes_.find(hv) != this->hashes_.end()) {
            this->hk_.add(hv);
        } else {
            auto nmin = this->hk_.query(this->hk_.hash(this->top()));
            if(this->hk_.n_updates() > 1000 && nmin >= std::pow(theta_, 2) * this->hk_.n_updates()) {// Expand if you feel like it
                this->max_heap_size_ += std::sqrt(this->max_heap_size_);
                this->heap_.reserve(this->max_heap_size_);
            }
            if(this->heap_.size() < this->max_heap_size_) {
                this->heap_.emplace_back(std::move(x));
                this->hk_.add(hv);
                this->hashes_.emplace(hv);
                std::push_heap(this->heap_.begin(), this->heap_.end(), Comparator(this->hk_));
            } else {
                auto yhv = this->hk_.hash(this->top());
                const auto cmpcount = this->hk_.query(yhv);
                if(std::tie(old_count, yhv) > std::tie(cmpcount, hv)) {
                    assert(old_count >= cmpcount || this->hk_.hash(this->top()) > this->hk_.hash(x));

                    // 3.4:Optimization 1 -- detecting fingerprint collisions
                    if(old_count > cmpcount + 1) {
                        old_count = 0;
                    }
                    else if(old_count == cmpcount + 1 || old_count == cmpcount) {
                        // 3.4:Optimization 2 -- selective increment
                        // Note: we will replace the top of the heap even if
                        // cmpcount is nmin if the new hashvalue is smaller,
                        // as the items themselves are equivalent
                        this->hk_.add(hv);
                        std::pop_heap(this->heap_.begin(), this->heap_.end(), Comparator(this->hk_));
                        this->hashes_.erase(this->hash(this->heap_.back()));
                        this->hashes_.emplace(hv);
                        this->heap_.back() = std::move(x);
                        assert(this->hashes_.size() == this->heap_.size());
                        std::push_heap(this->heap_.begin(), this->heap_.end(), Comparator(this->hk_));
                    }
                }
            }
        }
        return old_count;
    }
};

} // namespace hk

} // namespace sketch
#endif
