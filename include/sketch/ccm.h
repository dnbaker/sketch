#pragma once
#include <ctime>
#include <deque>
#include <queue>
#include "hash.h"
#include "update.h"
#include "median.h"
#include "compact_vector/compact_vector.hpp"
#include "vec/vec.h"


namespace sketch {

inline namespace common { namespace detail {

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}
template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline void zero_memory(compact::ts_vector<T1, BITS, T2, Allocator> &v, size_t newsz=0) {
   std::memset(v.get(), 0, v.bytes()); // zero array
}

} }

inline namespace cm {
using common::detail::tmpbuffer;
using common::Allocator;
using std::int64_t;

#if NOT_THREADSAFE
template<size_t NBITS>
class DefaultStaticCompactVectorType: public ::compact::vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>> {
public:
    DefaultStaticCompactVectorType(size_t nb, size_t nelem): ::compact::vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>>(nelem) {}
};
using DefaultCompactVectorType = ::compact::vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;
#else

using DefaultCompactVectorType = ::compact::ts_vector<uint64_t, 0, uint64_t, Allocator<uint64_t>>;
template<size_t NBITS>
class DefaultStaticCompactVectorType: public ::compact::ts_vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>> {
public:
    DefaultStaticCompactVectorType(size_t nb, size_t nelem): ::compact::ts_vector<uint64_t, NBITS, uint64_t, Allocator<uint64_t>>(nelem) {}
};
#endif

namespace detail {
template<typename T, typename AllocatorType>
static inline double sqrl2(const std::vector<T, AllocatorType> &v, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<double, 8> mem(nhashes);
    double *ptr = mem.get();
#if defined(_BLAZE_MATH_CUSTOMMATRIX_H_)
    blaze::CustomMatrix<T, blaze::aligned, blaze::unpadded> data(v.data(), nhashes, size_t(1) << l2sz);
    for(size_t i = 0; i < data.rows(); ++i) {
        ptr[i] = blaze::norm(row(data, i));
    }
#else
    using VT = typename vec::SIMDTypes<T>::VType;
    using VS = vec::SIMDTypes<T>;
    VT sum = VS::set1(0);
    static constexpr size_t ct = VS::COUNT;
    for(size_t i = 0; i < nhashes; ++i) {
        const T *p1 = &v[i << l2sz], *p2 = &v[(i+1)<<l2sz];
        if(VS::is_aligned) {
            while(p2 - p1 > ct) {
                const auto el = *reinterpret_cast<const VT *>(p1);
                sum = VS::add(sum, VS::mul(el, el));
                p1 += ct;
            }
        } else {
            while(p2 - p1 > ct) {
                const auto el = VT::loadu(p1);
                sum = VS::add(sum, VS::mul(el, el));
                p1 += ct;
            }
        }
        T full_sum = sum.sum();
        while(p1 < p2)
            full_sum += *p1 * *p1, ++p1;
        ptr[i] = full_sum;
    }
#endif
    return std::sqrt(median(ptr, nhashes));
}

template<typename T, typename AllocatorType>
static inline double sqrl2(const std::vector<T, AllocatorType> &v, const std::vector<T, AllocatorType> &v2, uint32_t nhashes, uint32_t l2sz) {
    assert(v.size() == v2.size());
    tmpbuffer<double, 8> mem(nhashes);
    double *ptr = mem.get();
#if defined(_BLAZE_MATH_CUSTOMMATRIX_H_)
    using CM = blaze::CustomMatrix<T, blaze::aligned, blaze::unpadded>;
    const CM lv(v.data(), nhashes, size_t(1) << l2sz);
    const CM rv(v2.data(), nhashes, size_t(1) << l2sz);
    for(auto i = 0u; i < lv.rows(); ++i) {
        ptr[i] = blaze::norm(row(lv, i) * row(rv, i)); // Elementwise multiplication
    }
#else
    using VT = typename vec::SIMDTypes<T>::VType;
    using VS = vec::SIMDTypes<T>;
    VT sum = VS::set1(0);
    for(size_t i = 0; i < nhashes; ++i) {
        auto p1 = &v[i << l2sz], p2 = &v2[i << l2sz], p1e = &v[(i + 1) << l2sz];
        T full_sum = std::abs(*p1++ * *p2++);
        while(p1 != p1e) full_sum += std::pow(*p1++ * *p2++, 2);
        ptr[i] = std::sqrt(full_sum);
    }
#endif
    return median(ptr, nhashes);
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline double sqrl2(const compact::vector<T1, BITS, T2, Allocator> &v, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<double, 8> mem(nhashes);
    double *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        double sum = 0;
        while(start != end) {
            int64_t val = v[start++];
            sum += val * val;
        }
        ptr[i] = std::sqrt(sum);
    }
    return median(ptr, nhashes);
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline double sqrl2(const compact::vector<T1, BITS, T2, Allocator> &v, const compact::vector<T1, BITS, T2, Allocator> &v2, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<double, 8> mem(nhashes);
    double *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        double sum = 0;
        while(start != end) {
            sum += v[start] * v2[start];
            ++start;
        }
        ptr[i] = std::sqrt(sum);
    }
    return median(ptr, nhashes);
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline double sqrl2(const compact::ts_vector<T1, BITS, T2, Allocator> &v, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<double> mem(nhashes);
    double *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        double sum = 0;
        do {
            int64_t val = v[start++];
            sum += val * val;
        } while(start != end);
        ptr[i] = std::sqrt(sum);
    }
    return median(ptr, nhashes);
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline double sqrl2(const compact::ts_vector<T1, BITS, T2, Allocator> &v, const compact::ts_vector<T1, BITS, T2, Allocator> &v2, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<double, 8> mem(nhashes);
    double *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        double sum = 0;
        do {
            sum += v[start] * v2[start];
            ++start;
        } while(start != end);
        ptr[i] = std::sqrt(sum);
    }
    return median(ptr, nhashes);
}

template<typename T>
struct IndexedValue {
    using Type = typename std::decay_t<decltype(*(std::declval<T>().cbegin()))>;
};
} // namespace detail


template<typename UpdateStrategy=update::Increment,
         typename VectorType=DefaultCompactVectorType,
         typename HashStruct=WangHash,
         bool conservative_update=true>
class ccmbase_t {
    static_assert(!std::is_same<UpdateStrategy, update::CountSketch>::value || std::is_signed<typename detail::IndexedValue<VectorType>::Type>::value,
                  "If CountSketch is used, value must be signed.");

protected:
    VectorType        data_;
    UpdateStrategy updater_;
    unsigned nhashes_;
    unsigned l2sz_:16;
    unsigned nbits_:16;
    HashStruct hf_;
    uint64_t mask_;
    uint64_t subtbl_sz_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;

public:
    using counter_register_type = typename std::decay<decltype(data_[0])>::type;
    static constexpr bool supports_deletion() {
        return !conservative_update;
    }
    size_t size() const {return data_.size();}
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this),
                              seeds_.size() * sizeof(seeds_[0]) + data_.bytes());
    }
    size_t seeds_size() const {return seeds_.size();}
    void clear() {
        common::detail::zero_memory(data_, ilog2(subtbl_sz_));
    }
    double l2est() const {
        return detail::sqrl2(data_, nhashes_, l2sz_);
    }
    double join_size_l2est(const ccmbase_t &o) const {
        PREC_REQ(o.size() == this->size(), "tables must have the same size\n");
        return detail::sqrl2(data_, o.data_, nhashes_, l2sz_);
    }
    template<typename Func>
    void for_each_register(const Func &func) {
        for(size_t i = 0; i < data_.size(); ++i)
            func(data_[i]);
    }
    template<typename Func>
    void for_each_register(const Func &func) const {
        for(size_t i = 0; i < data_.size(); ++i)
            func(data_[i]);
    }
    ccmbase_t(ccmbase_t &&o): data_(std::move(o.data_)), updater_(std::move(o.updater_)), nhashes_(o.nhashes_), l2sz_(o.l2sz_),
                              nbits_(o.nbits_), hf_(std::move(o.hf_)), mask_(o.mask_), subtbl_sz_(o.subtbl_sz_), seeds_(std::move(o.seeds_))
    {
    }
    ccmbase_t(const ccmbase_t &o) = default;
    //ccmbase_t(ccmbase_t &&o) = default;
    template<typename... Args>
    ccmbase_t(int nbits, int l2sz, int64_t nhashes=4, uint64_t seed=0, Args &&... args):
            data_(nbits, nhashes << l2sz),
            updater_(seed + l2sz * nbits * nhashes),
            nhashes_(nhashes), l2sz_(l2sz),
            nbits_(nbits), hf_(std::forward<Args>(args)...),
            mask_((1ull << l2sz) - 1),
            subtbl_sz_(1ull << l2sz)
    {
        if(HEDLEY_UNLIKELY(nbits < 0)) throw std::runtime_error("Number of bits cannot be negative.");
        if(HEDLEY_UNLIKELY(l2sz < 0)) throw std::runtime_error("l2sz cannot be negative.");
        if(HEDLEY_UNLIKELY(nhashes < 0)) throw std::runtime_error("nhashes cannot be negative.");
        std::mt19937_64 mt(seed + 4);
        while(seeds_.size() < static_cast<unsigned>(nhashes)) seeds_.emplace_back(mt());
        clear();
        VERBOSE_ONLY(std::fprintf(stderr, "data size: %zu. nbits per entry: %u\n", data_.size(), nbits);)
    }
    VectorType &ref() {return data_;}
    template<typename T, typename=std::enable_if_t<!std::is_arithmetic<T>::value>>
    auto addh(const T &x) {
        uint64_t hv = hf_(x);
        return add(hv);
    }
    auto addh(uint64_t val) {return add(val);}
    auto addh_val(uint64_t val) {return add(val);}
    template<typename T>
    T hash(T val) const {
        return hf_(val);
    }
    uint64_t subhash(uint64_t val, uint64_t seedind) const {
        return hash(((val ^ seeds_[seedind]) & mask_) + (val & mask_));
    }
    double wj_est(const ccmbase_t &o) const {
        std::fprintf(stderr, "[%s:%s:%d] Warning: This function should not be used.\n");
#if WJMETH0
        tmpbuffer<double> counts(nhashes_);
        auto p = counts.get();
#elif MINMETH
        double minest = std::numeric_limits<double>::max();
#else
        double minest = 0.;
#endif
        for(size_t i = 0; i < nhashes_; ++i) {
            uint64_t n = 0, d = 0;
            uint64_t d1, d2;
            for(size_t j = (i << l2sz_), e = (i + 1) << l2sz_; j < e; ++j) {
                d1 = data_[j] > 0 ? data_[i]: -data_[i], d2 = o.data_[j] >0 ? o.data_[j]: -o.data_[j];
                n += std::min(d1, d2); d += std::max(d1, d2);
            }
#if WJMETH0
            *p++ = double(n) / d;
#elif MINMETH
            minest = std::min(double(n) / d, minest);
#else
            minest += double(n) / d;
#endif
        }
#if WJMETH0
        return median(counts.get(), nhashes_);
#elif MINMETH
        return minest;
#else
        return minest / nhashes_;
#endif
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
        return true;
    }
    using Space = vec::SIMDTypes<uint64_t>;
    uint32_t may_contain(Space::VType val) const {
        throw std::runtime_error("This needs to be rewritten after subhash refactoring.");
        return true;
    }
    static constexpr bool is_increment = std::is_same<UpdateStrategy, update::Increment>::value;
    ssize_t add(const uint64_t val) {
        unsigned nhdone = 0;
        ssize_t ret;
        CONST_IF(conservative_update) {
            std::vector<uint64_t> indices, best_indices;
            indices.reserve(nhashes_);
            //std::fprintf(stderr, "Doing SIMD stuff\n");
            //std::fprintf(stderr, "Doing Leftover stuff\n");
            while(nhdone < nhashes_) {
                assert(seeds_.data());
                uint64_t hv = hash(val, nhdone);
                auto index = subtbl_sz_ * nhdone++ + (hv & mask_);
                indices.push_back(index);
            }
#if 0
            if(val == 137) {
                for(const auto v: indices) std::fprintf(stderr, "index for 137: %u\n", unsigned(v));
            }
#endif
            best_indices.push_back(indices[0]);
            ssize_t minval = data_.operator[](indices[0]);
            for(size_t i(1); i < indices.size(); ++i) {
                unsigned score;
                if((score = data_.operator[](indices[i])) == minval) {
                    best_indices.push_back(indices[i]);
                } else if(score < minval) {
                    best_indices.clear();
                    best_indices.push_back(indices[i]);
                    minval = score;
                }
            }
            //std::fprintf(stderr, "Now update\n");
            updater_(best_indices, data_, nbits_);
            ret = minval;
            //std::fprintf(stderr, "Now updated\n");
        } else { // not conservative update. This means we support deletions
            ret = std::numeric_limits<decltype(ret)>::max();
            const auto maxv = 1ull << nbits_;
            std::vector<uint64_t> indices{0};
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val, nhdone);
                auto ind = (hv & mask_) + subtbl_sz_ * nhdone++;
                indices[0] = ind;
                updater_(indices, data_, maxv);
                ret = std::min(ret, ssize_t(data_[ind]));
            }
        }
        return ret + is_increment;
    }
    auto hash(uint64_t x, unsigned index) const {
        return hash(x ^ seeds_[index]);
    }
    uint64_t est_count(uint64_t val) const {
        uint64_t ret = std::numeric_limits<uint64_t>::max();
        for(unsigned i = 0; i < nhashes_; ++i) {
            auto hv = hash(val, i);
            ret = std::min(ret, uint64_t(data_[(hv & mask_) + subtbl_sz_ * i]));
        }
        return updater_.est_count(ret);
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

template<typename HashStruct=WangHash, typename CounterType=int32_t, typename=typename std::enable_if<std::is_signed<CounterType>::value>::type>
class csbase_t {
    /*
     * Commentary: because of chance, one can end up with a negative number as an estimate.
     * Either the item collided with another item which was quite large and it was outweighed
     * or it and others in the bucket were not heavy enough and by chance it did
     * not weigh over the other items with the opposite sign. Treat these as 0s.
    */
    std::vector<CounterType, Allocator<CounterType>> core_;
    uint32_t np_, nh_;
    const HashStruct hf_;
    uint64_t mask_;
    std::vector<CounterType, Allocator<CounterType>> seeds_;
    uint64_t seedseed_;

    CounterType       *data()       {return core_.data();}
    const CounterType *data() const {return core_.data();}
public:
    template<typename...Args>
    csbase_t(unsigned np, unsigned nh=1, unsigned seedseed=137, Args &&...args):
        core_(uint64_t(nh) << np), np_(np), nh_(nh), hf_(std::forward<Args>(args)...),
        mask_((1ull << np_) - 1),
        seeds_(nh_),
        seedseed_(seedseed)
    {
        //DEPRECATION_WARNING("csbase_t will be deprecated in favor of cs4wbase_t moving forward.");
        DefaultRNGType gen(np + nh + seedseed);
        for(auto &el: seeds_) el = gen();
    }
    double l2est() const {
        return sqrl2(core_, nh_, np_);
    }
    CounterType addh_val(uint64_t val) {
        std::vector<CounterType> counts(nh_);
        auto cptr = counts.data();
        uint64_t v = hf_(val);
        cptr[0] = add(v, 0);
        for(unsigned ind = 1;ind < nh_; ++ind)
            cptr[ind] = add(hf_(seeds_[ind] ^ val), ind);
        return median(cptr, nh_);
    }
    template<typename T>
    CounterType addh_val(const T &x) {
        uint64_t hv = hf_(x);
        return addh_val(hv);
    }
    template<typename T, typename=std::enable_if_t<!std::is_arithmetic<T>::value>>
    void addh(const T &x) {
        uint64_t hv = hf_(x);
        addh(hv);
    }
    void addh(uint64_t val) {
        uint64_t v = hf_(val);
        auto it = seeds_.begin();
        add(v, 0);
        unsigned ind = 1;
        while(ind < nh_)
            add(hf_(*it++ ^ val), ind++);
    }
    template<typename Func>
    void for_each_register(const Func &func) {
        for(size_t i = 0; i < core_.size(); ++i)
            func(core_[i]);
    }
    template<typename Func>
    void for_each_register(const Func &func) const {
        for(size_t i = 0; i < core_.size(); ++i)
            func(core_[i]);
    }
    void subh(uint64_t val) {
        uint64_t v = hf_(val);
        auto it = seeds_.begin();
        sub(v, 0);
        unsigned ind = 1;
        while(ind < nh_)
            sub(hf_(*it++ ^ val), ind++);
    }
    auto subh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        uint64_t v = hf_(val);
        auto it = seeds_.begin();
        *cptr++ = sub(v, 0);
        unsigned ind = 1;
        while(ind < nh_)
            *cptr++ = sub(hf_(*it++ ^ val), ind++);
        return median(counts.get(), nh_);
    }
    INLINE size_t index(uint64_t hv, unsigned subidx) const noexcept {
        return (hv & mask_) + (subidx << np_);
    }
    INLINE auto add(uint64_t hv, unsigned subidx) noexcept {
#if !NDEBUG
        at_pos(hv, subidx) += sign(hv);
        return at_pos(hv, subidx);
#else
        return at_pos(hv, subidx) += sign(hv);
#endif
    }
    INLINE auto vatpos(uint64_t hv, unsigned subidx) const noexcept {
        return at_pos(hv, subidx) * sign(hv);
    }
    INLINE auto sub(uint64_t hv, unsigned subidx) noexcept {
        return at_pos(hv, subidx) -= sign(hv);
    }
    INLINE auto &at_pos(uint64_t hv, unsigned subidx) noexcept {
        assert(index(hv, subidx) < core_.size() || !std::fprintf(stderr, "hv & mask_: %zu. subidx %d. np: %d. nh: %d. size: %zu\n", size_t(hv&mask_), subidx, np_, nh_, core_.size()));
        return core_[index(hv, subidx)];
    }
    INLINE auto at_pos(uint64_t hv, unsigned subidx) const noexcept {
        assert((hv & mask_) + (subidx << np_) < core_.size());
        return core_[index(hv, subidx)];
    }
    INLINE int sign(uint64_t hv) const {
        return hv & (1ul << np_) ? 1: -1;
    }
    CounterType est_count(uint64_t val) const {
        common::detail::tmpbuffer<CounterType> mem(nh_);
        CounterType *ptr = mem.get();
        uint64_t v = hf_(val);
        auto it = seeds_.begin();
        *ptr++ = vatpos(v, 0);
        for(unsigned ind = 1;ind < nh_; ++it, ++ptr, ++ind) {
            auto hv = hf_(*it ^ val);
            *ptr = vatpos(hv, ind);
        }
        //std::for_each(mem.get(), mem.get() + nh_, [p=mem.get()](const auto &x) {std::fprintf(stderr, "Count estimate for ind %zd is %u\n", &x - p, int32_t(x));});
        ///
        return median(mem.get(), nh_);
    }
    csbase_t &operator+=(const csbase_t &o) {
        precondition_require(o.size() == this->size(), "tables must have the same size\n");
        using VS = vec::SIMDTypes<CounterType>;
        using VT = typename VS::VType;
        VT sum = VS::set1(0);
        static constexpr uint32_t lim = ilog2(VS::COUNT);
        if(np_ > lim && VS::aligned(o.data()) && VS::aligned(data())) {
            size_t i = 0;
            do {
                VS::store(data() + i, VS::add(VS::load(o.data() + i), VS::load(data() + i)));
                i += VS::COUNT;
            } while(i < core_.size());
        } else {
            for(size_t i = 0; i < core_.size(); ++i)
                core_[i] += o.core_[i];
        }
        return *this;
    }
    csbase_t operator+(const csbase_t &o) const {
        auto tmp = *this;
        tmp += o;
        return tmp;
    }
    csbase_t &operator-=(const csbase_t &o) {
        // TODO: SIMD optimize (but how often is this needed?)
        PREC_REQ(core_.size() == o.core_.size(), "mismatched sizes");
        for(size_t i = 0; i < core_.size(); ++i)
            core_[i] -= o.core_[i];
        return *this;
    }
    csbase_t operator-(const csbase_t &o) const {
        auto tmp = *this;
        tmp -= o;
        return tmp;
    }
    csbase_t fold(int n=1) const {
        PREC_REQ(n >= 1, "n < 0 is meaningless and n = 1 uses a copy instead.");
        PREC_REQ(n <= np_, "Can't fold to less than 1");
        csbase_t ret(np_ - n, nh_, seedseed_);
        schism::Schismatic<uint32_t> div(core_.size());
        // More cache-efficient way to traverse than iterating over the final sketch
        for(size_t i = 0; i < core_.size(); ++i)
            ret.core_[div.mod(i)] += core_[i];
        return ret;
    }
};


template<typename CounterType=int32_t, typename HasherSetType=KWiseHasherSet<4>>
class cs4wbase_t {
    /*
     * Commentary: because of chance, one can end up with a negative number as an estimate.
     * Either the item collided with another item which was quite large and it was outweighed
     * or it and others in the bucket were not heavy enough and by chance it did
     * not weigh over the other items with the opposite sign. Treat these as 0s.
    */
    static_assert(std::is_signed<CounterType>::value, "CounterType must be signed");

    // Note: in order to hash other types, you'd need to subclass the HasherSet
    // class in hash.h and provide an overload for your type, or hash the items
    // yourself and insert them first.
    // This is more cumbersome.

    std::vector<CounterType, Allocator<CounterType>> core_;
    uint32_t np_, nh_;
    uint64_t mask_;
    uint64_t seedseed_;
    const HasherSetType hf_;
    CounterType       *data()       {return core_.data();}
    const CounterType *data() const {return core_.data();}
    // TODO: use a simpler hash function under the assumption that it doesn't matter?

    size_t size() const {return core_.size();}
public:
    cs4wbase_t(unsigned np, unsigned nh=1, unsigned seedseed=137):
        np_(np),
        nh_(nh),
        mask_((1ull << np_) - 1),
        seedseed_(seedseed),
        hf_(nh_, seedseed)
    {
        assert(hf_.size() == nh_);
        nh_ += (nh % 2 == 0);
        core_.resize(nh_ << np_);
        POST_REQ(core_.size() == (nh_ << np_), "core must be properly sized");
    }
    double l2est() const {
        return sqrl2(core_, nh_, np_);
    }
    CounterType addh_val(uint64_t val) {
        std::vector<CounterType> counts(nh_);
        auto cptr = counts.data();
        for(unsigned added = 0; added < nh_; ++added)
            cptr[added] = add(val, added);
        return median(cptr, nh_);
    }
    auto addh(uint64_t val) {return addh_val(val);}
    auto nhashes() const {return nh_;}
    auto p() const {return np_;}
    template<typename T, typename=std::enable_if_t<std::is_arithmetic<T>::value>>
    auto addh_val(T x) {
        uint64_t hv = hf_(static_cast<uint64_t>(x));
        return addh_val(hv);
    }
    template<typename T, typename=std::enable_if_t<!std::is_arithmetic<T>::value>>
    auto addh_val(const T &x) {
        uint64_t hv = hf_(x);
        return addh_val(hv);
    }
    template<typename T,  typename=std::enable_if_t<std::is_arithmetic<T>::value>>
    auto addh(T x) {return addh_val(static_cast<uint64_t>(x));}
    void subh(uint64_t val) {
        for(unsigned added = 0; added < nh_; ++added)
            sub(val, added);
    }
    auto subh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        for(unsigned added = 0; added < nh_; ++added)
            cptr[added] = sub(val, added);
        return median(cptr, nh_);
    }
    INLINE size_t index(uint64_t hv, unsigned subidx) const noexcept {
        return (hv & mask_) + (subidx << np_);
    }
    INLINE auto add(uint64_t hv, unsigned subidx) noexcept {
        hv = hf_(hv, subidx);
        auto &ref = at_pos(hv, subidx);
        if(ref != std::numeric_limits<CounterType>::max()) // easy branch to predict
            ref += sign(hv);
        return ref * sign(hv);
    }
    INLINE auto sub(uint64_t hv, unsigned subidx) noexcept {
        hv = hf_(hv, subidx);
        auto &ref = at_pos(hv, subidx);
        if(ref != std::numeric_limits<CounterType>::min()) // easy branch to predict
            ref -= sign(hv);
        return ref * sign(hv);
    }
    CounterType update(uint64_t val, const double increment=1.) {
        std::vector<CounterType> counts(nh_);
        auto cptr = counts.data();
        for(unsigned added = 0; added < nh_; ++added) {
            auto hv = hf_(val, added);
            auto &ref = at_pos(hv, added);
            auto shv = sign(hv);
            ref += increment * shv;
            cptr[added] = shv * ref;
        }
        return median(cptr, nh_);
    }
    INLINE auto &at_pos(uint64_t hv, unsigned subidx) noexcept {
        assert(index(hv, subidx) < core_.size() || !std::fprintf(stderr, "hv & mask_: %zu. subidx %d. np: %d. nh: %d. size: %zu\n", size_t(hv&mask_), subidx, np_, nh_, core_.size()));
        return core_[index(hv, subidx)];
    }
    INLINE auto at_pos(uint64_t hv, unsigned subidx) const noexcept {
        assert((hv & mask_) + (subidx << np_) < core_.size());
        return core_[index(hv, subidx)];
    }
    double dot_product(const cs4wbase_t &o) const {
        auto myp = data(), op = o.data();
        common::detail::tmpbuffer<CounterType> mem(nh_);
        auto memp = mem.get();
        const size_t tsz = (1ull << np_);
        double ret = 0.;
        for(unsigned i = 0u; i < nh_; ++i) {
            auto lmyp = myp + tsz, lop = op + tsz;
#if _OPENMP > 201307L
        #pragma omp simd
#endif
            for(size_t j = 0; j < tsz; ++j)
                ret += lmyp[i] * lop[i];
            memp[i] = ret;
        }
        return median(memp, nh_);
    }
    INLINE int sign(uint64_t hv) const noexcept {
        return hv & (1ul << np_) ? 1: -1;
    }
    using Space = vec::SIMDTypes<uint64_t>;
    INLINE void subh(Space::VType hv) noexcept {
        hv.for_each([&](auto x) {for(size_t i = 0; i < nh_; sub(x, i++));});
    }
    INLINE void addh(Space::VType hv) noexcept {
        hv.for_each([&](auto x) {for(size_t i = 0; i < nh_; add(x, i++));});
    }
    CounterType est_count(uint64_t val) const {
        common::detail::tmpbuffer<CounterType> mem(nh_);
        CounterType *ptr = mem.get(), *p = ptr;
        for(unsigned i = 0; i < nh_; ++i) {
            auto v = hf_(val, i);
            *p++ = at_pos(v, i) * sign(v);
        }
        return median(ptr, nh_);
    }
    cs4wbase_t &operator+=(const cs4wbase_t &o) {
        precondition_require(o.size() == this->size(), "tables must have the same size\n");
        using OT = typename vec::SIMDTypes<CounterType>::Type;
        using VS = vec::SIMDTypes<CounterType>;
        static constexpr uint32_t lim = ilog2(VS::COUNT);
        if(np_ > lim && VS::aligned(o.data()) && VS::aligned(data())) {
            size_t i = 0;
            do {
                VS::store(reinterpret_cast<OT *>(data() + i),
                    VS::add(VS::load(reinterpret_cast<const OT *>(o.data() + i)),
                    VS::load(reinterpret_cast<const OT *>(data() + i)))
                );
                i += VS::COUNT;
            } while(i < core_.size());
        } else {
            for(size_t i = 0; i < core_.size(); ++i)
                core_[i] += o.core_[i];
        }
        return *this;
    }
    cs4wbase_t operator+(const cs4wbase_t &o) const {
        auto tmp = *this;
        tmp += o;
        return tmp;
    }
    cs4wbase_t &operator-=(const cs4wbase_t &o) {
        // TODO: SIMD optimize (but how often is this needed?)
        PREC_REQ(size() == o.size(), "mismatched sizes");
        for(size_t i = 0; i < size(); ++i)
            core_[i] -= o.core_[i];
        return *this;
    }
    cs4wbase_t operator-(const cs4wbase_t &o) const {
        auto tmp = *this;
        tmp -= o;
        return tmp;
    }
    cs4wbase_t fold(int n=1) const {
        PREC_REQ(n >= 1, "n < 0 is meaningless and n = 1 uses a copy instead.");
        PREC_REQ(n <= int(np_), "Can't fold to less than 1");
        cs4wbase_t ret(np_ - n, nh_, seedseed_);
        unsigned destmod = (1ull << ret.p()) - 1;
        // More cache-efficient way to traverse than iterating over the final sketch
        const size_t coresubsz = 1ull << p();
        for(auto h = 0u; h < nh_; ++h) {
            auto destptr = &ret.core_[h << ret.p()];
            auto coreptr = &core_[h << p()];
            for(size_t i = 0; i < coresubsz; ++i)
                destptr[i & destmod] += coreptr[i];
        }
        return ret;
    }
    void read(std::FILE *fp) {
        std::fread(&np_, sizeof(np_), 1, fp);
        std::fread(&nh_, sizeof(nh_), 1, fp);
        std::fread(&seedseed_, sizeof(seedseed_), 1, fp);
        core_.resize(size_t(nh_) << np_);
        std::fread(data(), sizeof(CounterType), core_.size(), fp);
        mask_ = (1ull << np_) - 1;
    }
    void write(std::FILE *fp) const {
        std::fwrite(&np_, sizeof(np_), 1, fp);
        std::fwrite(&nh_, sizeof(nh_), 1, fp);
        std::fwrite(&seedseed_, sizeof(seedseed_), 1, fp);
        std::fwrite(data(), sizeof(CounterType), core_.size(), fp);
    }
    void read(std::string p) const {
        std::FILE *ofp = std::fopen(p.data(), "rb");
        if(!ofp)
            throw std::invalid_argument("File not found");
        read(ofp);
        std::fclose(ofp);
    }
    void write(std::string p) const {
        std::FILE *ofp = std::fopen(p.data(), "wb");
        if(!ofp)
            throw std::invalid_argument("File not found");
        write(ofp);
        std::fclose(ofp);
    }

};


template<typename VectorType=DefaultCompactVectorType,
         typename HashStruct=WangHash>
class cmmbase_t: protected ccmbase_t<update::Increment, VectorType, HashStruct> {
    uint64_t stream_size_;
    using BaseType = ccmbase_t<update::Increment, VectorType, HashStruct>;
public:
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

template<typename CMType, template<typename...> class QueueContainer=std::deque, typename...Args>
class SlidingWindow {
    using qc = QueueContainer<uint64_t, Args...>;
    qc hashes_;
public:
    CMType cm_;
    size_t queue_size_;
    SlidingWindow(size_t queue_size, CMType &&cm, qc &&hashes=qc()):
        hashes_(std::move(hashes)),
        cm_(std::move(cm)),
        queue_size_(queue_size)
    {
    }
    void addh(uint64_t v) {
        cm_.addh(v);
        if(hashes_.size() == queue_size_) {
            cm_.subh(hashes_.front());
            hashes_.pop_front();
            hashes_.push_back(v);
        }
    }
    CMType &sketch() {
        return cm_;
    }
    const CMType &sketch() const {
        return cm_;
    }
    CMType &&release() {
        return std::move(cm_);
    }
};

using ccm_t = ccmbase_t<>;
using cmm_t = cmmbase_t<>;
using cs_t = csbase_t<>;
using cs4w_t = cs4wbase_t<>;
using pccm_t = ccmbase_t<update::PowerOfTwo>;

} // namespace cm
} // namespace sketch
