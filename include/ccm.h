#pragma once
#include <ctime>
#include <deque>
#include <queue>
#include "hash.h"
#include "update.h"

namespace sketch {

namespace cm {
using common::detail::tmpbuffer;

namespace detail {
template<typename T, typename AllocatorType>
static inline float sqrl2(const std::vector<T, AllocatorType> &v, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<float, 8> mem(nhashes);
    float *ptr = mem.get();
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
            full_sum += *p1++;
        ptr[i] = std::sqrt(full_sum);
    }
    common::sort::insertion_sort(ptr, ptr + nhashes);
    float ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
}
template<typename T, typename AllocatorType>
static inline float sqrl2(const std::vector<T, AllocatorType> &v, const std::vector<T, AllocatorType> &v2, uint32_t nhashes, uint32_t l2sz) {
    assert(v.size() == v2.size());
    tmpbuffer<float, 8> mem(nhashes);
    float *ptr = mem.get();
    using VT = typename vec::SIMDTypes<T>::VType;
    using VS = vec::SIMDTypes<T>;
    VT sum = VS::set1(0);
    static constexpr size_t ct = VS::COUNT;
    const bool is_aligned = VT::aligned(v.data()) && VT::aligned(v2.data());
    for(size_t i = 0; i < nhashes; ++i) {
        const T *p1 = &v[i << l2sz], *p2 = &v2[i<<l2sz], *p1e = &v[(i + 1) << l2sz];
#if 0
        while(p1e - p1 > ct) {
            auto lv = VS::load(p1), VS::load(p2);
            const auto lv = *reinterpret_cast<const VT *>(p1);
            const auto rv = *reinterpret_cast<const VT *>(p2);
            sum = VS::add(sum, VS::abs(VS::mul(lv, rv)));
            p1 += ct;
            p2 += ct;
        }
        T full_sum = sum.sum();
        while(p1 < p1e)
            full_sum += *p1++ * *p2++;
#endif
        // SIMD absolute value is hard, ignore it.
        T full_sum = std::abs(*p1++ * *p2++);
        while(p1 != p1e) full_sum += std::abs(*p1++ * *p2++);

        ptr[i] = std::sqrt(full_sum);
    }
    common::sort::insertion_sort(ptr, ptr + nhashes);
    float ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline float sqrl2(const compact::vector<T1, BITS, T2, Allocator> &v, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<float, 8> mem(nhashes);
    float *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        float sum = 0;
        while(start != end) {
            int64_t val = v[start++];
            sum += val * val;
        }
        ptr[i] = std::sqrt(sum);
    }
    common::sort::insertion_sort(ptr, ptr + nhashes);
    float ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
}

template<typename T1, unsigned int BITS, typename T2, typename Allocator>
static inline float sqrl2(const compact::vector<T1, BITS, T2, Allocator> &v, const compact::vector<T1, BITS, T2, Allocator> &v2, uint32_t nhashes, uint32_t l2sz) {
    tmpbuffer<float, 8> mem(nhashes);
    float *ptr = mem.get();
    for(size_t i = 0; i < nhashes; ++i) {
        size_t start = i << l2sz, end = (i + 1) << l2sz;
        float sum = 0;
        while(start != end) {
            sum += v[start] * v2[start];
            ++start;
        }
        ptr[i] = std::sqrt(sum);
    }
    common::sort::insertion_sort(ptr, ptr + nhashes);
    float ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
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
    common::sort::insertion_sort(ptr, ptr + nhashes);
    double ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
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
    common::sort::insertion_sort(ptr, ptr + nhashes);
    double ret = (ptr[nhashes >> 1] + ptr[(nhashes - 1) >> 1]) * .5;
    return ret;
}

template<typename T>
struct IndexedValue {
    using Type = typename std::decay_t<decltype(*(std::declval<T>().cbegin()))>;
};
} // namespace detail


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
    ccmbase_t(ccmbase_t &&o): data_(std::move(o.data_)), updater_(std::move(updater_)), nhashes_(o.nhashes_), l2sz_(o.l2sz_),
                              nbits_(o.nbits_), hf_(std::move(hf_)), mask_(o.mask_), subtbl_sz_(o.subtbl_sz_), seeds_(std::move(o.seeds_))
    {
    }
    ccmbase_t(const ccmbase_t &o) = default;
    //ccmbase_t(ccmbase_t &&o) = default;
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
#if VERBOSE_AF
        std::fprintf(stderr, "size of data: %zu\n", data_.size());
        std::fprintf(stderr, "%i bits for each number, %i is log2 size of each table, %i is the number of subtables. %zu is the number of 64-bit hashes with %u nhashesper64bitword\n", nbits, l2sz, nhashes, seeds_.size(), nperhash64);
        std::fprintf(stderr, "Size of updater: %zu. seeds length: %zu\n", sizeof(updater_), seeds_.size());
#endif
    }
    VectorType &ref() {return data_;}
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
        auto cptr = counts.get();
        sort::insertion_sort(cptr, p);
        return (cptr[(nhashes_ >> 1)] + cptr[(nhashes_ - 1 ) >> 1]) * .5;
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
        static constexpr bool is_increment = std::is_same<UpdateStrategy, update::Increment>::value;
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
            //std::fprintf(stderr, "Doing SIMD stuff\n");
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(Space::COUNT * nperhash64)) {
                //std::fprintf(stderr, "SIMD\n");
                Space::VType(hash(Space::xor_fn(vb.simd_, Space::load(sptr++)))).for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64;) {
                        const uint32_t index = ((subval >> (k++ * nbitsperhash)) & mask_) + nhdone * subtbl_sz_;
                        assert(index < data_.size());
                        assert(index < data_.size() || !std::fprintf(stderr, "nhdone: %u, subtblsz: %zu, index %u. k: %u\n", nhdone, size_t(subtbl_sz_), index, k));
                        indices.push_back(index);
                        ++nhdone;
                    }
                });
                seedind += Space::COUNT;
            }
            //std::fprintf(stderr, "Doing Leftover stuff\n");
            while(nhdone < nhashes_) {
                assert(seeds_.data());
                uint64_t hv = hash(val ^ seeds_[seedind]);
                for(unsigned k(0), e = std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); k < e;) {
                    const uint32_t index = ((hv >> (k++ * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone;
                    //std::fprintf(stderr, "index: %u.\n", index);
                    assert(index < data_.size() || !std::fprintf(stderr, "nhdone: %u, subtblsz: %zu, index %u. k: %u\n", nhdone, size_t(subtbl_sz_), index, k));
                    indices.push_back(index);
                    if(++nhdone == nhashes_) {
                        //std::fprintf(stderr, "Going\n");
                        goto end;
                    }
                }
                ++seedind;
            }
            end:
            //std::fprintf(stderr, "Now get best\n");
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
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(Space::COUNT * nperhash64)) {
                Space::VType(hash(Space::xor_fn(vb.simd_, Space::load(sptr++)))).for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64;) {
                        auto ref = data_.operator[](((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_);
                        updater_(ref, 1u << nbits_);
                        ret = std::min(ret, ssize_t(ref));
                    }
                });
                seedind += Space::COUNT;
            }
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[seedind++]);
                for(unsigned k(0), e = std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); k != e;) {
                    auto ref = data_.operator[](((hv >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_);
                    updater_(ref, 1u << nbits_);
                    ret = std::min(ret, ssize_t(ref));
                }
            }
        }
        return ret + std::is_same<UpdateStrategy, update::Increment>::value;
    }
    uint64_t est_count(uint64_t val) const {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType tmp;
        //const Space::VType and_val = Space::set1(mask_);
        const Space::VType vb = Space::set1(val);
        unsigned nhdone = 0, seedind = 0;
        const auto nperhash64 = lut::nhashesper64bitword[l2sz_];
        const auto nbitsperhash = l2sz_;
        uint64_t count = std::numeric_limits<uint64_t>::max();
        while(nhashes_ - nhdone > Space::COUNT * nperhash64) {
            Space::VType(hash(Space::xor_fn(vb.simd_, Space::load(sptr++)))).for_each([&](const uint64_t subval){
                for(unsigned k = 0; k < nperhash64; count = std::min(count, uint64_t(data_.operator[](((subval >> (k++ * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++))));
            });
            seedind += Space::COUNT;
        }
        FOREVER {
            assert(nhdone < nhashes_);
            uint64_t hv = hash(val ^ seeds_[seedind++]);
            for(unsigned k = 0, e = std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); k != e; ++k) {
                uint32_t index = ((hv >> (k * nbitsperhash)) & mask_) + nhdone * subtbl_sz_;
                assert(index < data_.size());
                count = std::min(count, uint64_t(data_.operator[](index)));
                if(++nhdone == nhashes_) goto end;
            }
        }
        end:
        return updater_.est_count(count);
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
    uint32_t np_, nh_, nph_;
    const HashStruct hf_;
    uint64_t mask_;
    std::vector<CounterType, Allocator<CounterType>> seeds_;

    CounterType       *data()       {return core_.data();}
    const CounterType *data() const {return core_.data();}
public:
    template<typename...Args>
    csbase_t(unsigned np, unsigned nh=1, unsigned seedseed=137, Args &&...args):
        core_(uint64_t(nh) << np), np_(np), nh_(nh), nph_(64 / (np + 1)), hf_(std::forward<Args>(args)...),
        mask_((1ull << np_) - 1),
        seeds_((nh_ + (nph_ - 1)) / nph_ - 1)
    {
        DefaultRNGType gen(np + nh + seedseed);
        for(auto &el: seeds_) el = gen();
        // Just to make sure that simd addh is always accessing owned memory.
        while(seeds_.size() < sizeof(Space::Type) / sizeof(uint64_t)) seeds_.emplace_back(gen());
    }
    double l2est() const {
        return sqrl2(core_, nh_, np_);
    }
    CounterType addh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1), *cptr++ = add(v, added++));
        auto it = seeds_.begin();
        while(added < nh_) {
            v = hf_(*it++ ^ val);
            for(unsigned k = nph_; k--; v >>= (np_ + 1)) {
                *cptr++ = add(v, added++);
                if(added == nh_) break; // this could be optimized by pre-scanning, I think.
            }
        }
        sort::insertion_sort(counts.get(), cptr);
        cptr = counts.get();
        return (cptr[(nh_ >> 1)] + cptr[(nh_ - 1 ) >> 1]) >> 1;
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
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1), sub(v, added++));
        auto it = seeds_.begin();
        while(added < nh_) {
            v = hf_(*it++ ^ val);
            for(unsigned k = nph_; k--; v >>= (np_ + 1)) {
                sub(v, added++);
                if(added == nh_) break; // this could be optimized by pre-scanning, I think.
            }
        }
    }
    auto subh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1), *cptr++ = sub(v, added++));
        auto it = seeds_.begin();
        while(added < nh_) {
            v = hf_(*it++ ^ val);
            for(unsigned k = nph_; k--; v >>= (np_ + 1)) {
                *cptr++ = sub(v, added++);
                if(added == nh_) break; // this could be optimized by pre-scanning, I think.
            }
        }
        sort::insertion_sort(counts.get(), cptr);
        cptr = counts.get();
        return (cptr[(nh_ >> 1)] + cptr[(nh_ - 1 ) >> 1]) >> 1;
    }
    INLINE size_t index(uint64_t hv, unsigned subidx) const noexcept {
        return (hv & mask_) + (subidx << np_);
    }
    INLINE auto add(uint64_t hv, unsigned subidx) noexcept {
#if !NDEBUG
        //std::fprintf(stderr, "Value: %d is about to be incremented at index %u\n", unsigned(index(hv, subidx)));
        at_pos(hv, subidx) += sign(hv);
        //std::fprintf(stderr, "Value: %d was supposedly incremented by sign %d. Index: %zu/%zu\n", unsigned(at_pos(hv, subidx)), sign(hv), index(hv, subidx), core_.size());
        return at_pos(hv, subidx);
#else
        return at_pos(hv, subidx) += sign(hv);
#endif
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
    INLINE void subh(Space::VType hv) noexcept {
        unsigned gadded = 0;
        Space::VType(hf_(hv)).for_each([&](uint64_t v) {
            for(uint32_t added = 0; added++ < std::min(nph_, nh_) && gadded < nh_;v >>= (np_ + 1))
                sub(v, gadded++);
        });
        for(auto it = seeds_.begin(); gadded < nh_;) {
            unsigned lastgadded = gadded;
            Space::VType(hf_(Space::xor_fn(Space::set1(*it++), hv))).for_each([&](uint64_t v) {
                unsigned added;
                for(added = lastgadded; added < std::min(lastgadded + nph_, nh_); sub(v, added++), v >>= (np_ + 1));
                gadded = added;
            });
        }
    }
    INLINE void addh(Space::VType hv) noexcept {
        unsigned gadded = 0;
        Space::VType(hf_(hv)).for_each([&](uint64_t v) {
            for(uint32_t added = 0; added++ < std::min(nph_, nh_) && gadded < nh_;v >>= (np_ + 1))
                add(v, gadded++);
        });
        for(auto it = seeds_.begin(); gadded < nh_;) {
            unsigned lastgadded = gadded;
            Space::VType(hf_(Space::xor_fn(Space::set1(*it++), hv))).for_each([&](uint64_t v) {
                unsigned added;
                for(added = lastgadded; added < std::min(lastgadded + nph_, nh_); add(v, added++), v >>= (np_ + 1));
                gadded = added;
            });
        }
    }
    CounterType est_count(uint64_t val) const {
        common::detail::tmpbuffer<CounterType> mem(nh_);
        CounterType *ptr = mem.get(), *p = ptr;
        uint64_t v = hf_(val);
        unsigned added;
        for(added = 0; added < std::min(nph_, nh_); v >>= (np_ + 1))
            *p++ = at_pos(v, added++) * sign(v);
        if(added == nh_) goto end;
        for(auto it = seeds_.begin();;) {
            v = hf_(*it++ ^ val);
            for(unsigned k = 0; k < nph_; ++k, v >>= (np_ + 1)) {
                *p++ = at_pos(v, added++) * sign(v);
                if(added == nh_) goto end;
            }
        }
        end:
        //std::for_each(mem.get(), mem.get() + nh_, [p=mem.get()](const auto &x) {std::fprintf(stderr, "Count estimate for ind %zd is %u\n", &x - p, int32_t(x));});
        ///
        if(nh_ > 1) {
            sort::insertion_sort(ptr, ptr + nh_);
            CounterType start1 = ptr[(nh_ - 1)>>1];
            start1 += ptr[(nh_-1)>>1];
            return start1 >> 1;
        } else return ptr[0];
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
};
template<typename CounterType=int32_t, typename=typename std::enable_if<std::is_signed<CounterType>::value>::type>
class cs4wbase_t {
    /*
     * Commentary: because of chance, one can end up with a negative number as an estimate.
     * Either the item collided with another item which was quite large and it was outweighed
     * or it and others in the bucket were not heavy enough and by chance it did
     * not weigh over the other items with the opposite sign. Treat these as 0s.
    */
    std::vector<CounterType, Allocator<CounterType>> core_;
    uint32_t np_, nh_;
    uint64_t mask_;
    const KWiseHasherSet<4> hf_;
    CounterType       *data()       {return core_.data();}
    const CounterType *data() const {return core_.data();}

    size_t size() const {return core_.size();}
public:
    template<typename...Args>
    cs4wbase_t(unsigned np, unsigned nh=1, unsigned seedseed=137, Args &&...args):
        core_(uint64_t(nh) << np), np_(np), nh_(nh),
        mask_((1ull << np_) - 1),
        hf_(seedseed)
    {
    }
    double l2est() const {
        return sqrl2(core_, nh_, np_);
    }
    CounterType addh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        auto cp2 = cptr;
        for(unsigned added = 0; added < nh_; ++added)
            *cptr++ = add(val, added);

        sort::insertion_sort(cp2, cptr);
        cptr = cp2;
        return (cptr[(nh_ >> 1)] + cptr[(nh_ - 1 ) >> 1]) / static_cast<CounterType>(2);
    }
    auto addh(uint64_t val) {return addh_val(val);}
    void subh(uint64_t val) {
        for(unsigned added = 0; added < nh_; ++added)
            sub(val, added);
    }
    auto subh_val(uint64_t val) {
        tmpbuffer<CounterType> counts(nh_);
        auto cptr = counts.get();
        for(unsigned added = 0; added < nh_; ++added) {
            *cptr++ = sub(val, added);
        }
        sort::insertion_sort(counts.get(), cptr);
        cptr = counts.get();
        return (cptr[(nh_ >> 1)] + cptr[(nh_ - 1 ) >> 1]) >> 1;
    }
    INLINE size_t index(uint64_t hv, unsigned subidx) const noexcept {
        return (hv & mask_) + (subidx << np_);
    }
    INLINE auto add(uint64_t hv, unsigned subidx) noexcept {
        hv = hf_(hv, subidx);
        auto &ref = at_pos(hv, subidx);
        if(ref != std::numeric_limits<CounterType>::max()) // easy branch to predict
            ref += sign(hv);
        return ref;
    }
    INLINE auto sub(uint64_t hv, unsigned subidx) noexcept {
        hv = hf_(hv, subidx);
        auto &ref = at_pos(hv, subidx);
        if(ref != std::numeric_limits<CounterType>::min()) // easy branch to predict
            ref -= sign(hv);
        return ref;
    }
    INLINE auto &at_pos(uint64_t hv, unsigned subidx) noexcept {
        assert(index(hv, subidx) < core_.size() || !std::fprintf(stderr, "hv & mask_: %zu. subidx %d. np: %d. nh: %d. size: %zu\n", size_t(hv&mask_), subidx, np_, nh_, core_.size()));
        return core_[index(hv, subidx)];
    }
    INLINE auto at_pos(uint64_t hv, unsigned subidx) const noexcept {
        assert((hv & mask_) + (subidx << np_) < core_.size());
        return core_[index(hv, subidx)];
    }
    INLINE int sign(uint64_t hv) const noexcept {
        return hv & (1ul << np_) ? 1: -1;
    }
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
        if(nh_ > 1) {
            sort::insertion_sort(ptr, ptr + nh_);
            CounterType start1 = ptr[(nh_ - 1)>>1];
            start1 += ptr[(nh_-1)>>1];
            return start1 >> 1;
        } else {
            return ptr[0];
        }
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
};


template<typename VectorType=DefaultCompactVectorType,
         typename HashStruct=common::WangHash>
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
    SlidingWindow(size_t queue_size, CMType &&cm, qc &&hashes=qc()): queue_size_(queue_size), cm_(std::move(cm)) {
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
