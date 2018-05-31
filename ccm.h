#pragma once
#include "common.h"
#include "compact_vector/include/compact_vector.hpp"
#include "aesctr/aesctr.h"
#include <ctime>

namespace sketch {

namespace cm {

namespace detail {
    template<typename IntType, typename=std::enable_if_t<std::is_signed_v<IntType>>> static constexpr IntType signarr{static_cast<IntType>(-1), static_cast<IntType>(1)};
    template<typename T>
    struct IndexedValue {
        using Type = typename std::decay_t<decltype(*(T(100, 10).cbegin()))>;
    };
    template<typename T>
    static constexpr int range_check(unsigned nbits, T val) {
        if constexpr(std::is_signed_v<T>) {
            if(val < -(1ull << (nbits - 1)))
                return -1;
            return val > (1ull << (nbits - 1)) - 1;
        } else {
            return val >= (1ull << nbits);
        }
    }
}

namespace update {

struct Increment {
    // Saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
        ref += (ref < maxval);
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType nbits) {
            unsigned count = con[ref[0]];
            if(detail::range_check<typename detail::IndexedValue<Container>::Type>(nbits, ++count) == 0)
                for(const auto el: ref)
                    con[el] = count;
    }
    template<typename... Args>
    Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
        return val;
    }
    template<typename T1, typename T2>
    uint64_t combine(const T1 &i, const T2 &j) {
        return uint64_t(i) + uint64_t(j);
    }
};


struct CountSketch {
    // Saturates
    template<typename T, typename IntType, typename IntType2>
    void operator()(T &ref, IntType maxval, IntType2 hash) {
        static constexpr size_t shift = sizeof(hash) * CHAR_BIT - 1;
        ref = (int64_t)ref + detail::signarr<IntType2>[hash>>shift];
    }
    template<typename T, typename Container, typename IntType, typename IntType2>
    void operator()(std::vector<T> &ref, std::vector<T> &hashes, Container &con, IntType nbits) {
        using IDX = typename detail::IndexedValue<Container>::Type;
        IDX count = con[ref[0]], newval;
        static constexpr size_t shift = sizeof(hashes[0]) * CHAR_BIT - 1;
        assert(ref.size() == hashes.size());
        for(size_t i(0); i < ref.size(); ++i) {
            newval = count + detail::signarr<IntType2>[hashes[i]>>shift];
            if(detail::range_check<IDX>(nbits, newval) == 0)
                con[ref[i]] = newval;
        }
    }
    template<typename... Args>
    void Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
        return val;
    }
    template<typename T1, typename T2>
    uint64_t combine(const T1 &i, const T2 &j) {
        return uint64_t(i) + uint64_t(j);
    }
};

struct PowerOfTwo {
    aes::AesCtr<uint64_t, 8> rng_;
    uint64_t  gen_;
    uint8_t nbits_;
    // Also saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
#if !NDEBUG
        std::fprintf(stderr, "maxval: %zu. ref: %zu\n", size_t(maxval), size_t(ref));
#endif
        if(ref >= maxval) return;
        if(unsigned(ref) == 0) ref = 1;
        else {
            if(__builtin_expect(nbits_ < ref, 0)) gen_ = rng_(), nbits_ = 64;
            ref += ((gen_ & (UINT64_C(-1) >> (64 - unsigned(ref)))) == 0);
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
            if((gen_ & (UINT64_C(-1) >> (64 - val))) == 0) {
#if !NDEBUG
            //std::fprintf(stderr, "We incremented a second time!\n");
#endif
                ++val;
                if(detail::range_check(nbits, val) == 0)
                    for(const auto el: ref)
                        con[el] = val;
            }
            gen_ >>= (val - 1);
        }
    }
    template<typename T1, typename T2>
    uint64_t combine(const T1 &i, const T2 &j) {
        return uint64_t(i) + (i == j);
    }
    PowerOfTwo(uint64_t seed=0): rng_(seed ? seed: std::time(nullptr)), gen_(rng_()), nbits_(64) {}
    uint64_t est_count(uint64_t val) const {
#if !NDEBUG
        //std::fprintf(stderr, "Getting count for item %" PRIu64 ". Result: %" PRIu64 "\n", val,
        //             val ? uint64_t(1) << (val - 1): 0);
#endif
        return val ? uint64_t(1) << (val - 1): 0;
    }
};

} // namespace update

using namespace common;

template<typename UpdateStrategy=update::Increment,
         typename VectorType=compact::vector<uint32_t, uint64_t, common::Allocator<uint64_t>>,
         typename HashStruct=common::WangHash>
class ccmbase_t {
    static_assert(!std::is_same_v<UpdateStrategy, update::CountSketch> || std::is_signed_v<typename detail::IndexedValue<VectorType>::Type>,
                  "If CountSketch is used, value must be signed.");

protected:
    VectorType        data_;
    UpdateStrategy updater_;
    const unsigned nhashes_;
    const unsigned l2sz_:16;
    const unsigned nbits_:16;
    const uint64_t mask_;
    const uint64_t subtbl_sz_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;

public:
    static constexpr bool is_count_sketch() {
        return std::is_same_v<UpdateStrategy, update::CountSketch>;
    }
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(data_) + sizeof(updater_) + sizeof(unsigned) + sizeof(mask_) + sizeof(subtbl_sz_) + sizeof(seeds_),
                              seeds_.size() * sizeof(seeds_[0]) + data_.bytes());
    }
    ccmbase_t(int nbits, int l2sz, int nhashes=4, uint64_t seed=0):
            data_(nbits, nhashes << l2sz), updater_(seed),
            nhashes_(nhashes), l2sz_(l2sz),
            nbits_(nbits),
            mask_((1ull << l2sz) - 1), subtbl_sz_(1ull << l2sz)
    {
        if(__builtin_expect(nbits < 0, 0)) throw std::runtime_error("Number of bits cannot be negative.");
        if(__builtin_expect(l2sz < 0, 0)) throw std::runtime_error("l2sz cannot be negative.");
        if(__builtin_expect(nhashes < 0, 0)) throw std::runtime_error("nhashes cannot be negative.");
        std::mt19937_64 mt(seed + 4);
        while(seeds_.size() < (unsigned)nhashes) seeds_.emplace_back(mt());
        std::memset(data_.get(), 0, data_.bytes());
        std::fprintf(stderr, "%i bits for each number, %i is log2 size of each table, %i is the number of subtables\n", nbits, l2sz, nhashes);
    }
    VectorType &ref() {return data_;}
    uint64_t index(uint64_t hash, unsigned subtbl) const {
        return subtbl_sz_ * subtbl + (hash & mask_);
    }
    void addh(uint64_t val) {
        this->addh_conservative(val);
    }
    void addh_conservative(uint64_t val) {this->add_conservative(val);}
    void addh_liberal(uint64_t val) {this->add_liberal(val);}
    template<typename T>
    T hash(T val) const {
        return HashStruct()(val);
    }
    bool may_contain(uint64_t val) const {
        unsigned nhdone = 0;
        bool ret = 1;
        Space::VType v;
        const Space::Type *seeds(reinterpret_cast<Space::Type *>(&seeds_[0]));
        while(nhashes_ - nhdone >= Space::COUNT) {
            v = hash(Space::xor_fn(Space::set1(val), *seeds++));
            v.for_each([&](const uint64_t &hv) {
                ref &= data_[(hv & mask_) + nhdone++ * subtbl_sz_];
            });
            if(!ret) return false;
        }
        while(nhdone < nhashes_) {
            if(!data_[hash(val ^ seeds_[nhdone]) & mask_ + (nhdone * subtbl_sz_)]) return false;
            ++nhdone;
        }
        return true;
    }
    uint32_t may_contain(Space::VType val) const {
        Space::VType tmp, hv, and_val;
        unsigned nhdone = 0, vindex = 0;
        const Space::Type *seeds(reinterpret_cast<const Space::Type *>(seeds_.data()));
        and_val = Space::set1(mask_);
        uint32_t ret = static_cast<uint32_t>(-1) >> ((sizeof(ret) * CHAR_BIT) - Space::COUNT);
        uint32_t bitmask;
        while(nhdone < nhashes_) {
            bitmask = static_cast<uint32_t>(-1) >> ((sizeof(ret) * CHAR_BIT) - Space::COUNT) & ~(1u << vindex);
            hv = Space::load(seeds++);
            val.for_each([&](uint64_t w) {
                tmp = Space::set1(w);
                tmp = Space::xor_fn(tmp.simd_, hv.simd_);
                tmp = hash(tmp.simd_);
                tmp = Space::and_fn(and_val.simd_, tmp.simd_);
                tmp.for_each([&](const uint64_t &subw) {
                    ret &= (~(data_[subw + nhdone * subtbl_sz_] == 0) << vindex++);
                });
            });
            vindex = 0;
            ++nhdone;
        }
        return ret;
    }
    void add_conservative(const uint64_t val) {
        std::vector<uint64_t> indices, best_indices, hashes, best_hashes;
        indices.reserve(nhashes_);
        if constexpr(is_count_sketch()) {
            hashes.reserve(nhashes_);
        }
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp, mask = Space::set1(mask_);
        while((int)nhashes_ - (int)nhdone >= (ssize_t)Space::COUNT) {
            
            tmp = hash(Space::xor_fn(vb.simd_, Space::load(sptr++)));
            if constexpr(is_count_sketch()) {
                tmp.for_each([&](uint64_t subval) {hashes.push_back(subval);});
            }
            tmp = Space::and_fn(mask.simd_, tmp.simd_);
            tmp.for_each([&](uint64_t &subval){
                indices.push_back(subval + nhdone++ * subtbl_sz_);
            });
        }
        while(nhdone < nhashes_) {
            uint64_t hv = hash(val ^ seeds_[nhdone]);
            if constexpr(is_count_sketch()) hashes.push_back(hv);
            hv &= mask_;
            hv += nhdone * subtbl_sz_;
            indices.push_back(hv);
            ++nhdone;
        }
        best_indices.push_back(indices[0]);
        best_hashes.push_back(indices[1]);
        size_t minval = data_[indices[0]];
        unsigned score;
        for(size_t i(1); i < indices.size(); ++i) {
            if((score = data_[indices[i]]) == minval) {
                best_indices.push_back(indices[i]);
                if constexpr(is_count_sketch())
                    best_hashes.push_back(hashes[i]);
            } else if(score < minval) {
                best_indices.clear();
                best_indices.push_back(indices[i]);
                if constexpr(is_count_sketch()) {
                    best_hashes.clear();
                    best_hashes.push_back(hashes[i]);
                }
                minval = score;
            }
        }
        if constexpr(is_count_sketch())
            updater_(best_indices, best_hashes, data_, nbits_);
        else updater_(best_indices, data_, nbits_);
    }
    void add_liberal(uint64_t val) {
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        while(nhashes_ - nhdone > Space::COUNT) {
            Space::VType tmp = hash(Space::xor_fn(Space::set1(val), Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                std::fprintf(stderr, "Querying at position %u, with value %u", nhdone * subtbl_sz_ + (subval & mask_), unsigned(data_[subtbl_sz_ * nhdone + (subval & mask_)]));
#endif
                updater_(data_[subtbl_sz_ * nhdone++ + (subval & mask_)], nbits_);
            });
        }
        while(nhdone < nhashes_) {
            updater_(data_[subtbl_sz_ * nhdone + (hash(val ^ seeds_[nhdone]) & mask_, nbits_)]);
            ++nhdone;
        }
    }
    uint64_t est_count(uint64_t val) {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType tmp;
        const Space::VType and_val = Space::set1(mask_), vb = Space::set1(val);
        unsigned nhdone = 0;
        if constexpr(!is_count_sketch()) {
            uint64_t count = std::numeric_limits<uint64_t>::max();
            while(nhashes_ - nhdone > Space::COUNT) {
                tmp = Space::and_fn(hash(Space::xor_fn(vb.simd_, Space::load(sptr++))), and_val.simd_);
                tmp.for_each([&](uint64_t &subval){
                    count = std::min(count, uint64_t(data_[subval + subtbl_sz_ * nhdone++]));
                });
            }
            while(nhdone < nhashes_) {
                uint64_t ind = index(hash(val ^ seeds_[nhdone]), nhdone);
                count = std::min(count, uint64_t(data_[ind]));
                ++nhdone;
            }
            return updater_.est_count(count);
        } else {
            std::vector<int64_t> estimates; estimates.reserve(nhashes_);
            while(nhashes_ - nhdone > Space::COUNT) {
                tmp = hash(Space::xor_fn(vb.simd_, Space::load(sptr++)));
                tmp.for_each([&](uint64_t &subval){
                    estimates.push_back(data_[(subval & mask_) + subtbl_sz_ * nhdone++] * detail::signarr<int64_t>[subval >> (sizeof(uint64_t) * CHAR_BIT - 1)]);
                });
            }
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[nhdone]);
                estimates.push_back(data_[hv & mask_ + subtbl_sz_ * nhdone++] + detail::signarr<int64_t>[hv >> (sizeof(uint64_t) * CHAR_BIT - 1)]);
                ++nhdone;
            }
            sort::insertion_sort(std::begin(estimates), std::end(estimates));
            return std::max(static_cast<int64_t>(0),
                            nhashes_ & 1 ? estimates[nhashes_>>1]
                                         : (estimates[nhashes_>>1] + estimates[(nhashes_>>1) - 1]) / 2);
        }
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
            data_[i] = std::min((unsigned)data_[i], (unsigned)other.data_[i]);
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

using ccm_t = ccmbase_t<>;
using pccm_t = ccmbase_t<update::PowerOfTwo>;
using cvector_i32 = compact::vector<int32_t, uint64_t, Allocator<uint64_t>>;
using cvector_i64 = compact::vector<int64_t, uint64_t, Allocator<uint64_t>>;
using cs_t = ccmbase_t<update::Increment, cvector_i32>;
using cs64_t = ccmbase_t<update::Increment, cvector_i64>;
// Note that cs_t needs to have a signed integer.

} // namespace cm
} // namespace sketch
