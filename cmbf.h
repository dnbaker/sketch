#pragma once
#include "common.h"
#include "compact_vector/include/compact_vector.hpp"
#include "aesctr/aesctr.h"
#include <ctime>

namespace sketch {
namespace cmbf {


namespace update {

struct Increment {
    // Saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
        ref += (ref < maxval);
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType maxval) {
#if !NDEBUG
            //std::fprintf(stderr, "Max value is %u, alue at pos is %u, incrementing to %u\n", unsigned(maxval), unsigned(con[ref[0]]), unsigned(con[ref[0]]) + 1);
#endif
            if(con[ref[0]] < maxval) {
                for(const auto el: ref) {
                    con[el] = con[el] + 1;
                }
            }
    }
    template<typename... Args>
    Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
#if !NDEBUG
        //std::fprintf(stderr, "Est count: %" PRIu64 "\n", val);
#endif
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
            const unsigned oldref = ref;
            ref += ((gen_ & (UINT64_C(-1) >> (64 - unsigned(ref)))) == 0);
            gen_ >>= oldref, nbits_ -= oldref;
        }
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType maxval) {
        uint64_t val = con[ref[0]];
#if !NDEBUG
        //std::fprintf(stderr, "Value before incrementing: %zu\n", size_t(val));
        //for(const auto i: ref) std::fprintf(stderr, "These should all be the same: %u\n", unsigned(con[i]));
#endif
        if(val >= maxval) {
#if !NDEBUG
        std::fprintf(stderr, "value %zu >= maxval (%zu)\n", size_t(val), size_t(maxval));
#endif
            return;
        }
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
                for(const auto el: ref)
                    con[el] = val;
            }
            gen_ >>= oldval;
            nbits_ -= oldval;
        }
    }
    template<typename T1, typename T2>
    uint64_t combine(const T1 &i, const T2 &j) {
        return uint64_t(i) + (i == j);
    }
    PowerOfTwo(uint64_t seed=0): rng_(seed), gen_(rng_()), nbits_(64) {}
    uint64_t est_count(uint64_t val) const {
#if !NDEBUG
        //std::fprintf(stderr, "Getting count for item %" PRIu64 ". Result: %" PRIu64 "\n", val,
        //             val ? uint64_t(1) << (val - 1): 0);
#endif
        return uint64_t(1) << (val - 1);
    }
};

} // namespace update

using common::VType;
using common::Type;
using common::Space;

template<typename UpdateStrategy=update::Increment,
         typename VectorType=compact::vector<uint64_t, uint64_t, common::Allocator<uint64_t>>,
         typename HashStruct=common::WangHash>
class cmbfbase_t {

protected:
    VectorType        data_;
    UpdateStrategy updater_;
    const unsigned nhashes_;
    const unsigned l2sz_:16;
    const unsigned nbits_:16;
    const uint64_t max_tbl_val_;
    const uint64_t mask_;
    const uint64_t subtbl_sz_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;

public:
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(data_) + sizeof(updater_) + sizeof(unsigned) + sizeof(max_tbl_val_) + sizeof(mask_) + sizeof(subtbl_sz_) + sizeof(seeds_),
                              seeds_.size() * sizeof(seeds_[0]) + data_.bytes());
    }
    cmbfbase_t(int nbits, int l2sz, int nhashes=4, uint64_t seed=0):
            data_(nbits, nhashes << l2sz), updater_(seed + l2sz * nbits * nhashes),
            nhashes_(nhashes), l2sz_(l2sz),
            nbits_(nbits), max_tbl_val_((1ull<<nbits) - 1),
            mask_((1ull << l2sz) - 1), subtbl_sz_(1ull << l2sz)
    {
        if(__builtin_expect(nbits < 0, 0)) throw std::runtime_error("Number of bits cannot be negative.");
        if(__builtin_expect(l2sz < 0, 0)) throw std::runtime_error("l2sz cannot be negative.");
        if(__builtin_expect(nhashes < 0, 0)) throw std::runtime_error("nhashes cannot be negative.");
        std::fprintf(stderr, "Initialized cmbfbase_t with %zu table entries\n", size_t(nhashes << l2sz));
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
        assert(data_.size() == subtbl_sz_ * nhashes_);
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
        std::vector<uint64_t> indices, best_indices;
        indices.reserve(nhashes_);
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp, mask = Space::set1(mask_);
        while((int)nhashes_ - (int)nhdone >= (ssize_t)Space::COUNT) {
            tmp = Space::and_fn(mask.simd_, hash(Space::xor_fn(vb.simd_, Space::load(sptr++))));
            tmp.for_each([&](uint64_t &subval){
                indices.push_back(subval + nhdone++ * subtbl_sz_);
            });
        }
        while(nhdone < nhashes_) {
            indices.push_back((hash(val ^ seeds_[nhdone]) & mask_) + nhdone * subtbl_sz_);
            ++nhdone;
        }
        best_indices.push_back(indices[0]);
        size_t minval = data_[indices[0]];
        unsigned score;
        for(size_t i(1); i < indices.size(); ++i) {
            if((score = data_[indices[i]]) == minval) {
                best_indices.push_back(indices[i]);
            } else if(score < minval) {
                best_indices.clear();
                best_indices.push_back(indices[i]);
                minval = score;
            }
        }
        updater_(best_indices, data_, max_tbl_val_);
    }
    void add_liberal(uint64_t val) {
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        while(nhashes_ - nhdone > Space::COUNT) {
            Space::VType tmp = hash(Space::xor_fn(Space::set1(val), Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                std::fprintf(stderr, "Querying at position %u, with value %u", nhdone * subtbl_sz_ + (subval & mask_), unsigned(data_[subtbl_sz_ * nhdone++ + (subval & mask_)]));
#endif
                updater_(data_[subtbl_sz_ * nhdone++ + (subval & mask_)], max_tbl_val_);
            });
        }
        while(nhdone < nhashes_) {
            updater_(data_[subtbl_sz_ * nhdone + (hash(val ^ seeds_[nhdone]) & mask_, max_tbl_val_)]);
            ++nhdone;
        }
    }
    uint64_t est_count(uint64_t val) {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        uint64_t count = std::numeric_limits<uint64_t>::max(), nhdone = 0;
        static const Space::VType and_val = Space::set1(mask_), vb = Space::set1(val);
        Space::VType tmp;
        while((int)nhashes_ - (int)nhdone > (ssize_t)Space::COUNT) {
            tmp = Space::and_fn(hash(Space::xor_fn(vb.simd_, Space::load(sptr++))), and_val.simd_);
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                const auto ind  = subval + subtbl_sz_ * nhdone;
                assert(ind < data_.size() || !std::fprintf(stderr, "index %zu is too big for nhdone %" PRIu64 ". subval %" PRIu64 " with mask %zu. subtbl size %zu\n", size_t(ind), nhdone, subval, size_t(mask_), size_t(subtbl_sz_)));
#endif
                count = std::min(count, uint64_t(data_[subval + subtbl_sz_ * nhdone++]));
            });
        }
        while(nhdone < nhashes_) {
            uint64_t ind = index(hash(val ^ seeds_[nhdone]), nhdone);
            count = std::min(count, uint64_t(data_[ind]));
            ++nhdone;
        }
        return updater_.est_count(count);
    }
    cmbfbase_t operator+(const cmbfbase_t &other) const {
        cmbfbase_t cpy = *this;
        cpy += other;
        return cpy;
    }
    cmbfbase_t operator&(const cmbfbase_t &other) const {
        cmbfbase_t cpy = *this;
        cpy &= other;
        return cpy;
    }
    cmbfbase_t &operator&=(const cmbfbase_t &other) {
        if(seeds_.size() != other.seeds_.size() || !std::equal(seeds_.cbegin(), seeds_.cend(), other.seeds_.cbegin()))
            throw std::runtime_error("Could not add sketches together with different hash functions.");
        for(size_t i(0), e(data_.size()); i < e; ++i) {
            data_[i] = std::min((unsigned)data_[i], (unsigned)other.data_[i]);
        }
        return *this;
    }
    cmbfbase_t &operator+=(const cmbfbase_t &other) {
        if(seeds_.size() != other.seeds_.size() || !std::equal(seeds_.cbegin(), seeds_.cend(), other.seeds_.cbegin()))
            throw std::runtime_error("Could not add sketches together with different hash functions.");
        for(size_t i(0), e(data_.size()); i < e; ++i) {
            data_[i] = updater_.combine(data_[i], other.data_[i]);
        }
        return *this;
    }
};

using cmbf_t = cmbfbase_t<>;
using cmbf_exp_t = cmbfbase_t<update::PowerOfTwo>;

} // namespace cmbf
} // namespace sketch
