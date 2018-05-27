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
            if(con[0] < maxval) {
                for(const auto el: ref) {
#if !NDEBUG
                    // std::fprintf(stderr, "incrementing el %u to count %u\n", unsigned(el), unsigned(con[el]) + 1);
#endif
                    con[el] = con[el] + 1;
#if !NDEBUG
                    // std::fprintf(stderr, "incremented el %u to count %u\n", unsigned(el), unsigned(con[el]));
#endif
                }
            }
    }
    template<typename... Args>
    Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
        return val;
    }
};

struct PowerOfTwo {
    aes::AesCtr<uint64_t, 8> rng_;
    uint64_t  gen_;
    uint8_t nbits_;
    // Also saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
        if(ref >= maxval) return;
        if(__builtin_expect(nbits_ < ref, 0)) gen_ = rng_(), nbits_ = 64;
        ref += ((gen_ & (UINT64_C(-1) >> (64 - ref))) == 0);
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType maxval) {
        uint64_t val = con[ref[0]];
#if !NDEBUG
        std::fprintf(stderr, "Value before incrementing: %zu\n", size_t(val));
        for(const auto i: ref) std::fprintf(stderr, "These should all be the same: %u\n", unsigned(con[i]));
#endif
        if(val >= maxval) {
#if !NDEBUG
        std::fprintf(stderr, "value %zu >= maxval (%zu)\n", size_t(val), size_t(maxval));
#endif
            return;
        }
        if(__builtin_expect(nbits_ < ref[0], 0)) gen_ = rng_(), nbits_ = 64;
        ++val;
#if !NDEBUG
        std::fprintf(stderr, "Value after incrementing: %zu. Setting all values in container of size %zu\n", size_t(val), ref.size());
#endif
        if((gen_ & (UINT64_C(-1) >> (64 - ref[0]))) == 0)
            for(const auto el: ref)
                con[el] = val;
    }
    PowerOfTwo(uint64_t seed=0): rng_(seed ? seed: std::time(nullptr)), gen_(rng_()), nbits_(64) {}
    uint64_t est_count(uint64_t val) const {
#if !NDEBUG
        std::fprintf(stderr, "Getting count for item %" PRIu64 ". Result: %" PRIu64 "\n", val,
                     val ? uint64_t(1) << (val - 1): 0);
#endif
        return val ? uint64_t(1) << (val - 1): 0;
    }
};

} // namespace update

using common::VType;
using common::Type;
using common::Space;

template<typename VectorType=compact::vector<uint64_t, uint64_t, common::Allocator<uint64_t>>,
         typename HashStruct=common::WangHash,
         typename UpdateStrategy=update::Increment>
class cmbfbase_t {

protected:
    VectorType        data_;
    UpdateStrategy updater_;
    HashStruct      hasher_;
    const unsigned nhashes_;
    const unsigned l2sz_:16;
    const unsigned nbits_:16;
    const uint64_t max_tbl_val_;
    const uint64_t mask_;
    const uint64_t subtbl_sz_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;
    
public:
    cmbfbase_t(int nbits, int l2sz, int nhashes=4, uint64_t seed=0):
            data_(nbits, nhashes << l2sz), updater_(seed), hasher_(),
            nhashes_(nhashes), l2sz_(l2sz),
            nbits_(nbits), max_tbl_val_((1ull<<nbits) - 1),
            mask_((1ull << l2sz) - 1), subtbl_sz_(1ull << l2sz)
    {
        if(__builtin_expect(nbits < 0, 0)) throw std::runtime_error("Number of bits cannot be negative.");
        if(__builtin_expect(l2sz < 0, 0)) throw std::runtime_error("l2sz cannot be negative.");
        if(__builtin_expect(nhashes < 0, 0)) throw std::runtime_error("nhashes cannot be negative.");
        std::mt19937_64 mt(seed + 4);
        while(seeds_.size() < (unsigned)nhashes) seeds_.emplace_back(mt());
        std::memset(data_.get(), 0, data_.bytes());
    }
    VectorType &ref() {return data_;}
    uint64_t index(uint64_t hash, unsigned subtbl) const {
#if !NDEBUG
        std::fprintf(stderr, "calculating index for hash = %" PRIu64 " and subtl = %u with subtable size = %zu. Result: %" PRIu64 "\n",
                     hash, subtbl, size_t(subtbl_sz_), subtbl * subtbl_sz_ + (hash & mask_));
#endif
        return subtbl_sz_ * subtbl + (hash & mask_);
    }
    void addh(uint64_t val) {this->addh_conservative(val);}
    void addh_conservative(uint64_t val) {this->add_conservative(hasher_(val));}
    void addh_liberal(uint64_t val) {this->add_liberal(hasher_(val));}
    void add_conservative(uint64_t val) {
        std::vector<uint64_t> indices, best_indices;
        indices.reserve(nhashes_);
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp;
        while(nhashes_ - nhdone >= Space::COUNT) {
            tmp = hasher_(Space::xor_fn(vb.simd_, Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                std::fprintf(stderr, "Hash is %" PRIu64 " with index %" PRIu64 " value = %u\n", subval, index(subval, nhdone), unsigned(data_[index(subval, nhdone)]));
#endif
                indices.push_back(index(subval, nhdone++));
            });
        }
        while(nhdone < nhashes_) {
#if !NDEBUG
            uint64_t ind = index(hasher_(val ^ seeds_[nhdone]), nhdone);
            std::fprintf(stderr, "index: %" PRIu64 ", which has value %u\n", ind, unsigned(data_[ind]));
#endif
            indices.push_back(index(hasher_(val ^ seeds_[nhdone]), nhdone));
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
            Space::VType tmp = hasher_(Space::xor_fn(Space::set1(val), Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                std::fprintf(stderr, "Querying at position %u, with value %u", nhdone * subtbl_sz_ + (subval & mask_), unsigned(data_[subtbl_sz_ * nhdone++ + (subval & mask_)]));
#endif
                updater_(data_[subtbl_sz_ * nhdone++ + (subval & mask_)], max_tbl_val_);
            });
        }
        while(nhdone < nhashes_) {
            updater_(data_[subtbl_sz_ * nhdone + (hasher_(val ^ seeds_[nhdone]) & mask_, max_tbl_val_)]);
            ++nhdone;
        }
    }
    uint64_t est_count(uint64_t val) {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        uint64_t count = 0, nhdone = 0, ind;
        static const Space::VType and_val = Space::set1(mask_), vb = Space::set1(val);
        Space::VType tmp;
        while(nhashes_ - nhdone > Space::COUNT) {
            tmp = Space::and_fn(hasher_(Space::xor_fn(vb.simd_, Space::load(sptr++))), and_val.simd_);
            tmp.for_each([&](uint64_t &subval){
                ind = index(subval, nhdone);
#if !NDEBUG
                std::fprintf(stderr, "Index is %" PRIu64 ". value at index %u is %u\n", ind, unsigned(nhdone), unsigned(data_[ind]));
#endif
                count = std::min(count, uint64_t(data_[ind]));
            });
        }
        while(nhdone < nhashes_) {
            uint64_t subval = (val ^ seeds_[nhdone]) & mask_;
#if !NDEBUG
            std::fprintf(stderr, "subval: %" PRIu64 ". With subtbl (%zu): %" PRIu64 ". Value at position: %u\n", subval, size_t(subtbl_sz_), subtbl_sz_ * nhdone + subval, unsigned(subtbl_sz_ * nhdone));
#endif
            count = std::min(count, uint64_t(data_[subtbl_sz_ * nhdone + (hasher_(val ^ seeds_[nhdone]) & mask_)]));
            ++nhdone;
        }
        return updater_.est_count(count);
    }
};

using cmbf_t = cmbfbase_t<>;

} // namespace cmbf
} // namespace sketch
