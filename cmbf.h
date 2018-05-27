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
            if((gen_ & (UINT64_C(-1) >> (64 - val))) == 0) {
                ++val;
                for(const auto el: ref)
                    con[el] = val;
            }
        }
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
            data_(nbits, nhashes << l2sz), updater_(seed),
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
        std::fprintf(stderr, "%i bits for each number, %u is log2 size of each table, %i is the number of subtables\n", nbits, l2sz, nhashes);
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
    void add_conservative(const uint64_t val) {
        std::vector<uint64_t> indices, best_indices;
        indices.reserve(nhashes_);
        unsigned nhdone = 0;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        Space::VType vb = Space::set1(val), tmp, mask = Space::set1(mask_);
        while((int)nhashes_ - (int)nhdone >= (ssize_t)Space::COUNT) {
            tmp = Space::and_fn(mask.simd_, HashStruct()(Space::xor_fn(vb.simd_, Space::load(sptr++))));
            tmp.for_each([&](uint64_t &subval){
                indices.push_back(subval + nhdone++ * subtbl_sz_);
            });
        }
        while(nhdone < nhashes_) {
            indices.push_back((HashStruct()(val ^ seeds_[nhdone]) & mask_) + nhdone * subtbl_sz_);
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
            Space::VType tmp = HashStruct()(Space::xor_fn(Space::set1(val), Space::load(sptr++)));
            tmp.for_each([&](uint64_t &subval){
#if !NDEBUG
                std::fprintf(stderr, "Querying at position %u, with value %u", nhdone * subtbl_sz_ + (subval & mask_), unsigned(data_[subtbl_sz_ * nhdone++ + (subval & mask_)]));
#endif
                updater_(data_[subtbl_sz_ * nhdone++ + (subval & mask_)], max_tbl_val_);
            });
        }
        while(nhdone < nhashes_) {
            updater_(data_[subtbl_sz_ * nhdone + (HashStruct()(val ^ seeds_[nhdone]) & mask_, max_tbl_val_)]);
            ++nhdone;
        }
    }
    uint64_t est_count(uint64_t val) {
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        uint64_t count = std::numeric_limits<uint64_t>::max(), nhdone = 0;
        static const Space::VType and_val = Space::set1(mask_), vb = Space::set1(val);
        Space::VType tmp;
        while(nhashes_ - nhdone > Space::COUNT) {
            tmp = Space::and_fn(HashStruct()(Space::xor_fn(vb.simd_, Space::load(sptr++))), and_val.simd_);
            tmp.for_each([&](uint64_t &subval){
                count = std::min(count, uint64_t(data_[subval + subtbl_sz_ * nhdone++]));
                ++nhdone;
            });
        }
        while(nhdone < nhashes_) {
            uint64_t ind = index(HashStruct()(val ^ seeds_[nhdone]), nhdone);
            count = std::min(count, uint64_t(data_[ind]));
            ++nhdone;
        }
        return updater_.est_count(count);
    }
};

using cmbf_t = cmbfbase_t<>;

} // namespace cmbf
} // namespace sketch
