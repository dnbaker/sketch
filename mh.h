#pragma once
#include "./common.h"

/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  HyperMinHash
 */

namespace sketch::minhash {

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

template<typename T, typename SizeType=::std::uint32_t, ASSERT_INT_T(T), ASSERT_INT_T(SizeType)>
class AbstractMinHash {
    SizeType ss_;
    AbstractMinHash(SizeType sketch_size): ss_(sketch_size) {}
    auto &sketch();
    const auto &sketch() const;
    auto nvals() const {return ss_;}
    std::vector<T, common::Allocator<T>> sketch2vec() const {
        std::vector<T, common::Allocator<T>> ret;
        auto &s = sketch();
        ret.reserve(s.size());
        for(const auto el: this->sketch()) ret.emplace_back(el);
        return ret;
    }
};

template<typename T, typename Hasher=WangHash, typename SizeType=::std::uint32_t>
class KMinHash: public AbstractMinHash<T, SizeType> {
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;
    std::vector<T, common::Allocator<uint64_t>> hashes_;
    Hasher hf_;
    // Uses k hash functions with seeds.
    // TODO: reuse code from hll/bf for vectorized hashing.
public:
    KMinHash(size_t nkeys, size_t sketch_size, uint64_t seedseed=137, Hasher &&hf=Hasher()):
        AbstractMinHash(sketch_size),
        hashes_(nkeys),
        hf_(std::move(hf))
    {
        aes::AesCtr<u64, 4> gen(seedseed);
        seeds_.reserve(nseeds());
        while(seeds_.size() < nseeds()) seeds_.emplace_back(gen());
        throw std::runtime_error("NotImplemented.");
    }
    size_t nseeds() const {return this->nvals() * sizeof(uint64_t) / sizeof(T);}
    size_t nhashes_per_seed() const {return sizeof(uint64_t) / sizeof(T);}
    size_t nhashes_per_vector() const {return sizeof() / sizeof(Space::Type);}
    void addh(T val) {
        // Hash stuff.
        // Then use a vectorized minimization comparison instruction
    }
    SET_SKETCH(hashes_)
};

template<typename T, typename Hasher=WangHash, typename SizeType=::std::uint32_t>
class RangeMinHash: public AbstractMinHash<T, SizeType> {
    Hasher hf_;
    // Consider using a memory pool for locality.
    // These minimizers are not
    std::set<T> minimizers;

    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher()):
        AbstractMinHash<T, SizeType>(sketch_size), hf_(std::move(hf))
    {
        throw std::runtime_error("NotImplemented.");
    }
    void addh(T val) {
        // Not vectorized, can be improved.
        val = hf_(val);
        if(auto it = minimizers.find(val); it != minimizers.end())
            return;
        else 
            minimizers.insert(it, val), minimizers.erase(minimizers.begin());
    }
    SET_SKETCH(minimizers_)
};

template<typename T, typename Hasher=WangHash, typename SizeType=::std::uint32_t>
class HyperMinHash {
    // TODO: this
};

}
