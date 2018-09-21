#pragma once
#include "hll.h" // For common.h and clz functions
#include "aesctr/aesctr.h"

/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  HyperMinHash
 */

namespace sketch::minhash {
using namespace common;
using namespace hll;

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

template<typename T, typename SizeType=uint32_t, typename=::std::enable_if_t<::std::is_integral_v<T>>, typename=::std::enable_if_t<::std::is_integral_v<SizeType>>>
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

template<typename T, typename Hasher=common::WangHash, typename SizeType=uint32_t>
class KMinHash: public AbstractMinHash<T, SizeType> {
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;
    std::vector<T, common::Allocator<uint64_t>> hashes_;
    NO_ADDRESS Hasher hf_;
    // Uses k hash functions with seeds.
    // TODO: reuse code from hll/bf for vectorized hashing.
public:
    KMinHash(size_t nkeys, size_t sketch_size, uint64_t seedseed=137, Hasher &&hf=Hasher()):
        AbstractMinHash<T, SizeType>(sketch_size),
        hashes_(nkeys),
        hf_(std::move(hf))
    {
        aes::AesCtr<uint64_t, 4> gen(seedseed);
        seeds_.reserve(nseeds());
        while(seeds_.size() < nseeds()) seeds_.emplace_back(gen());
        throw std::runtime_error("NotImplemented.");
    }
    size_t nseeds() const {return this->nvals() * sizeof(uint64_t) / sizeof(T);}
    size_t nhashes_per_seed() const {return sizeof(uint64_t) / sizeof(T);}
    size_t nhashes_per_vector() const {return sizeof(uint64_t) / sizeof(Space::Type);}
    void addh(T val) {
        // Hash stuff.
        // Then use a vectorized minimization comparison instruction
    }
    SET_SKETCH(hashes_)
};

template<typename T, typename Hasher=common::WangHash, typename SizeType=uint32_t>
class RangeMinHash: public AbstractMinHash<T, SizeType> {
    NO_ADDRESS Hasher hf_;
    // Consider using a memory pool for locality.
    // These minimizers are not
    std::set<T> minimizers_;

    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher()):
        AbstractMinHash<T, SizeType>(sketch_size), hf_(std::move(hf))
    {
        throw std::runtime_error("NotImplemented.");
    }
    void addh(T val) {
        // Not vectorized, can be improved.
        val = hf_(val);
        if(auto it = minimizers_.find(val); it != minimizers_.end())
            return;
        else 
            minimizers_.insert(it, val), minimizers_.erase(minimizers_.begin());
    }
    SET_SKETCH(minimizers_)
};

template<typename T, typename Hasher=WangHash, typename SizeType=uint32_t, typename VectorType=std::vector<uint64_t, Allocator<uint64_t>>>
class HyperMinHash {
    // VectorType must support assignment and other similar operations.
    std::vector<uint8_t, Allocator<uint8_t>> core_; // HyperLogLog core
    VectorType mcore_;                              // Minimizers core
    uint32_t p_, q_, r_;
    static constexpr uint64_t seeds [] __attribute__ ((aligned (16))) = {0x611890f6d10bf441, 0x430c0277b68144b5};
public:
    template<typename... Args>
    HyperMinHash(unsigned p, unsigned q, unsigned r, Args &&...args):
        core_(1ull << p), p_(p), q_(q), r_(r), VectorType(std::forward<Args>(args)...) {
#if !NDEBUG
        print_params();
#endif
    }
    auto p() const {return p_;}
    auto q() const {return q_;}
    auto r() const {return r_;}
    int print_params(std::FILE *fp=stderr) const {
        return std::fprintf(fp, "p: %u. q: %u. r: %u.\n", p(), q(), r());
    }
    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (p_ - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (p_ - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
#if LZ_COUNTER
        ++clz_counts_[clz(((hashval << 1)|1) << (np_ - 1)) + 1];
#endif
    }

    INLINE void addh(uint64_t element) {
        const __m128 i = _mm_set1_epi64x(element);
        add(Hasher()(i));
    }
    INLINE void addh(VType element) {
        element.for_each([&](auto &el) {this->addh(element);});
    }
#if sizeof(VType) > sizeof(__m128i)
    INLINE void add(VType element) {
        for(__m128i *p = (__m128i *)&element, *e = (__m128i *)(&element + 1);p < e; add(*p++));
    }
#endif
    INLINE void add(__m128i hashval) {
        uint64_t arr[2];
        static_assert(sizeof(hashval) == sizeof(arr), "Size sanity check");
        std::memcpy(&arr[0], &hashval, sizeof(hashval));
        const uint32_t index(arr[0] >> q());
        const uint8_t lzt(clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
        if(core_[index] < lzt) {
            core_[index] = lzt;
            mcore_[index] = arr[1];
        } else if(core_[index] == lzt) {
            if(mcore_[index] < arr[1])
                mcore_[index] = arr[1];
        }
    }
    // TODO: jaccard index support
};

}
