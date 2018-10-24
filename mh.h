#pragma once
#include <mutex>
//#include <queue>
#include "hll.h" // For common.h and clz functions
#include "aesctr/aesctr.h"

/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  HyperMinHash
 */

namespace sketch {
namespace minhash {
using namespace common;
using namespace hll;

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

template<typename T, typename SizeType=uint32_t, typename=::std::enable_if_t<::std::is_integral_v<T>>, typename=::std::enable_if_t<::std::is_integral_v<SizeType>>>
class AbstractMinHash {
protected:
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


template<typename Container1, typename Container2>
std::uint64_t intersection_size(const Container1 &c1, const Container2 &c2) {
    // These containers must be sorted.
    std::uint64_t ret = 0;
    auto it1 = c1.begin();
    auto it2 = c2.end();
    const auto e1 = c1.end();
    const auto e2 = c2.end();
    start:
#if HAVE_SPACESHIP_OPERATOR
    switch(*it1 <=> *it2) {
        case -1: if(++it1 == e1) goto end; break;
        case 0: ++ret; if(++it1 == e1 || ++it2 == e2) goto end; break;
        case 1:               if(++it2 == e2) goto end; break;
        default: __builtin_unreachable();
    }
#else
    if(*it1 < *it2) {
        if(++it1 == e1) goto end;
    } else if(*it1 > *it2) {
        if(++it2 == e2) goto end;
    } else {
        ++ret;
        if(++it1 == e1 || ++it2 == e2) goto end;
    }
#endif
    goto start;
    end:
    return ret;
}


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
        throw std::runtime_error("NotImplemented.");
        // Hash stuff.
        // Then use a vectorized minimization comparison instruction
    }
    SET_SKETCH(hashes_)
};

/*
The sketch is the set of minimizers.

*/
template<typename T,
         typename Hasher=common::WangHash,
         typename SizeType=uint32_t,
         bool force_non_int=false, // In case you're absolutely sure you want to use a non-integral value
         typename=std::enable_if_t<
            force_non_int || std::is_arithmetic_v<T>
         >
        >
class RangeMinHash: public AbstractMinHash<T, SizeType> {
    NO_ADDRESS Hasher hf_;
    ///using HeapType = std::priority_queue<T, std::vector<T, Allocator<T>>>;
    using HeapType = std::set<T, std::greater<T>>;
    HeapType minimizers_; // using std::greater<T> so that we can erase from begin()

public:
    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher()):
        AbstractMinHash<T, SizeType>(sketch_size), hf_(std::move(hf))
    {
    }
    void addh(T val) {
        // Not vectorized, can be improved.
        val = hf_(val);
        add(val);
    }
    void add(T val) {
        if(__builtin_expect(minimizers_.size() < this->ss_, 0))
            minimizers_.insert(val);
        else if(auto it = minimizers_.find(val); it == minimizers_.end()) {
            minimizers_.insert(it, val), minimizers_.erase(minimizers_.begin());
        }
    }
    template<typename T2>
    INLINE void addh(T2 val) {
        if constexpr(std::is_same_v<T2, T>) {
            val = hf_(val);
            add(val);
        } else {
            val = hf_(val);
            if constexpr(sizeof(T2) == VECTOR_WIDTH) {
                reinterpret_cast<vec::UType<T> *>(&val)->for_each([&](T sv) {add(sv);});
            } else {
                T *ptr = reinterpret_cast<T *>(&val);
                for(unsigned i = 0; i < sizeof(val) / sizeof(T); add(ptr[i++]));
            }
        }
    }
    auto begin() {return minimizers_.begin();}
    const auto begin() const {return minimizers_.begin();}
    auto end() {return minimizers_.end();}
    const auto end() const {return minimizers_.end();}
    template<typename C2>
    size_t intersection_size(const C2 &o) const {
        auto it = this->minimizers_.begin();
        auto oit = o.begin();
        size_t ret = 0;
        while(it != minimizers_.end() && oit != o.end()) {
            if(*it == *oit) ++it, ++oit, ++ret;
            else if(*it < *oit) ++it;
            else ++oit;
        }
        return ret;
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = intersection_size(o);
        return is / (minimizers_.size() + o.size() - is);
    }
    template<typename Container>
    Container to_container() const {
        return Container(std::rbegin(minimizers_), std::rend(minimizers_.end()));
    }
    void clear() {
        HeapType tmp;
        std::swap(tmp, minimizers_);
    }
    std::vector<T> mh2vec() const {return to_container<std::vector<T>>();}
    size_t size() const {return minimizers_.size();}
    SET_SKETCH(minimizers_)
};

template<typename T=uint64_t, typename Hasher=WangHash, typename SizeType=uint32_t,
         typename VectorType=std::vector<T, Allocator<T>>>
class HyperMinHash {
    // VectorType must support assignment and other similar operations.
    std::vector<uint8_t, Allocator<uint8_t>> core_; // HyperLogLog core
    VectorType mcore_;                              // Minimizers core
    uint32_t p_, q_, r_;
    uint64_t seeds_ [2] __attribute__ ((aligned (16)));
#ifndef NOT_THREADSAFE
    std::mutex mutex_;
#endif
    static_assert(std::is_same_v<std::decay_t<decltype(mcore_[0])>, T>, "Vector must derefernce to T.");
public:
    auto &core() {return core_;}
    uint64_t mask() const {
        return uint64_t(1) << r_ - 1;
    }
    const auto &core() const {return core_;}
    template<typename... Args>
    HyperMinHash(unsigned p, unsigned q=0, unsigned r=64, Args &&...args):
        core_(1ull << p),
        mcore_(std::forward<Args>(args)...),
        p_(p), q_(q ? q: 64 - p), r_(r),
        seeds_{0xB0BAF377C001D00DuLL, 0x430c0277b68144b5}
    {
        mcore_.resize(core_.size());
#if !NDEBUG
        print_params();
#endif
    }
    uint64_t max_mhval() const {
        return (uint64_t(1) << q_) - 1;
    }
    double estimate_hll_portion(double relerr=1e-2) const {
        return hll::detail::ertl_ml_estimate(detail::sum_counts(*this), p(), q(), relerr);
    }
    double report(double relerr=1e-2) const {
        const auto csum = detail::sum_counts(*this);
        if(double est = hll::detail::ertl_ml_estimate(csum, p(), q(), relerr);est < static_cast<double>(core_.size() << 10))
            return est;
        double rsum = 0.;
        for(size_t i(0); i < csum.size(); ++i)
            rsum += std::ldexp(1., -csum[i]) * (1. + (static_cast<double>(mcore_[i]) / (max_mhval())));
        return rsum ? static_cast<double>(core_.size() * core_.size()) / rsum: std::numeric_limits<double>::infinity();
    }
    auto p() const {return p_;}
    auto q() const {return q_;}
    auto r() const {return r_;}
    void set_seeds(uint64_t seed1, uint64_t seed2) {
        seeds_[0] = seed1; seeds_[1] = seed2;
    }
    int print_params(std::FILE *fp=stderr) const {
        return std::fprintf(fp, "p: %u. q: %u. r: %u.\n", p(), q(), r());
    }
    const __m128i &seeds_as_sse() const {return *reinterpret_cast<const __m128i *>(seeds_);}

    INLINE void addh(uint64_t element) {
        __m128i i = _mm_set1_epi64x(element);
        i ^= seeds_as_sse();
        i = Hasher()(i);
        add(i);
    }
    template<typename ET>
    INLINE void addh(ET element) {
        element.for_each([&](auto &el) {this->addh(element);});
    }
    HyperMinHash &operator+=(const HyperMinHash &o) {
        for(size_t i(0); i < core_.size(); ++i) {
            if(core_[i] < o.core_[i]) {
                core_[i] = o.core_[i];
                mcore_[i] = o.mcore_[i];
            } else if(core_[i] == o.core_[i]) {
                using std::min;
                mcore_[i] = min(mcore_[i], o.mcore_[i]);
            }
        }
        return *this;
    }
    HyperMinHash(const HyperMinHash &a, const HyperMinHash &b): HyperMinHash(a.p(), a.q(), a.r())
    {
        *this += a;
        *this += b;
    }
    HyperMinHash(const HyperMinHash &) = delete;
    HyperMinHash &operator=(const HyperMinHash &) = delete;
    HyperMinHash(HyperMinHash &&) = default;
    HyperMinHash &operator=(HyperMinHash &&) = default;
    INLINE void add(__m128i hashval) {
    // TODO: Consider looking for a way to use the leading zero count to store the rest of a key
    // Not sure this is valid for the purposes of an independent hash.
        uint64_t arr[2];
        static_assert(sizeof(hashval) == sizeof(arr), "Size sanity check");
        std::memcpy(&arr[0], &hashval, sizeof(hashval));
        const uint32_t index(arr[0] >> q());
        const uint8_t lzt(clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
#ifndef NOT_THREADSAFE
        // This won't be optimal, but it's a patch to make it work.
        if(core_[index] <= lzt) {
            std::lock_guard<std::mutex> lock(mutex_);
            if(core_[index] < lzt) {
                core_[index] = lzt;
                mcore_[index] = arr[1];
            } else mcore_[index] = std::min(mcore_[index], arr[1]);
        }
#else
        if(core_[index] < lzt) {
            core_[index] = lzt;
            mcore_[index] = arr[1];
        } else if(core_[index] == lzt) mcore_[index] = std::min(arr[1], (uint64_t)mcore_[index]);
#endif
    }
    // TODO: jaccard index support
};

} // namespace minhash
namespace mh = minhash;
} // namespace sketch
