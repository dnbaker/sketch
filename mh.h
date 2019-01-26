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

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

namespace detail {


// From BinDash https://github.com/zhaoxiaofei/bindash

static inline uint64_t univhash2(uint64_t s, uint64_t t) {
    uint64_t x = (1009) * s + (1000*1000+3) * t;
    return (48271 * x + 11) % ((1ULL << 31) - 1);
}

inline int densifybin(std::vector<uint64_t> &hashes) {
    uint64_t min = hashes.front(), max = min;
    for(auto it(hashes.begin() + 1); it != hashes.end(); ++it) {
        min = std::min(min, *it);
        max = std::max(max, *it);
    }
    if (UINT64_MAX != max) { return 0; }  // Full sketch
    if (UINT64_MAX == min) { return -1; } // Empty sketch
    for (uint64_t i = 0; i < hashes.size(); i++) {
        // This is quadratic w.r.t. the number of hashes
        uint64_t j = i;
        uint64_t nattempts = 0;
        while (UINT64_MAX == hashes[j])
            j = univhash2(i, nattempts++) % hashes.size();
        hashes[i] = hashes[j];
    }
    return 1;
}

static constexpr double HMH_C = 0.169919487159739093975315012348;

template<typename FType, typename=typename std::enable_if<std::is_floating_point<FType>::value>::type>
inline FType beta(FType v) {
    FType ret = -0.370393911 * v;
    v = std::log1p(v);
    FType poly = v * v;
    ret += 0.070471823 * v;
    ret += 0.17393686 * poly;
    poly *= v;
    ret += 0.16339839 * poly;
    poly *= v;
    ret -= 0.09237745 * poly;
    poly *= v;
    ret += 0.03738027 * poly;
    poly *= v;
    ret += -0.005384159 * poly;
    poly *= v;
    ret += 0.00042419 * poly;
    return ret;
}

} // namespace detail

template<typename T, typename Cmp=std::greater<T>>
class AbstractMinHash {
protected:
    uint64_t ss_;
    AbstractMinHash(uint64_t sketch_size): ss_(sketch_size) {}
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


template<typename Container, typename Cmp=typename Container::key_compare>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    std::uint64_t ret = 0;
    auto it1 = c1.begin();
    auto it2 = c2.begin();
    const auto e1 = c1.end();
    const auto e2 = c2.end();
    for(;;) {
        if(cmp(*it1, *it2)) {
            if(++it1 == e1) break;
        } else if(cmp(*it2, *it1)) {
            if(++it2 == e2) break;
        } else {
            ++ret;
            if(++it1 == e1 || ++it2 == e2) break;
        }
    }
    return ret;
}


template<typename T, typename Cmp=std::greater<T>, typename Hasher=common::WangHash>
class KMinHash: public AbstractMinHash<T, Cmp> {
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;
    std::vector<T, common::Allocator<uint64_t>> hashes_;
    Hasher hf_;
    // Uses k hash functions with seeds.
    // TODO: reuse code from hll/bf for vectorized hashing.
public:
    KMinHash(size_t nkeys, size_t sketch_size, uint64_t seedseed=137, Hasher &&hf=Hasher()):
        AbstractMinHash<T, Cmp>(sketch_size),
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

template<typename T, typename Cmp> class FinalRMinHash; // Forward definition
/*
The sketch is the set of minimizers.

*/
template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash
        >
class RangeMinHash: public AbstractMinHash<T, Cmp> {
protected:
    Hasher hf_;
    Cmp cmp_;
    ///using HeapType = std::priority_queue<T, std::vector<T, Allocator<T>>>;
    std::set<T, Cmp> minimizers_; // using std::greater<T> so that we can erase from begin()

public:
    using final_type = FinalRMinHash<T, Cmp>;
    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()):
        AbstractMinHash<T, Cmp>(sketch_size), hf_(std::move(hf)), cmp_(std::move(cmp))
    {
    }
    auto rbegin() const {return minimizers_.rbegin();}
    auto rbegin() {return minimizers_.rbegin();}
    T max_element() const {
        return *begin();
    }
    T min_element() const {
        return *rbegin();
    }
    INLINE void addh(T val) {
        val = hf_(val);
        this->add(val);
    }
    INLINE void add(T val) {
        if(minimizers_.size() == this->ss_) {
            if(cmp_(max_element(), val)) {
                minimizers_.insert(val);
                if(minimizers_.size() > this->ss_) minimizers_.erase(minimizers_.begin());
            }
        } else minimizers_.insert(val);
    }
    template<typename T2>
    INLINE void addh(T2 val) {
        val = hf_(val);
        CONST_IF(std::is_same<T2, T>::value) {
            add(val);
        } else {
            static_assert(sizeof(val) % sizeof(T) == 0, "val must be the same size or greater than inserted element.");
            T *ptr = reinterpret_cast<T *>(&val);
            for(unsigned i = 0; i < sizeof(val) / sizeof(T); add(ptr[i++]));
        }
    }
    auto begin() {return minimizers_.begin();}
    const auto begin() const {return minimizers_.begin();}
    auto end() {return minimizers_.end();}
    const auto end() const {return minimizers_.end();}
    template<typename C2>
    size_t intersection_size(const C2 &o) const {
        return minhash::intersection_size(o, *this, Cmp());
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = this->intersection_size(o);
        return is / (minimizers_.size() + o.size() - is);
    }
    template<typename Container>
    Container to_container() const {
        if(this->ss_ != size()) // If the sketch isn't full, add UINT64_MAX to the end until it is.
            minimizers_.resize(this->ss_, std::numeric_limits<uint64_t>::max());
        return Container(std::rbegin(minimizers_), std::rend(minimizers_.end()));
    }
    void clear() {
        decltype(minimizers_)().swap(minimizers_);
    }
    final_type finalize() const {
        std::vector<T> reta(minimizers_.begin(), minimizers_.end());
        reta.insert(reta.end(), this->ss_ - reta.size(), std::numeric_limits<uint64_t>::max());
        return final_type{std::move(reta)};
    }
    std::vector<T> mh2vec() const {return to_container<std::vector<T>>();}
    size_t size() const {return minimizers_.size();}
    using key_compare = typename decltype(minimizers_)::key_compare;
    SET_SKETCH(minimizers_)
};

namespace weight {
struct EqualWeight {
    template<typename T>
    constexpr double operator()(T &x) const {return 1.;}
};
}

template<typename T, typename Cmp>
struct FinalRMinHash {
    std::vector<T> first;
    Cmp cmp;
    size_t intersection_size(const FinalRMinHash &o) const {
        return minhash::intersection_size(first, o.first, Cmp());
    }
    double jaccard_index(const FinalRMinHash &o) const {
        double is = intersection_size(o);
        return is / ((size() << 1) - is);
    }
    double cardinality() const {
        std::vector<T> diffs;
        for(auto i1 = first.begin(), i2 = i1; ++i2 != first.end();++i1) {
            diffs.push_back(*i1 - *i2);
        }
        sort::insertion_sort(diffs.begin(), diffs.end());
        return double(1ull << 63) / (diffs[diffs.size() >> 1] + diffs[(diffs.size() >> 1) - 1]);
    }
#define I1D if(++i1 == lsz) break
#define I2D if(++i2 == lsz) break
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalRMinHash &o, const WeightFn &fn=WeightFn()) const {
        assert(o.size() == size());
        const size_t lsz = size();
        double denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(cmp(first[i1], o.first[i2])) {
                denom += fn(first[i1]);
                I1D;
            } else if(cmp(o.first[i2], first[i1])) {
                denom += fn(o.first[i2]);
                I2D;
            } else {
                const auto v1 = fn(first[i1]), v2 = fn(o.first[i2]);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1D; I2D;
            }
        }
        return num / denom;
    }
    FinalRMinHash(std::vector<T> &&first): first(std::move(first)), cmp() {}
    size_t size() const {return first.size();}
};


template<typename T, typename Cmp, typename CountType>
struct FinalCRMinHash: public FinalRMinHash<T, Cmp> {
    std::vector<CountType> second;
    size_t countsum() const {return std::accumulate(second.begin(), second.end(), size_t(0), [](auto sz, auto sz2) {sz += sz2;});}
    double histogram_intersection(const FinalCRMinHash &o) const {
        assert(o.size() == this->size());
        const size_t lsz = this->size();
        size_t denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(this->cmp(this->first[i1], o.first[i2])) {
                denom += second[i1];
                I1D;
            } else if(this->cmp(o.first[i2], this->first[i1])) {
                denom += o.second[i2];
                I2D;
            } else {
                const auto v1 = o.second[i2], v2 = second[i1];
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1D; I2D;
            }
        }
        return static_cast<double>(num) / denom;
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalCRMinHash &o, const WeightFn &fn=WeightFn()) const {
        assert(o.size() == this->size());
        const size_t lsz = this->size();
        double denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(this->cmp(this->first[i1], o.first[i2])) {
                denom += second[i1] * fn(this->first[i1]);
                I1D;
            } else if(this->cmp(o.first[i2], this->first[i1])) {
                denom += o.second[i2] * fn(o.first[i2]);
                I2D;
            } else {
                const auto v1 = second[i1] * fn(this->first[i1]), v2 = o.second[i2] * fn(o.first[i2]);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1D; I2D;
            }
        }
#undef I1D
#undef I2D
        return num / denom;
    }
    FinalCRMinHash(std::vector<T> &&first, std::vector<CountType> &&second): FinalRMinHash<T, Cmp>(std::move(first)), second(std::move(second)) {
        if(this->first.size() != this->second.size()) {
            char buf[512];
            std::sprintf(buf, "Illegal FinalCRMinHash: hashes and counts must have equal length. Length one: %zu. Length two: %zu", this->first.size(), second.size());
            throw std::runtime_error(buf);
        }
    }
};

template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash,
         typename CountType=uint32_t
        >
class CountingRangeMinHash: AbstractMinHash<T, Cmp> {
    struct VType {
        T first;
        mutable CountType second;
        inline bool operator<(const VType &b) const {
            return Cmp()(this->first , b.first);
        }
        inline bool operator>(const VType &b) const {
            return !Cmp()(this->first , b.first);
        }
        inline bool operator==(const VType &b) const {
            return this->first == b.first;
        }
        VType(T v, CountType c): first(v), second(c) {}
        VType(const VType &o): first(o.first), second(o.second) {}
    };
    Hasher hf_;
    Cmp cmp_;
    std::set<VType> minimizers_; // using std::greater<T> so that we can erase from begin()
public:
    const auto &min() const {return minimizers_;}
    using size_type = CountType;
    using final_type = FinalCRMinHash<T, Cmp, CountType>;
    auto size() const {return minimizers_.size();}
    auto begin() {return minimizers_.begin();}
    auto begin() const {return minimizers_.begin();}
    auto rbegin() {return minimizers_.rbegin();}
    auto rbegin() const {return minimizers_.rbegin();}
    auto end() {return minimizers_.end();}
    auto end() const {return minimizers_.end();}
    auto rend() {return minimizers_.rend();}
    auto rend() const {return minimizers_.rend();}
    CountingRangeMinHash(size_t n, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()): AbstractMinHash<T, Cmp>(n), hf_(std::move(hf)), cmp_(std::move(cmp)) {}
    INLINE void add(T val) {
        if(minimizers_.size() == this->ss_) {
            if(cmp_(begin()->first, val)) {
                auto it = minimizers_.find(VType(val, 0));
                if(it == minimizers_.end()) {
                    minimizers_.erase(begin());
                    minimizers_.insert(VType(val, CountType(1)));
                } else ++it->second;
            }
        } else minimizers_.insert(VType(val, CountType(1)));
    }
    INLINE void addh(T val) {
        val = hf_(val);
        this->add(val);
    }
    double histogram_intersection(const CountingRangeMinHash &o) const {
        assert(o.size() == size());
        size_t denom = 0, num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
#define I1D if(++i1 == minimizers_.end()) break
#define I2D if(++i2 == o.minimizers_.end()) break
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                denom += i1->second;
                I1D;
            } else if(cmp_(i2->first, i1->first)) {
                denom += i2->second;
                I2D;
            } else {
                const auto v1 = i1->second, v2 = i2->second;
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1D;
                I2D;
            }
        }
        return static_cast<double>(num) / denom;
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const CountingRangeMinHash &o, const WeightFn &fn) const {
        assert(o.size() == size());
        double denom = 0, num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                denom += (i1->second) * fn(i1->first);
                I1D;
            } else if(cmp_(i2->first, i1->first)) {
                denom += (i2->second) * fn(i2->first);
                I2D;
            } else {
                const auto v1 = i1->second * fn(i1->first), v2 = i2->second * fn(i2->first);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1D;
                I2D;
            }
        }
#undef I1D
#undef I2D
        return static_cast<double>(num) / denom;
    }
    final_type finalize() const {
        std::vector<T> reta; std::vector<CountType> retc;
        reta.reserve(this->ss_); retc.reserve(this->ss_);
        for(auto &p: minimizers_)
            reta.push_back(p.first), retc.push_back(p.second);
        assert(reta.size() == retc.size());
        size_t nelem = this->ss_;
        reta.insert(reta.end(), nelem, std::numeric_limits<T>::max());
        retc.insert(retc.end(), nelem, 0);
#if !NDEBUG
        for(size_t i = 0; i < size() - 1; ++i)
            assert(Cmp()(reta[i], reta[i + 1]));
#endif
        return final_type{std::move(reta), std::move(retc)};
    }
    template<typename Func>
    void for_each(const Func &func) const {
        for(const auto &i: minimizers_) {
            func(i);
        }
    }
    template<typename Func>
    void for_each(const Func &func) {
        for(auto &i: minimizers_) {
            func(i);
        }
    }
    void print() const {
        for_each([](auto &p) {std::fprintf(stderr, "key %s with value %zu\n", std::to_string(p.first).data(), size_t(p.second));});
    }
    template<typename C2>
    size_t intersection_size(const C2 &o) const {
        return minhash::intersection_size(o, *this, [&](auto &x, auto &y) {return cmp_(x.first, y.first);});
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = this->intersection_size(o);
        return is / (minimizers_.size() + o.size() - is);
    }
};

template<typename T=uint64_t, typename Hasher=WangHash>
class HyperMinHash {
    DefaultCompactVectorType core_;
    Hasher hf_;
    uint32_t p_, r_;
    uint64_t seeds_ [2] __attribute__ ((aligned (sizeof(uint64_t) * 2)));
#ifndef NOT_THREADSAFE
    std::mutex mutex_; // I should be able to replace most of these with atomics.
#endif
public:
    static constexpr uint32_t q() {return 6u;} // To hold popcount for a 64-bit integer.
    enum ComparePolicy {
        Manual = 0,
        U8 = 1,
        U16 = 2,
        U32 = 3,
        U64 = 4
    };
    auto &core() {return core_;}
    const auto &core() const {return core_;}
    uint64_t mask() const {
        return (uint64_t(1) << r_) - 1;
    }
    auto max_mhval() const {return mask();}
    template<typename... Args>
    HyperMinHash(unsigned p, unsigned r, Args &&...args):
        core_(r + q(), 1ull << p),
        hf_(std::forward<Args>(args)...),
        p_(p), r_(r),
        seeds_{0xB0BAF377C001D00DuLL, 0x430c0277b68144b5uLL} // Fully arbitrary seeds
    {
        std::fprintf(stderr, "Pointer for data: %p\n", static_cast<void *>(core_.get()));
        common::detail::zero_memory(core_); // Second parameter is a dummy for interface compatibility with STL
#if !NDEBUG
        std::fprintf(stderr, "p: %u. r: %u\n", p, r);
        print_params();
#endif
    }
    void clear() {
        std::memset(core_.get(), 0, core_.bytes());
    }
    HyperMinHash(const HyperMinHash &a): core_(a.r() + q(), 1ull << a.p()), hf_(a.hf_), p_(a.p_), r_(a.r_) {
        seeds_as_sse() = a.seeds_as_sse();
        assert(a.core_.bytes() == core_.bytes());
        std::memcpy(core_.get(), a.core_.get(), core_.bytes());
    }
    void print_all(std::FILE *fp=stderr) {
        for(size_t i = 0; i < core_.size(); ++i) {
            size_t v = core_[i];
            std::fprintf(stderr, "Index %zu has value %d for lzc and %d for remainder, with full value = %zu\n", i, int(get_lzc(v)), int(get_mhr(v)), size_t(core_[i]));
        }
    }
    auto minimizer_size() const {
        return r_ + q();
    }
    ComparePolicy simd_policy() const {
        switch(minimizer_size()) {
            case 8: return ComparePolicy::U8;
            case 16: return ComparePolicy::U16;
            case 32: return ComparePolicy::U32;
            case 64: return ComparePolicy::U64;
            default: return ComparePolicy::Manual;
        }
        __builtin_unreachable();
    }
    // Encoding and decoding table entries
    auto get_lzc(uint64_t entry) const {
        return entry >> r_;
    }
    auto get_mhr(uint64_t entry) const {
        return entry & max_mhval();
    }
    template<typename I1, typename I2,
             typename=typename std::enable_if<std::is_integral<I1>::value && std::is_integral<I1>::value>::type>
    auto encode_register(I1 lzc, I2 min) const {
        // We expect that min has already been masked so as to eliminate unnecessary operations
        assert(min <= max_mhval());
        return (uint64_t(lzc) << r_) | min;
    }
    std::array<uint64_t, 64> sum_counts() const {
        // TODO: this
        // Note: we have whip out more complicated vectorized maxes for
        // widths of 16, 32, or 64
        std::array<uint64_t, 64> ret{0};
        using hll::detail::SIMDHolder;
        if(core_.bytes() >= sizeof(SIMDHolder)) {
            switch(simd_policy()) {
                case U8:
                    const Space::Type mask = Space::set1(UINT64_C(0x3f3f3f3f3f3f3f3f));
                    for(const SIMDHolder *ptr = reinterpret_cast<const SIMDHolder *>(core_.get()), *eptr = reinterpret_cast<const SIMDHolder *>(core_.get() + core_.bytes());
                        ptr != eptr; ++ptr) {
                        auto tmp = *ptr;
                        tmp = Space::and_fn(Space::srli(*reinterpret_cast<VType *>(&tmp), r_), mask);
                        tmp.inc_counts(ret);
                    }
                    break;
#if 0
                case U16:
                case U32:
                case U64:
#endif
#define MANUAL_CORE \
            for(const auto i: core_) {\
                uint8_t lzc = get_lzc(i);\
                if(__builtin_expect(lzc > 64, 0)) {std::fprintf(stderr, "Value for %d should not be %d\n", int(i), int(get_lzc(i))); std::exit(1);}\
                ++ret[lzc];\
            }
                case Manual: default: MANUAL_CORE
            }
        } else {
            MANUAL_CORE
#undef MANUAL_CORE
        }
        return ret;
    }
    double estimate_hll_portion(double relerr=1e-2) const {
        return hll::detail::ertl_ml_estimate(this->sum_counts(), p(), q(), relerr);
    }
    double report(double relerr=1e-2) const {
        const auto csum = this->sum_counts();
#if __cplusplus >= 201703L
        if(double est = hll::detail::ertl_ml_estimate(csum, p(), 64 - p(), relerr);est < static_cast<double>(core_.size() << 10))
            return est;
#else
        double est = hll::detail::ertl_ml_estimate(csum, p(), 64 - p(), relerr);
        if(est < static_cast<double>(core_.size() << 10))
            return est;
#endif
        const double mhinv = 1. / max_mhval();
        double sum = csum[0] * (1 + static_cast<double>((uint64_t(1) << p()) - get_mhr(core_[0])) * mhinv);
        for(int i = 1; i < static_cast<int64_t>(csum.size()); ++i)
            sum += std::ldexp(csum[i], -i) * (1. + static_cast<double>((uint64_t(1) << p()) - get_mhr(core_[i])) * mhinv);
        return sum ? static_cast<double>(std::pow(core_.size(), 2)) / sum: std::numeric_limits<double>::infinity();
    }
    auto p() const {return p_;}
    auto r() const {return r_;}
    void set_seeds(uint64_t seed1, uint64_t seed2) {
        seeds_[0] = seed1; seeds_[1] = seed2;
    }
    void set_seeds(__m128i s) {
        seeds_as_sse() = s;
    }
    int print_params(std::FILE *fp=stderr) const {
        return std::fprintf(fp, "p: %u. q: %u. r: %u.\n", p(), q(), r());
    }
    const __m128i &seeds_as_sse() const {return *reinterpret_cast<const __m128i *>(seeds_);}
    __m128i &seeds_as_sse() {return *reinterpret_cast<__m128i *>(seeds_);}

    INLINE void addh(uint64_t element) {
        add(hf_(_mm_set1_epi64x(element) ^ seeds_as_sse()));
    }
    template<typename ET>
    INLINE void addh(ET element) {
        element.for_each([&](uint64_t el) {this->addh(el);});
    }
    HyperMinHash &operator+=(const HyperMinHash &o) {
        // This needs:
        // Vectorized maxes
        for(size_t i(0); i < core_.size(); ++i) {
            if(core_[i] < o.core_[i]) core_[i] = o.core_[i];
            // This can also be accelerated for specific minimizer sizes
        }
        return *this;
    }
    HyperMinHash &operator=(const HyperMinHash &a)
#if 0
    {
        core_ = DefaultCompactVectorType(q() + a.r(), 1 << a.p());
        seeds_as_sse() = a.seeds_as_sse();
        std::memcpy(core_.get(), a.core_.get(), core_.bytes());
    }
#else
    = delete;
#endif
    HyperMinHash(const HyperMinHash &a, const HyperMinHash &b): HyperMinHash(a)
    {
        *this += b;
    }
    HyperMinHash operator+(const HyperMinHash &a) const {
        if(__builtin_expect(a.p() != p() || a.q() != q(), 0)) throw std::runtime_error("Could not merge sketches of differing parameter sets");
        HyperMinHash ret(*this);
        ret += a;
        return ret;
    }
    //HyperMinHash(const HyperMinHash &) = delete;
    HyperMinHash(HyperMinHash &&) = default;
    HyperMinHash &operator=(HyperMinHash &&) = default;
    INLINE void add(__m128i hashval) {
    // TODO: Consider looking for a way to use the leading zero count to store the rest of a key
    // Not sure this is valid for the purposes of an independent hash.
        uint64_t arr[2];
        static_assert(sizeof(hashval) == sizeof(arr), "Size sanity check");
        std::memcpy(&arr[0], &hashval, sizeof(hashval));
        const uint64_t index(reinterpret_cast<uint64_t *>(&hashval)[0] >> (64 - p())),
                         lzt(hll::clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
        //std::fprintf(stderr, "Calling hash on thing. Size of core: %zu. Index: %zu\n", index, core_.size());
        //std::fprintf(stderr, "Calling hash on %zu\n", size_t(core_[index]));
        const uint64_t inserted_val = encode_register(lzt, reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval());
        //std::fprintf(stderr, "lzc: %d. oval: %zu. full val: %zu\n", int(get_lzc(inserted_val)), get_mhr(inserted_val), size_t(inserted_val));
        assert(get_lzc(inserted_val) == lzt);
        assert((reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) == get_mhr(inserted_val));
        //const uint64_t inserted_val = (reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) | (lzt << r_);
        //uint64_t oind = core_[index];
        //std::fprintf(stderr, "oind: %" PRIu64 "\n", oind);
        if(core_[index] < inserted_val) { // Consider other functions for specific register sizes.
            core_[index] = inserted_val;
            assert(encode_register(lzt, get_mhr(inserted_val)) == inserted_val);
            assert(lzt == get_lzc(inserted_val));
            //std::fprintf(stderr, "Register is now set to %zu with clz %d, which should match %d from other clz\n", size_t(core_[index]), int(get_lzc(inserted_val)), int(lzt));
        }
        //std::fprintf(stderr, "Register is now the same at%zu with clz %d, \n", size_t(core_[index]), int(get_lzc(core_[index])));
    }
    double jaccard_index(const HyperMinHash &o) const {
        size_t C = 0, N = 0;
        std::fprintf(stderr, "core size: %zu\n", core_.size());
#if EXPERIMENTAL_SIMD_CMP
        switch(simd_policy()) { // This can be accelerated for specific sizes
            default: [[fallthrough]];
            U8:      [[fallthrough]]; // 2-bit minimizers. TODO: write this
            U16:     [[fallthrough]]; // 10-bit minimizers. TODO: write this
            U32:     [[fallthrough]]; // 26-bit minimizers. TODO: write this
            U64:     [[fallthrough]]; // 58-bit minimizers. TODO: write this
            Manual:
                for(size_t i = 0; i < core_.size(); ++i) {
                    C += (get_lzc(core_[i]) == get_lzc(o.core_[i]));
                    N += (core_[i] || o.core_[i]);
                }
            break;
        }
#else
       for(size_t i = 0; i < core_.size(); ++i) {
           C += (get_lzc(core_[i]) == get_lzc(o.core_[i]));
           N += (core_[i] || o.core_[i]);
       }
#endif
        const double n = this->report(), m = o.report(), ec = expected_collisions(n, m);
        std::fprintf(stderr, "C: %zu. ec: %lf\n", C, ec);
        return C > ec ? (C - ec) / N: 0.;
    }
    double expected_collisions(double n, double m, bool easy_way=true) const {
        if(easy_way) {
            if(n < m) std::swap(n, m);
            if(std::log2(n) > ((1 << q()) + r()))
                throw std::range_error("Too high to calculate");
            if(std::log2(n) > p() + 5) {
                const double nm = n/m;
                const double phi = std::ldexp(nm, -4) / std::pow(1 + nm, 2);
#if !NDEBUG
                std::fprintf(stderr, "Using normal expected collisions method. p: %d, r: %d, phi: %lf, ret:%lf\n", p(), r(), phi, std::ldexp(0.169919487159739093975315012348, p() - r())  * phi);
#endif
                return std::ldexp(0.169919487159739093975315012348, p() - r())  * phi;
            }
        }
#if !NDEBUG
        std::fprintf(stderr, "Using slow expected collisions method\n");
#endif
        double x = 0.;
        for(size_t i = 1; i <= size_t(1) << p(); ++i) {
            for(size_t j = 1; j <= size_t(1) << r(); ++j) {
                double b1, b2;
                const int _p= p_, _r = r_; // Redeclaring as signed integers to avoid underflow
                if(i != size_t(1) << q()) {
                    b1 = std::ldexp((size_t(1) << r()) + j, -_p - _r - i);
                    b2 = std::ldexp((size_t(1) << r()) + j + 1, -_p - _r - i);
                } else {
                    b1 = std::ldexp(j, -p_ - _r - i - 1);
                    b2 = std::ldexp(j + 1, -p_ - _r - i - 1);
                }
                const double prx = std::pow(1.-b2, n) - std::pow(1.-b1, n);
                const double pry = std::pow(1.-b2, m) - std::pow(1.-b1, m);
                x += prx * pry;
            }
        }
        return std::ldexp(x, p());
    }
};

} // namespace minhash
namespace mh = minhash;
} // namespace sketch
