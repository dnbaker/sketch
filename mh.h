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

static constexpr double HMH_C = 0.169919487159739093975315012348;

template<typename FType, typename=std::enable_if_t<std::is_floating_point_v<FType>>>
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


template<typename Container, typename Cmp=typename Container::key_compare>
std::uint64_t intersection_size(const Container &c1, const Container &c2) {
    Cmp cmp;
    // These containers must be sorted.
    std::uint64_t ret = 0;
    auto it1 = c1.begin();
    auto it2 = c2.begin();
    const auto e1 = c1.end();
    const auto e2 = c2.end();
    start:
#if HAVE_SPACESHIP_OPERATOR
    switch(*it1 <=> *it2) {
        case -1: if(++it1 == e1) goto end;
        case 0: ++ret; if(++it1 == e1 || ++it2 == e2) goto end;
        case 1:               if(++it2 == e2) goto end;
        default: __builtin_unreachable();
    }
#else
    if(cmp(*it1, *it2)) {
        if(++it1 == e1) goto end;
    } else if(cmp(*it2, *it1)) {
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
        minimizers_.insert(val);
        if(minimizers_.size() > this->ss_)
            minimizers_.erase(minimizers_.begin());
    }
    template<typename T2>
    INLINE void addh(T2 val) {
        if constexpr(std::is_same_v<T2, T>) {
            val = hf_(val);
            add(val);
        } else {
            val = hf_(val);
            T *ptr = reinterpret_cast<T *>(&val);
            for(unsigned i = 0; i < sizeof(val) / sizeof(T); add(ptr[i++]));
        }
    }
    auto begin() {return minimizers_.begin();}
    const auto begin() const {return minimizers_.begin();}
    auto end() {return minimizers_.end();}
    const auto end() const {return minimizers_.end();}
#if 0
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
#endif
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = intersection_size(*this, o);
        return is / (minimizers_.size() + o.size() - is);
    }
    template<typename Container>
    Container to_container() const {
        if(this->ss_ != size()) // If the sketch isn't full, add UINT64_MAX to the end until it is.
            minimizers_.resize(this->ss_, std::numeric_limits<uint64_t>::max());
        return Container(std::rbegin(minimizers_), std::rend(minimizers_.end()));
    }
    void clear() {
        HeapType tmp;
        std::swap(tmp, minimizers_);
    }
    std::vector<T> mh2vec() const {return to_container<std::vector<T>>();}
    size_t size() const {return minimizers_.size();}
    SET_SKETCH(minimizers_)
    using key_compare = typename HeapType::key_compare;
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
        core_(r, 1ull << p),
        hf_(std::forward<Args>(args)...),
        p_(p), r_(r),
        seeds_{0xB0BAF377C001D00DuLL, 0x430c0277b68144b5uLL} // Fully arbitrary seeds
    {
        std::fprintf(stderr, "Pointer for data: %p\n", static_cast<void *>(core_.get()));
        common::detail::zero_memory(core_, 0); // Second parameter is a dummy for interface compatibility with STL
#if !NDEBUG
        std::fprintf(stderr, "p: %u. r: %u\n", p, r);
        print_params();
#endif
    }
    HyperMinHash(const HyperMinHash &a): core_(a.core_), p_(a.p_), r_(a.r_) {
        seeds_as_sse() = a.seeds_as_sse();
        std::memcpy(core_.get(), a.core_.get(), core_.bytes());
    }
    void print_all(std::FILE *fp=stderr) {
        for(size_t i = 0; i < core_.size(); ++i) {
            std::fprintf(stderr, "Index %zu has value %d for lzc and %d for remainder\n", i, get_lzc(core_[i]), get_mhr(core_[i]));
        }
    }
    static constexpr uint32_t q() {return 6u;} // To hold popcount for a 64-bit integer.
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
             typename=std::enable_if_t<std::is_integral_v<I1> && std::is_integral_v<I1>>>
    auto encode_register(I1 lzc, I2 min) const {
        // We expect that min has already been masked so as to eliminate unnecessary operations
        assert(min <= max_mhval());
        return (uint64_t(lzc) << r_) | min;
    }
    std::array<uint64_t, 64> sum_counts() const {
        if(__builtin_expect(r_ == 0, 0)) { // No remained, is a hyperloglog sketch
            return hll::detail::sum_counts(core_);
        }
        std::array<uint64_t, 64> ret;
        std::memset(&ret[0], 0, sizeof(ret));
        for(const auto i: core_) {
            if(get_lzc(i) > 64) {std::fprintf(stderr, "Value for %d should not be %d\n", i, get_lzc(i)); std::exit(1);}
            ++ret[get_lzc(i)];
        }
        return ret;
    }
    double estimate_hll_portion(double relerr=1e-2) const {
        return hll::detail::ertl_ml_estimate(this->sum_counts(), p(), 64 - p(), relerr);
    }
    double report(double relerr=1e-2) const {
        const auto csum = this->sum_counts();
        if(double est = hll::detail::ertl_ml_estimate(csum, p(), 64 - p(), relerr);est < static_cast<double>(core_.size() << 10))
            return est;
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
        for(size_t i(0); i < core_.size(); ++i) {
            if(core_[i] < o.core_[i]) core_[i] = o.core_[i];
            // This can also be accelerated for specific minimizer sizes
        }
        return *this;
    }
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
    HyperMinHash &operator=(const HyperMinHash &) = delete;
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
#if OLD_WAYYYYYYYYYYY
#ifndef NOT_THREADSAFE
        // This won't be optimal, but it's a patch to make it work.
        // I bet there's a way to make this lockfree.
        if(core_[index] <= lzt) {
            std::lock_guard<std::mutex> lock(mutex_);
            if(core_[index] < lzt) {
                core_[index] = lzt;
                mcore_[index] = reinterpret_cast<uint64_t *>(&hashval)[1];
            } else mcore_[index] = std::min(mcore_[index], reinterpret_cast<uint64_t *>(&hashval)[1]);
        }
#else
        if(core_[index] < lzt) {
            core_[index] = lzt;
            mcore_[index] = reinterpret_cast<uint64_t *>(&hashval)[1];
        } else if(core_[index] == lzt) mcore_[index] = std::min(reinterpret_cast<uint64_t *>(&hashval)[1], (uint64_t)mcore_[index]);
#endif  // NOT_THREADSAFE
#else
        // We also use max instead of min for "minimizers" because we can pack comparisons.
        const uint64_t inserted_val = encode_register(lzt, reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval());
        std::fprintf(stderr, "lzc: %d. oval: %zu\n", get_lzc(inserted_val), get_mhr(inserted_val));
        //const uint64_t inserted_val = (reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) | (lzt << r_);
        //uint64_t oind = core_[index];
        //std::fprintf(stderr, "oind: %" PRIu64 "\n", oind);
        if(core_[index] < inserted_val) // Consider other functions for specific register sizes.
            core_[index] = inserted_val;
#endif // #if OLD_WAYYYYYYYYYYY
        std::fprintf(stderr, "Called hash on thing\n");
        const uint64_t newval = core_[index];
        assert(core_[index] >= newval);
    }
    double jaccard_index(const HyperMinHash &o) const {
        size_t C = 0, N = 0;
        std::fprintf(stderr, "core size: %zu\n", core_.size());
        switch(simd_policy()) {
            default:
            U8:  [[fallthrough]] // 2-bit minimizers. TODO: write this
            U16: [[fallthrough]] // 10-bit minimizers. TODO: write this
            U32: [[fallthrough]] // 26-bit minimizers. TODO: write this
            U64: [[fallthrough]] // 58-bit minimizers. TODO: write this
            Manual:
                for(size_t i = 0; i < core_.size(); ++i) {
                    std::fprintf(stderr, "lzcs: %u, %u\n", get_lzc(core_[i]), get_lzc(o.core_[i]));
                    C += (get_lzc(core_[i]) == get_lzc(o.core_[i]));
                    N += (core_[i] || o.core_[i]);
                }
            break;
        }
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
                std::fprintf(stderr, "Using normal way\n");
                return std::ldexp(0.169919487159739093975315012348, p() - r())  * phi;
            }
        }
        std::fprintf(stderr, "Using slow way\n");
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
        std::fprintf(stderr, "Successfully slow wayed\n");
        return std::ldexp(x, p());
    }
};

} // namespace minhash
namespace mh = minhash;
} // namespace sketch
