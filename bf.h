#ifndef CRUEL_BLOOM_H__
#define CRUEL_BLOOM_H__
#include "common.h"


#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH 1
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if defined(NDEBUG)
#  if NDEBUG == 0
#    undef NDEBUG
#  endif
#endif


namespace sketch {
namespace bf {
using namespace common;



template<typename ValueType>
#if HAS_AVX_512
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX512>;
#elif __AVX2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX>;
#elif __SSE2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::SSE>;
#else
using Allocator = std::allocator<ValueType, ss::Alignment::Normal>;
#endif

// TODO: add a compact, 6-bit version
// For now, I think that it's preferable for thread safety,
// considering there's an intrinsic for the atomic load/store, but there would not
// be for bit-packed versions.

static constexpr size_t optimal_nhashes(size_t l2sz, size_t est_cardinality) {
    assert(l2sz <= 64u);
    l2sz  = size_t(1) << l2sz;
    return std::ceil(std::log(2.) * l2sz / est_cardinality);
}

template<typename HashStruct=WangHash>
class bfbase_t {
// Blocked bloom filter implementation.
// To make it general, the actual point of entry is a 64-bit integer hash function.
// Therefore, you have to perform a hash function to convert various types into a suitable query.
// We could also cut our memory requirements by switching to only using 6 bits per element,
// (up to 64 leading zeros), though the gains would be relatively small
// given how memory-efficient this structure is.

// Attributes
protected:
    uint8_t                                       np_;
    uint8_t                                       nh_;
    //uint8_t                                     perh_; // Number of hashes per 64-bit hash
    const HashStruct                              hf_;
    std::vector<uint64_t, Allocator<uint64_t>>  core_;
    std::vector<uint64_t, Allocator<uint64_t>> seeds_;
    uint64_t                                seedseed_;
    uint64_t                                    mask_;
public:
    static constexpr unsigned OFFSET = 6; // log2(CHAR_BIT * 8) == log2(64) == 6
    using HashType = HashStruct;

    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this),
                              core_.size() * sizeof(core_[0]) + seeds_.size() * sizeof(seeds_[0]));
    }
    uint64_t m() const {return p() << OFFSET;}
    uint64_t p() const {return np_ + OFFSET;}
    auto nhashes() const {return nh_;}
    uint64_t mask() const {return m() - UINT64_C(1);}
    bool is_empty() const {return np_ == OFFSET;}

    // Constructor
    template<typename... Args>
    explicit bfbase_t(size_t l2sz, unsigned nhashes, uint64_t seedval, Args &&... args):
        np_(l2sz > OFFSET ? l2sz - OFFSET: 0), nh_(nhashes), hf_(std::forward<Args>(args)...), seedseed_(seedval)
    {
        //if(l2sz < OFFSET) throw std::runtime_error("Need at least a power of size 6\n");
        if(np_ > 40u) throw std::runtime_error(std::string("Attempting to make a table that's too large. p:") + std::to_string(np_));
        if(np_) resize(1ull << p());
#if !NDEBUG
        else std::fprintf(stderr, "np is small. (%u). offset %i. \n", unsigned(np_), int(l2sz) - OFFSET);
#endif
    }
    explicit bfbase_t(size_t l2sz=OFFSET): bfbase_t(l2sz, 1, std::rand()) {}
    void reseed(uint64_t seedseed=0) {
        if(seedseed == 0) seedseed = seedseed_;
        std::mt19937_64 mt(seedseed);
#if !NDEBUG
        if(__builtin_expect(p() == 0, 0)) throw std::runtime_error(std::string("p is ") + std::to_string(p()));
#endif
        auto nperhash64 = lut::nhashesper64bitword[p()];
        //assert(is_pow2(nperhash64) || !std::fprintf(stderr, "nperhash64 %u(accessed by p = %u)\n", nperhash64, unsigned(p())));
        while(seeds_.size() * nperhash64 < nh_) {
#if __cplusplus >= 201703L
            if(auto val = mt(); std::find(seeds_.cbegin(), seeds_.cend(), val) == seeds_.cend())
                seeds_.emplace_back(val);
#else
            auto val = mt();
            if(std::find(seeds_.cbegin(), seeds_.cend(), val) == seeds_.cend()) seeds_.emplace_back(val);
#endif
        }
    }

    template<typename IndType>
    INLINE void set1(IndType ind) {
        ind &= mask_;
#if !NDEBUG
        core_.at(ind >> OFFSET) |= 1ull << (ind & 63);
#else
        core_[ind >> OFFSET] |= 1ull << (ind & 63);
#endif
        assert(is_set(ind));
    }

    template<typename IndType>
    INLINE bool is_set(IndType ind) const {
        ind &= mask_;
#if !NDEBUG
        return core_.at(ind >> OFFSET) & (1ull << (ind & 63));
#else
        return core_[ind >> OFFSET] & (1ull << (ind & 63));
#endif
    }

    template<typename IndType>
    INLINE bool is_set_and_set1(IndType ind) {
        ind &= mask_;
        const auto val = (1ull << (ind & 63));
        uint64_t &ref = core_[ind >> OFFSET];
        const auto ret = ref & val;
        ref |= val;
        return ret != 0;
    }

    INLINE bool all_set(const uint64_t &hv, unsigned n, unsigned shift) const {
        if(!is_set(hv)) return false;
        for(unsigned i(1); i < n;)
            if(!is_set(hv >> (i++ * shift)))
                return false;
        return true;
    }
    INLINE bool all_set_and_set1(const uint64_t &hv, unsigned n, unsigned shift) {
        auto ret = is_set_and_set1(hv);
        for(unsigned i(1); i < n; ret &= is_set_and_set1(hv >> (i++ * shift)));
        return ret;
    }
    std::string print_vals() const {
        static const char *bin = "01";
        std::string ret;
        ret.reserve(64 * core_.size());
        for(size_t i(0); i < core_.size(); ++i) {
            uint64_t val = core_[i];
            for(unsigned k(0); k < 64; ++k)
                ret += bin[val&1], val>>=1;
        }
        return ret;
    }

    INLINE void sub_set1(const uint64_t &hv, unsigned n, unsigned shift) {
        set1(hv);
        for(unsigned subhind = 1; subhind < n; set1((hv >> (subhind++ * shift))));
    }

    uint64_t popcnt_manual() const {
        return std::accumulate(core_.cbegin() + 1, core_.cend(), popcount(core_[0]), [](auto a, auto b) {return a + popcount(b);});
    }
    uint64_t popcnt() const { // Number of set bits
        Space::VType tmp;
        const Type *op(reinterpret_cast<const Type *>(data())),
                   *ep(reinterpret_cast<const Type *>(&core_[core_.size()]));
        uint64_t sum;
        for(sum = popcnt_fn(*op++); op < ep; sum += popcnt_fn(*op++));
        return sum;
    }
    void halve() {
        if(np_ < 17) {
            for(auto it(core_.begin()), hit(it + (core_.size() >> 1)), eit = core_.end(); hit != eit; *it++ |= *hit++);
            return;
        }
        Space::VType *s = reinterpret_cast<Space::VType *>(core_.data()), *hp = reinterpret_cast<Space::VType *>(core_.data() + (core_.size() >> 1));
        const Space::VType *const end = reinterpret_cast<const Space::VType *>(&core_[core_.size()]);
        while(hp != end) {
            *s = Space::or_fn(s->simd_, hp->simd_);
            ++s, ++hp;// Consider going backwards?
        }
        core_.resize(core_.size()>>1);
        core_.shrink_to_fit();
    }
    double est_err() const {
        // Calculates estimated false positive rate as a functino of the number of set bits.
        // Does not require count of inserted elements.
        return std::pow(1.-static_cast<double>(popcnt()) / m(), nh_);
    }

    unsigned intersection_count(const bfbase_t &other) const {
        if(other.m() != m()) throw std::runtime_error("Can't compare different-sized bloom filters.");
        auto &oc = other.core_;
        Space::VType tmp;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        tmp.simd_ = Space::and_fn(Space::load(op++), Space::load(tc++));
        auto sum = popcnt_fn(tmp);

#define REPEAT_7(x) x x x x x x x
#define REPEAT_8(x) REPEAT_7(x) x
#define PERFORM_ITER tmp = Space::and_fn(Space::load(op++), Space::load(tc++)); sum += popcnt_fn(tmp);
        if(core_.size() / Space::COUNT >= 8) {
            REPEAT_7(PERFORM_ITER)
            // Handle the last 7 times after initialization
            for(size_t i(1); i < core_.size() / Space::COUNT / 8;++i) {
                REPEAT_8(PERFORM_ITER)
            }
#undef PERFORM_ITER
        }
        const Type *endp = reinterpret_cast<const Type *>(&oc[oc.size()]);
        while(op < endp) {
            tmp.simd_ = Space::and_fn(Space::load(op++), Space::load(tc++));
            sum += popcnt_fn(tmp.simd_);
        }
        return sum;
    }

    double setbit_jaccard_index(const bfbase_t &other) const {
        if(other.m() != m()) throw std::runtime_error("Can't compare different-sized bloom filters.");
        auto &oc = other.core_;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        Space::VType l1 = *op++, l2 = *tc++;
        Space::VType tmp;
        uint64_t sum1(0), sum2(0), sumu(0);

#define PERFORM_ITER \
        l1 = *op++; l2 = *tc++; \
        sum1 += popcnt_fn(l1.simd_); sum2 += popcnt_fn(l2.simd_);\
        tmp = Space::or_fn(l1.simd_, l2.simd_); \
        sumu += popcnt_fn(tmp);
        if(core_.size() / Space::COUNT >= 8) {
            REPEAT_7(PERFORM_ITER)
            // Handle the last 7 times after initialization
            for(size_t i(1); i < core_.size() / Space::COUNT / 8;++i) {
                REPEAT_8(PERFORM_ITER)
            }
        }
        for(const Type *endp = reinterpret_cast<const Type *>(&oc[oc.size()]); op < endp;) {
            PERFORM_ITER
        }
        return static_cast<double>(sum1 + sum2 - sumu) / sumu;
    }
    double jaccard_index(const bfbase_t &other) const {
        if(other.m() != m()) throw std::runtime_error("Can't compare different-sized bloom filters.");
        auto &oc = other.core_;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        Space::VType l1 = *op++, l2 = *tc++;
        Space::VType tmp;
        uint64_t sum1(0), sum2(0), sumu(0);

        if(core_.size() / Space::COUNT >= 8) {
            REPEAT_7(PERFORM_ITER)
            // Handle the last 7 times after initialization
            for(size_t i(1); i < core_.size() / Space::COUNT / 8;++i) {
                REPEAT_8(PERFORM_ITER)
            }
        }
        const Type *endp = reinterpret_cast<const Type *>(&oc[oc.size()]);
        while(op < endp) {
            PERFORM_ITER
        }
#undef PERFORM_ITER
        double set1_est = -std::log(1 - static_cast<double>(sum1) / m()) * m() / nh_;
        double set2_est = -std::log(1 - static_cast<double>(sum2) / m()) * other.m() / nh_;
        double union_est = -std::log(1 - static_cast<double>(sumu) / m()) * m() / nh_;
        double olap = set1_est + set2_est - union_est;
#if !NDEBUG
        double ji = olap / union_est;
        std::fprintf(stderr, "set est 1: %lf. set est 2: %lf. Union est: %lf. Olap est: %lf. JI est: %lf\n", set1_est, set2_est, union_est, olap, ji);
#endif
        return olap / union_est;
    }

    INLINE void addh(const uint64_t element) {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()], npersimd = Space::COUNT * npw;
        const auto shift = p();
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        while(nleft > npersimd) {
            VType v(hf_(Space::set1(element) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {sub_set1(val, npw, shift);});
            nleft -= npersimd;
        }
        const uint64_t *sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            const auto todo = std::min(npw, nleft);
            sub_set1(hf_(element ^ *sptr++), todo, shift);
            nleft -= todo;
        }
        //std::fprintf(stderr, "Finishing with element %" PRIu64 ". New popcnt: %u\n", element, popcnt());
    }

    INLINE void addh(const std::string &element) {
#ifdef ENABLE_CLHASH
        CONST_IF(std::is_same<HashStruct, clhasher>::value)
            addh(hf_(element));
        else
#endif
            addh(std::hash<std::string>{}(element)); // IE, do if not replaced.
    }
    // Reset.
    void clear() {
        if(core_.size() >= (1<<15))
            std::memset(core_.data(), 0, core_.size() * sizeof(core_[0]));
        else if(__builtin_expect(core_.size() * sizeof(core_[0]) >= sizeof(VType), 1))
            for(VType v1 = Space::set1(0), *p1(reinterpret_cast<VType *>(&core_[0])), *p2(reinterpret_cast<VType *>(&core_[core_.size()])); p1 < p2; *p1++ = v1);
        else
            std::fill(core_.begin(), core_.end(), static_cast<uint64_t>(0));
    }
    bfbase_t(bfbase_t&&) = default;
    bfbase_t(const bfbase_t &other) = default;
    bfbase_t& operator=(const bfbase_t &other) {
        // Explicitly define to make sure we don't do unnecessary reallocation.
        core_ = other.core_; np_ = other.np_; nh_ = other.nh_; seedseed_ = other.seedseed_; return *this;
    }
    bfbase_t& operator=(bfbase_t&&) = default;
    bfbase_t clone() const {return bfbase_t(np_, nh_, seedseed_);}
    bool same_params(const bfbase_t &other) const {
        return std::tie(np_, nh_, seedseed_) == std::tie(other.np_, other.nh_, other.seedseed_);
    }

    bfbase_t &operator+=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        VType *els(reinterpret_cast<VType *>(core_.data()));
        const VType *oels(reinterpret_cast<const VType *>(other.core_.data()));
        unsigned i;
        if(core_.size() / Space::COUNT >= 8) {
            for(i = 0; i < core_.size() / Space::COUNT;) {
            // Simpler version of Duff's device, except our number is always divisible by 8
            // So we can just unroll it 8 at a time.
#define OR_ITER els[i].simd_ = Space::or_fn(els[i].simd_, oels[i].simd_); ++i;
                REPEAT_8(OR_ITER)
#undef OR_ITER
            }
        } else {
            for(i = 0; i < (core_.size() / Space::COUNT); ++i)
                els[i].simd_ = Space::or_fn(els[i].simd_, oels[i].simd_);
        }
        return *this;
    }

    bfbase_t &operator&=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        unsigned i;
        VType *els(reinterpret_cast<VType *>(core_.data()));
        const VType *oels(reinterpret_cast<const VType *>(other.core_.data()));
        if(core_.size() / Space::COUNT >= 8) {
            for(i = 0; i < (core_.size() / (Space::COUNT));) {
#define AND_ITER els[i].simd_ = Space::and_fn(els[i].simd_, oels[i].simd_); ++i;
                REPEAT_8(AND_ITER)
#undef AND_ITER
            }
        } else {
            for(i = 0; i < (core_.size() / Space::COUNT); ++i)
                els[i].simd_ = Space::and_fn(els[i].simd_, oels[i].simd_);
        }
        return *this;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup(new_size);
        clear();
        core_.resize(new_size >> OFFSET);
        clear();
        np_ = std::size_t(std::log2(new_size)) - OFFSET;
        reseed();
        mask_ = new_size - 1;
        assert(np_ < 64); // To handle underflow
    }
    // Getter for is_calculated_
    bool may_contain(uint64_t val) const {
        bool ret = true;
        unsigned nleft = nh_;
        assert(p() < sizeof(lut::nhashesper64bitword));
        unsigned npw = lut::nhashesper64bitword[p()];
        unsigned npersimd = Space::COUNT * npw;
        const auto shift = p();
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        const uint64_t *sptr;
        while(nleft > npersimd) {
            VType v(hf_(Space::set1(val) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {ret &= all_set(val, npw, shift);});
            if(!ret) goto f;
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            if((ret &= all_set(hf_(val ^ *sptr++), std::min(npw, nleft), shift)) == 0) goto f;
            nleft -= std::min(npw, nleft);
            assert(sptr <= &seeds_[seeds_.size()]);
        }
        f:
        return ret;
    }
    bool may_contain_and_addh(uint64_t val) {
        bool ret = true;
        unsigned nleft = nh_;
        assert(p() < sizeof(lut::nhashesper64bitword));
        unsigned npw = lut::nhashesper64bitword[p()];
        unsigned npersimd = Space::COUNT * npw;
        const auto shift = p();
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        const uint64_t *sptr;
        while(nleft > npersimd) {
            VType v(hf_(Space::set1(val) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {ret &= all_set_and_set1(val, npw, shift);});
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            ret &= all_set_and_set1(hf_(val ^ *sptr++), std::min(npw, nleft), shift);
            nleft -= std::min(npw, nleft);
            assert(sptr <= &seeds_[seeds_.size()]);
        }
        return ret;
    }
    void may_contain(const std::vector<uint64_t> vals, std::vector<uint64_t> &ret) const {
        return may_contain(vals.data(), vals.size(), ret);
    }
    void may_contain(const uint64_t *vals, size_t nvals, std::vector<uint64_t> &ret) const {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        ret.clear();
#if !NDEBUG
        std::fprintf(stderr, "nvals: %zu. nvals. Resize size: %zu\n", nvals, nvals >> 6 + ((nvals & 0x63u) != 0));
#endif
        ret.resize(nvals >> 6 + ((nvals & 0x63u) != 0), UINT64_C(-1));
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()], npersimd = Space::COUNT * npw;
        const auto shift = p();
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        VType seed, v;
        while(nleft > npersimd) {
            seed.simd_ = (*seedptr++).simd_;
            for(unsigned  i(0); i < nvals; ++i) {
                bool is_present = true;
                v.simd_ = hf_(Space::set1(vals[i]) ^ seed.simd_);
                v.for_each([&](const uint64_t &val) {
                    ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(val, npw, shift)) << (i & 63u));
                });
            }
            nleft -= npersimd;
        }
        const uint64_t *sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            uint64_t hv, seed = *sptr++;
            for(unsigned i(0); i < nvals; ++i) {
                hv = hf_(vals[i] ^ seed);
                ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(hv, std::min(npw, nleft), shift)) << (i & 63u));
            }
            nleft -= std::min(npw, nleft);
        }
    }

    const auto &core()    const {return core_;}
    const uint64_t *data() const {return core_.data();}

    void free() {
        decltype(core_) tmp{};
        std::swap(core_, tmp);
    }
    bfbase_t operator+(const bfbase_t &other) const {
        if(!same_params(other))
            throw std::runtime_error("Different parameters.");
        bfbase_t ret(*this);
        ret += other;
        return ret;
    }
    size_t size() const {return size_t(m());}
    const auto &seeds() const {return seeds_;}
    std::string seedstring() const {
        std::string ret;
        for(size_t i(0); i < seeds_.size() - 1; ++i) ret += std::to_string(seeds_[i]), ret += ',';
        ret += std::to_string(seeds_.back());
        return ret;
    }
};

using bf_t = bfbase_t<>;

// Returns the size of the set intersection
template<typename BloomType>
inline double intersection_size(BloomType &first, BloomType &other) noexcept {
    first.csum(), other.csum();
    return intersection_size(static_cast<const BloomType &>(first), static_cast<const BloomType &>(other));
}

template<typename BloomType>
inline double jaccard_index(const BloomType &h1, const BloomType &h2) {
    return h1.jaccard_index(h2);
}
template<typename BloomType>
inline double jaccard_index(BloomType &h1, BloomType &h2) {
    return h1.jaccard_index(h2);
}

template<typename BloomType>
static inline double intersection_size(const BloomType &h1, const BloomType &h2) {
    throw std::runtime_error("NotImplementedError");
    return 0.;
}

#undef REPEAT_7
#undef REPEAT_8

} // namespace bf
} // namespace sketch

#endif // #ifndef CRUEL_BLOOM_H__
