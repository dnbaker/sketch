#ifndef CRUEL_BLOOM_H__
#define CRUEL_BLOOM_H__
#ifndef NO_SLEEF
#define NO_SLEEF
#endif
#include "common.h"
#include "vec/vec.h"
#include "hash.h"


namespace sketch {
namespace bf {
using namespace hash;

#ifndef VEC_DISABLED__
using Space = vec::SIMDTypes<uint64_t>;
#endif


// TODO: add a compact, 6-bit version
// For now, I think that it's preferable for thread safety,
// considering there's an intrinsic for the atomic load/store, but there would not
// be for bit-packed versions.

static constexpr size_t optimal_nhashes(size_t l2sz, size_t est_cardinality) {
    l2sz  = size_t(1) << l2sz;
    double fpart = M_LN2 * l2sz / est_cardinality;
    size_t ipart = fpart;
    return ipart + double(ipart) != fpart; // Always round up?
}

template<typename HashStruct=WangHash>class bfbase_t;

template<typename T=::std::uint32_t, typename Alloc=Allocator<T>>
struct sparsebf_t {
    using value_type = T;
    using allocator_type = Alloc;
    std::vector<T, Alloc> core_;
    unsigned nh_, tsz_;
    sparsebf_t(std::vector<T, Alloc> &&o, size_t nhashes, size_t tablesz): core_(std::move(o)), nh_(nhashes), tsz_(tablesz) {}
    sparsebf_t(const std::vector<T, Alloc> &o, size_t nhashes, size_t tablesz): core_(o), nh_(nhashes), tsz_(tablesz) {}
    template<typename Hash>
    sparsebf_t(const bfbase_t<Hash> &dense);
    size_t popcnt() const {return core_.size();}
    auto nhashes()  const {return nh_;}
    template<typename Hash>
    double jaccard_index(const bfbase_t<Hash> &o);
};

template<typename HashStruct>
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
    HashStruct                                    hf_;
    std::vector<uint64_t, Allocator<uint64_t>>  core_;
    std::vector<uint64_t, Allocator<uint64_t>> seeds_;
    uint64_t                                seedseed_;
    uint64_t                                    mask_;
public:
    static constexpr unsigned OFFSET = 6; // log2(CHAR_BIT * 8) == log2(64) == 6
    using HashType = HashStruct;

    using final_type = bfbase_t;
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this),
                              core_.size() * sizeof(core_[0]) + seeds_.size() * sizeof(seeds_[0]));
    }
    uint64_t m() const {return core_.size() << OFFSET;}
    uint64_t p() const {return np_ + OFFSET;}
    auto nhashes() const {return nh_;}
    uint64_t mask() const {return m() - UINT64_C(1);}
    bool is_empty() const {return np_ == OFFSET;}
    double cardinality_estimate() const {
        const int ldv = -(int32_t(np_) + OFFSET);
        return std::log1p(-std::ldexp(this->popcnt(), ldv)) / ((nh_) * std::log1p(std::ldexp(-1., ldv)));
    }
    bool operator==(const bfbase_t &o) const {
        return np_ == o.np_ && nh_ == o.nh_ && seeds_ == o.seeds_ && core_ == o.core_;
    }
    bool operator!=(const bfbase_t &o) const {return !operator==(o);}

    // Constructor
    template<typename... Args>
    explicit bfbase_t(size_t l2sz, unsigned nhashes, uint64_t seedval=137, Args &&... args):
        np_(l2sz > OFFSET ? l2sz - OFFSET: 0), nh_(nhashes), hf_(std::forward<Args>(args)...), seedseed_(seedval)
    {
        //if(l2sz < OFFSET) throw std::runtime_error("Need at least a power of size 6\n");
        if(np_ > 40u) throw std::runtime_error(std::string("Attempting to make a table that's too large. p:") + std::to_string(np_));
        if(np_) resize(1ull << p());
    }
    explicit bfbase_t(size_t l2sz=OFFSET): bfbase_t(l2sz, 1, 137) {}
    explicit bfbase_t(const std::string &path) {
        read(path);
    }
    explicit bfbase_t(gzFile fp) {
        read(fp);
    }
    void reseed(uint64_t seedseed=0) {
        if(seedseed == 0) seedseed = seedseed_;
        std::mt19937_64 mt(seedseed);
#if !NDEBUG
        if(HEDLEY_UNLIKELY(p() == 0)) throw std::runtime_error(std::string("p is ") + std::to_string(p()));
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
#ifndef VEC_DISABLED__
    using VType = vec::SIMDTypes<uint64_t>::VType;
    using Type = vec::SIMDTypes<uint64_t>::Type;
#endif
    uint64_t popcnt() const { // Number of set bits
#ifndef VEC_DISABLED__
        VType tmp;
        const Type *op(reinterpret_cast<const Type *>(data())),
                   *ep(reinterpret_cast<const Type *>(&core_[core_.size()]));
        auto sum = popcnt_fn(*op++);
        while(op < ep)
            sum = Space::add(sum, popcnt_fn(*op++));
        return sum_of_u64s(sum);
#else
        return popcnt_manual();
#endif
    }
    static constexpr bool VecIsDisabled =
#ifndef VEC_DISABLED__
    true;
#else
    false;
#endif
    void halve() {
        if((np_ < 17) || VecIsDisabled) {
            for(auto it(core_.begin()), hit(it + (core_.size() >> 1)), eit = core_.end(); hit != eit; *it++ |= *hit++);
        }
#ifndef VEC_DISABLED__
        VType *s = reinterpret_cast<VType *>(core_.data()), *hp = reinterpret_cast<VType *>(core_.data() + (core_.size() >> 1));
        const VType *const end = reinterpret_cast<const VType *>(&core_[core_.size()]);
        while(hp != end) {
            *s = Space::or_fn(s->simd_, hp->simd_);
            ++s, ++hp;// Consider going backwards?
        }
#endif
        core_.resize(core_.size()>>1);
        core_.shrink_to_fit();
    }
    double est_err() const {
        // Calculates estimated false positive rate as a functino of the number of set bits.
        // Does not require count of inserted elements.
        return std::pow(1.-static_cast<double>(popcnt()) / m(), nh_);
    }

    unsigned intersection_count(const bfbase_t &other) const {
        PREC_REQ(same_params(other), "Can't compare different-sized bloom filters.");
#ifndef VEC_DISABLED__
        auto &oc = other.core_;
        VType tmp;
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
            sum = Space::add(popcnt_fn(tmp.simd_), sum);
        }
        return sum_of_u64s(sum);
#else
        return std::inner_product(core_.begin(), core_.end(), other.core_.begin(), uint64_t(0),
                                  std::plus<>{}, [](auto x, auto y) noexcept {return popcount(x & y);});
#endif
    }

    double intersection_size(const bfbase_t& other) const {
        return intersection_count(other);
    }

    double union_size(const bfbase_t &other) const {
        PREC_REQ(same_params(other), "Can't compare different-sized bloom filters.");
        auto &oc = other.core_;
#ifndef VEC_DISABLED__
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        VType l1 = *op++, l2 = *tc++;
        Space::Type sumu = popcnt_fn(Space::or_fn(l1.simd_, l2.simd_));

#define PERFORM_ITER \
        l1 = *op++; l2 = *tc++; \
        l1.simd_ = Space::or_fn(l1.simd_, l2.simd_); \
        sumu = Space::add(sumu, popcnt_fn(l1.simd_));

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
        uint64_t usumu = sum_of_u64s(sumu);
#else
        uint64_t usumu = std::inner_product(core_.begin(), core_.end(), oc.begin(), uint64_t(0),
                                            std::plus<>{}, [](auto x, auto y) noexcept {return popcount(x | y);});
#endif
        const int ldv = -(int32_t(np_) + OFFSET);
        return std::log1p(-std::ldexp(usumu, ldv)) / ((nh_) * std::log1p(std::ldexp(-1., ldv)));
    }
#ifndef VEC_DISABLED__
    double setbit_jaccard_index(const bfbase_t &other) const {
        if(other.m() != m()) throw std::runtime_error("Can't compare different-sized bloom filters.");
        auto &oc = other.core_;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        VType l1 = *op++, l2 = *tc++;
        Space::Type sum1(popcnt_fn(l1.simd_)), sum2 = popcnt_fn(l2.simd_), sumu = popcnt_fn(popcnt_fn(l1.simd_ | l2.simd_));
#undef PERFORM_ITER
#define PERFORM_ITER \
        l1 = *op++; l2 = *tc++; \
        sum1 = Space::add(sum1, popcnt_fn(l1.simd_));\
        sum2 = Space::add(sum2, popcnt_fn(l2.simd_));\
        l1.simd_ = Space::or_fn(l1.simd_, l2.simd_); \
        sumu = Space::add(sumu, popcnt_fn(l1.simd_));

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
        uint64_t usum1 = sum_of_u64s(sum1);
        uint64_t usum2 = sum_of_u64s(sum2);
        uint64_t usumu = sum_of_u64s(sumu);
        return static_cast<double>(usum1 + usum2 - usumu) / usumu;
    }
#endif
    INLINE std::array<double, 3> full_set_comparison(const bfbase_t &o) const {
        PREC_REQ(m() == o.m(), "Same size required.");
#ifndef VEC_DISABLED__
        auto &oc = o.core_;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        VType l1 = *op++, l2 = *tc++;
        Space::Type sum1, sum2, sumu;
        sum1 = popcnt_fn(l1.simd_);
        sum2 = popcnt_fn(l2.simd_);
        l1.simd_ = Space::or_fn(l1.simd_, l2.simd_);
        sumu = popcnt_fn(l1.simd_);

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
        double usum1 = sum_of_u64s(sum1);
        double usum2 = sum_of_u64s(sum2);
        double usumu = sum_of_u64s(sumu);
#else
        uint64_t usum1{0}, usum2{0}, usumu{0};
        for(size_t i = 0; i < core_.size(); ++i) {
            auto xi = core_[i];
            auto yi = o.core_[i];
            usum1 += popcount(xi);
            usum2 += popcount(yi);
            usumu += popcount(xi | yi);
        }
#endif
        const auto div = double(m()) / nh_;
        const auto nminv = -1. / m();
        double set1_est = -std::log1p(usum1 * nminv) * div;
        double set2_est = -std::log1p(usum2 * nminv) * div;
        double union_est = -std::log1p(usumu * nminv) * div;
        double olap = set1_est + set2_est - union_est;
        std::array<double, 3> ret;
        ret[2] = set1_est + set2_est - olap;
        ret[0] -= ret[2];
        ret[1] -= ret[2];
        ret[2] = std::max(ret[2], 0.);
        return ret;
    }
    double containment_index(const bfbase_t &other) const {
        auto full = full_set_comparison(other);
        return full[2] / (full[0] + full[2]);
    }
    double jaccard_index(const bfbase_t &other) const {
        if(other.m() != m()) throw std::runtime_error("Can't compare different-sized bloom filters.");
#ifndef VEC_DISABLED__
        auto &oc = other.core_;
        const Type *op(reinterpret_cast<const Type *>(oc.data())), *tc(reinterpret_cast<const Type *>(core_.data()));
        VType l1 = *op++, l2 = *tc++;
        Space::Type sum1, sum2, sumu;
        sum1 = popcnt_fn(l1.simd_);
        sum2 = popcnt_fn(l2.simd_);
        l1.simd_ = Space::or_fn(l1.simd_, l2.simd_);
        sumu = popcnt_fn(l1.simd_);

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
        double usum1 = sum_of_u64s(sum1);
        double usum2 = sum_of_u64s(sum2);
        double usumu = sum_of_u64s(sumu);
#else
        uint64_t usum1{0}, usum2{0}, usumu{0};
        for(size_t i = 0; i < core_.size(); ++i) {
            auto xi = core_[i];
            auto yi = other.core_[i];
            usum1 += popcount(xi);
            usum2 += popcount(yi);
            usumu += popcount(xi | yi);
        }
#endif
        const auto div = double(m()) / nh_;
        const auto nminv = -1. / m();
        double set1_est = -std::log1p(usum1 * nminv) * div;
        double set2_est = -std::log1p(usum2 * nminv) * div;
        double union_est = -std::log1p(usumu * nminv) * div;
        double olap = set1_est + set2_est - union_est;
        return olap / union_est;
    }

    INLINE void add(const uint64_t element) {addh(element);}
#ifdef __AVX512F__
    static constexpr size_t VSZ = 64;
#elif __AVX2__
    static constexpr size_t VSZ = 32;
#elif __SSE2__
    static constexpr size_t VSZ = 16;
#else
    static constexpr size_t VSZ = 8;
#endif
    INLINE void addh(const uint64_t element) {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()];
        const auto shift = p();
#if __AVX512F__
        const size_t npersimd = 8 * npw;
        const __m512i *sptr = (const __m512i *)&seeds_[0];
        for(;nleft > npersimd;nleft -= npersimd) {
            auto v = _mm512_set1_epi64(element) ^ *sptr++;
            sub_set1(*(const uint64_t *)&v, npw, shift);
            sub_set1(((const uint64_t *)&v)[1], npw, shift);
        }
        const uint64_t *seedptr = reinterpret_cast<const uint64_t *>(sptr);
#elif __AVX2__
        const size_t npersimd = 4 * npw;
        const __m256i *sptr = (const __m256i *)&seeds_[0];
        for(;nleft > npersimd;nleft -= npersimd) {
            auto v = _mm256_set1_epi64x(element) ^ *sptr++;
            sub_set1(*(const uint64_t *)&v, npw, shift);
            sub_set1(((const uint64_t *)&v)[1], npw, shift);
        }
        const uint64_t *seedptr = reinterpret_cast<const uint64_t *>(sptr);
#elif __SSE2__
        const size_t npersimd = 2 * npw;
        const __m128i *sptr = (const __m128i *)seeds_.data();
        while(nleft > npersimd) {
            auto v = _mm_set1_epi64x(element) ^ *sptr++;
            sub_set1(*(const uint64_t *)&v, npw, shift);
            sub_set1(((const uint64_t *)&v)[1], npw, shift);
            nleft -= npersimd;
        }
        const uint64_t *seedptr = reinterpret_cast<const uint64_t *>(sptr);
#else
        const uint64_t *seedptr = seeds_.data();
#endif
        while(nleft) {
            const auto todo = std::min(npw, nleft);
            sub_set1(hf_(element ^ *seedptr++), todo, shift);
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
    template<typename T>
    auto addh(const T &x) {
        uint64_t hv = hf_(x);
        return add(hv);
    }
    // Reset.
    void reset() {
        std::memset(core_.data(), 0, core_.size() * sizeof(core_[0]));
    }
    void clear() {reset();}
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
        return this->operator|=(other); // Interface consistency.
    }

    bfbase_t &operator&=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            snprintf(buf, 255, "For operator +=: np_ (%u) != other.np_ (%u) or nh != o.nh (%u/%u) or (seeds differ %zu/%zu)\n", np_, other.np_, nh_, other.nh_, size_t(seedseed_), size_t(other.seedseed_));
            throw std::runtime_error(buf);
        }
#ifndef VEC_DISABLED__
        unsigned i;
        VType *els(reinterpret_cast<VType *>(core_.data()));
        const VType *oels(reinterpret_cast<const VType *>(other.core_.data()));
        if(core_.size() / Space::COUNT >= 8) {
            for(i = 0; i < (core_.size() / (Space::COUNT)); i += 8) {
                els[i].simd_ = Space::and_fn(els[i].simd_, oels[i].simd_);
                els[i + 1].simd_ = Space::and_fn(els[i + 1].simd_, oels[i + 1].simd_);
                els[i + 2].simd_ = Space::and_fn(els[i + 2].simd_, oels[i + 2].simd_);
                els[i + 3].simd_ = Space::and_fn(els[i + 3].simd_, oels[i + 3].simd_);
                els[i + 4].simd_ = Space::and_fn(els[i + 4].simd_, oels[i + 4].simd_);
                els[i + 5].simd_ = Space::and_fn(els[i + 5].simd_, oels[i + 5].simd_);
                els[i + 6].simd_ = Space::and_fn(els[i + 6].simd_, oels[i + 6].simd_);
                els[i + 7].simd_ = Space::and_fn(els[i + 7].simd_, oels[i + 7].simd_);
            }
        } else
            for(i = 0; i < (core_.size() / Space::COUNT);
                els[i].simd_ = Space::and_fn(els[i].simd_, oels[i].simd_), ++i);
#else
        std::transform(core_.begin(), core_.end(), other.core_.begin(), core_.begin(), [](auto x, auto y) {return x & y;});
#endif
        return *this;
    }

    bfbase_t &operator^=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            snprintf(buf, 255, "For operator +=: np_ (%u) != other.np_ (%u) or nh != o.nh (%u/%u) or (seeds differ %zu/%zu)\n", np_, other.np_, nh_, other.nh_, size_t(seedseed_), size_t(other.seedseed_));
            throw std::runtime_error(buf);
        }
        unsigned i;
#ifndef VEC_DISABLED__
        VType *els(reinterpret_cast<VType *>(core_.data()));
        const VType *oels(reinterpret_cast<const VType *>(other.core_.data()));
        if(core_.size() / Space::COUNT >= 8) {
            for(i = 0; i < (core_.size() / (Space::COUNT));) {
#define AND_ITER els[i].simd_ = Space::xor_fn(els[i].simd_, oels[i].simd_); ++i;
                REPEAT_8(AND_ITER)
#undef AND_ITER
            }
        } else
            for(i = 0; i < (core_.size() / Space::COUNT); ++i)
                els[i].simd_ = Space::xor_fn(els[i].simd_, oels[i].simd_);
#else
        std::transform(core_.begin(), core_.end(), other.core_.begin(), core_.begin(), [](auto x, auto y) {return x ^ y;});
#endif
        return *this;
    }

    bfbase_t &operator|=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            snprintf(buf, 255, "For operator +=: np_ (%u) != other.np_ (%u) or nh != o.nh (%u/%u) or (seeds differ %zu/%zu)\n", np_, other.np_, nh_, other.nh_, size_t(seedseed_), size_t(other.seedseed_));
            throw std::runtime_error(buf);
        }
#ifndef VEC_DISABLED__
        unsigned i;
        VType *els(reinterpret_cast<VType *>(core_.data()));
        const VType *oels(reinterpret_cast<const VType *>(other.core_.data()));
        if(core_.size() / Space::COUNT >= 8) {
            for(i = 0; i + 8 <= (core_.size() / (Space::COUNT));i += 8) {
                els[i].simd_   = Space::or_fn(els[i].simd_, oels[i].simd_);
                els[i+1].simd_ = Space::or_fn(els[i+1].simd_, oels[i+1].simd_);
                els[i+2].simd_ = Space::or_fn(els[i+2].simd_, oels[i+2].simd_);
                els[i+3].simd_ = Space::or_fn(els[i+3].simd_, oels[i+3].simd_);
                els[i+4].simd_ = Space::or_fn(els[i+4].simd_, oels[i+4].simd_);
                els[i+5].simd_ = Space::or_fn(els[i+5].simd_, oels[i+5].simd_);
                els[i+6].simd_ = Space::or_fn(els[i+6].simd_, oels[i+6].simd_);
                els[i+7].simd_ = Space::or_fn(els[i+7].simd_, oels[i+7].simd_);
            }
        } else
            for(i = 0; i < (core_.size() / Space::COUNT); ++i)
                els[i].simd_ = Space::or_fn(els[i].simd_, oels[i].simd_);
#else
        std::transform(core_.begin(), core_.end(), other.core_.begin(), core_.begin(), [](auto x, auto y) {return x | y;});
#endif
        return *this;
    }
#define DEFOP(opchar)\
    bfbase_t operator opchar(const bfbase_t &other) const {\
        bfbase_t ret(*this);\
        ret opchar##= other;\
        return ret;\
    }
    DEFOP(&)
    DEFOP(|)
    DEFOP(^)
#undef DEFOP
    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup(new_size);
        clear();
        core_.resize(new_size >> OFFSET);
        clear();
        np_ = std::size_t(ilog2(new_size)) - OFFSET;
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
        const auto shift = p();
        const uint64_t *sptr = seeds_.data();
#ifndef VEC_DISABLED__
        unsigned npersimd = Space::COUNT * npw;
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        while(nleft > npersimd) {
            VType v(hf_(Space::set1(val) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {ret &= all_set(val, npw, shift);});
            if(!ret) return ret;
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
#endif
        while(nleft) {
            if((ret &= all_set(hf_(val ^ *sptr++), std::min(npw, nleft), shift)) == 0) return ret;
            nleft -= std::min(npw, nleft);
        }
        return ret;
    }
    bool may_contain_and_addh(uint64_t val) {
        bool ret = true;
        unsigned nleft = nh_;
        assert(p() < sizeof(lut::nhashesper64bitword));
        unsigned npw = lut::nhashesper64bitword[p()];
        const auto shift = p();
        const uint64_t *sptr = seeds_.data();
#ifndef VEC_DISABLED__
        unsigned npersimd = Space::COUNT * npw;
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        while(nleft > npersimd) {
            VType v(hf_(Space::set1(val) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {ret &= all_set_and_set1(val, npw, shift);});
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
#endif
        while(nleft) {
            ret &= all_set_and_set1(hf_(val ^ *sptr++), std::min(npw, nleft), shift);
            nleft -= std::min(npw, nleft);
            assert(sptr <= &seeds_[seeds_.size()]);
        }
        return ret;
    }
    auto &may_contain(const std::vector<uint64_t> vals, std::vector<uint64_t> &ret) const {
        return may_contain(vals.data(), vals.size(), ret);
    }
    auto &may_contain(const uint64_t *vals, size_t nvals, std::vector<uint64_t> &ret) const {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        ret.clear();
#if !NDEBUG
        std::fprintf(stderr, "nvals: %zu. nvals. Resize size: %zu\n", nvals, (nvals >> 6) + ((nvals & 0x63u) != 0));
#endif
        ret.resize((nvals >> 6) + ((nvals & 0x63u) != 0), UINT64_C(-1));
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()];
        const auto shift = p();
        const uint64_t *sptr = seeds_.data();
#ifndef VEC_DISABLED__
        unsigned npersimd = Space::COUNT * npw;
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        VType seed, v;
        while(nleft > npersimd) {
            seed.simd_ = (*seedptr++).simd_;
            for(unsigned  i(0); i < nvals; ++i) {
                v = hf_(Space::set1(vals[i]) ^ seed.simd_);
                v.for_each([&](const uint64_t &val) {
                    ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(val, npw, shift)) << (i & 63u));
                });
            }
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
#endif
        while(nleft) {
            uint64_t hv, seed = *sptr++;
            for(unsigned i(0); i < nvals; ++i) {
                hv = hf_(vals[i] ^ seed);
                ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(hv, std::min(npw, nleft), shift)) << (i & 63u));
            }
            nleft -= std::min(npw, nleft);
        }
        return ret;
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
    DBSKETCH_WRITE_STRING_MACROS
    DBSKETCH_READ_STRING_MACROS
    ssize_t write(gzFile fp) const {
        if(seeds_.size() > 255) {
            throw std::runtime_error("Error: serialization only allows up to 255 seeds, which means you're probably using at least 510 hash functions.");
        }
        uint8_t arr[] {np_, nh_, uint8_t(seeds_.size())};
        ssize_t rc;
        if((rc = gzwrite(fp, arr, sizeof(arr))) != sizeof(arr)) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ssize_t ret = rc;
        if((rc = gzwrite(fp, &hf_, sizeof(hf_))) != sizeof(hf_)) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ret += rc;
        if((rc = gzwrite(fp, &seedseed_, sizeof(seedseed_))) != sizeof(seedseed_)) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ret += rc;
        if((rc = gzwrite(fp, &mask_, sizeof(mask_))) != sizeof(mask_)) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ret += rc;
        if((rc = gzwrite(fp, seeds_.data(), seeds_.size() * sizeof(seeds_[0]))) != ssize_t(seeds_.size() * sizeof(seeds_[0]))) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ret += rc;
        if((rc = gzwrite(fp, core_.data(), core_.size() * sizeof(core_[0]))) != ssize_t(core_.size() * sizeof(core_[0]))) throw ZlibError(Z_ERRNO, "Failed writing to file");
        ret += rc;
        return ret;
    }
    ssize_t read(gzFile fp) {
        uint8_t arr[] {0,0,0};
        ssize_t ret = gzread(fp, arr, sizeof(arr));
        if(ret != 3) throw ZlibError("Failed to read");
        np_ = arr[0];
        nh_ = arr[1];
        seeds_.resize(arr[2]);
        ret += gzread(fp, &hf_, sizeof(hf_));
        ret += gzread(fp, &seedseed_, sizeof(seedseed_));
        ret += gzread(fp, &mask_, sizeof(mask_));
        if(np_) resize(1ull << p());
        ret += gzread(fp, core_.data(), core_.size() * sizeof(core_[0]));
        return ret;
    }
    template<typename F>
    void for_each_nonzero(const F &func) const {
        uint64_t index = 0;
        const uint64_t *it = reinterpret_cast<const uint64_t *>(&core_[0]); // Get pointer in case it's easier to optimize
        const uint64_t *const end = &*core_.end();
        do {
            auto v = *it++;
            if(v == 0) continue;
            auto nnz = popcount(v);
            auto nloops = (nnz + 7) / 8u;
            uint64_t t;
            // nloops is guaranteed to be at least one because otherwise v would have heen 0
            auto dofunc = [&]() {
                t = v & -v; func(index + ctz(v)); v ^= t;
            };
            switch(nnz % 8u) {
                case 0:
            do {        dofunc(); VEC_FALLTHROUGH
                case 7: dofunc(); VEC_FALLTHROUGH
                case 6: dofunc(); VEC_FALLTHROUGH
                case 5: dofunc(); VEC_FALLTHROUGH
                case 4: dofunc(); VEC_FALLTHROUGH
                case 3: dofunc(); VEC_FALLTHROUGH
                case 2: dofunc(); VEC_FALLTHROUGH
                case 1: dofunc();
                } while(--nloops);
                assert(v == 0);
            }
#if 0
            // Equivalent to
            while(v) {
                uint64_t t = v & -v;
                assert((t & (t - 1)) == 0);
                func(index + ctz(v));
                v ^= t;
            }
#endif
            index += sizeof(uint64_t) * CHAR_BIT;
        } while(it < end);
    }
    template<typename T=uint32_t, typename Alloc=std::allocator<T>>
    std::vector<T, Alloc> to_sparse_representation() const {
        std::vector<T, Alloc> ret;
        for_each_nonzero([&](uint64_t i) {ret.push_back(i);});
        return ret;
    }
    template<typename...Args>
    auto hash(Args &&...args) const {return hf_(std::forward<Args>(args)...);}
};

template<typename T, typename Alloc> template<typename Hash>
sparsebf_t<T, Alloc>::sparsebf_t(const bfbase_t<Hash> &dense): core_(dense.template to_sparse_representation<T, allocator_type>()), nh_(dense.nhashes()), tsz_(dense.m())
{
}

using bf_t = bfbase_t<>;

#undef REPEAT_7
#undef REPEAT_8
#undef PERFORM_ITER

} // inline namespace bf
using bf::bf_t;
using bf::sparsebf_t;
using bf::bfbase_t;
} // namespace sketch

#endif // #ifndef CRUEL_BLOOM_H__
