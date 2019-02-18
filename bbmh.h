#ifndef SKETCH_BB_MINHASH_H__
#define SKETCH_BB_MINHASH_H__
#include "div.h"
#include "common.h"
#include "hll.h"

namespace sketch {

namespace minhash {
using namespace common;

namespace detail {


// Based on content from BinDash https://github.com/zhaoxiaofei/bindash
static inline uint64_t twounivhash(uint64_t s, uint64_t t) {
    static constexpr uint64_t LARGE_PRIME = 9223372036854775783ull;
    return (UINT64_C(0x1e68e69958ce15c1) * (UINT64_C(0x84e09756b31589c9) * s + UINT64_C(0xd89576eb901ab7d3) * t) + UINT64_C(0x2f28f2976668b622)) % LARGE_PRIME;
}

template<typename T>
static constexpr T default_val() {
    return std::numeric_limits<T>::max();
}

template<typename Container>
inline int densifybin(Container &hashes) {
    using vtype = typename std::decay<decltype(hashes[0])>::type;
    const auto empty_val = default_val<vtype>();
    auto it = hashes.cbegin();
    auto min = *it, max = min;
    while(++it != hashes.end()) {
        auto v = *it;
        min = std::min(min, v);
        max = std::max(max, v);
    }
    if (empty_val != max) return 0; // Full sketch
    if (empty_val == min) return -1; // Empty sketch
    for (uint64_t i = 0; i < hashes.size(); i++) {
        uint64_t j = i, nattempts = 0;
        while(hashes[j] == empty_val)
            j = twounivhash(i, ++nattempts) % hashes.size();
        hashes[i] = hashes[j];
    }
    return 1;
}

template<typename T>
static inline double harmonic_cardinality_estimate(std::vector<T> &minvec, bool densify=true) {
    if(densify) if(detail::densifybin(minvec) < 0) return 0.;
    const double num = double(sizeof(T) * CHAR_BIT) / minvec.size();
    double sum = 0.;
    for(const auto v: minvec)
        sum += double(v) / num;
    return std::pow(minvec.size(), 2) / sum;
}

template<typename T>
static inline double harmonic_cardinality_estimate(const std::vector<T> &minvec) {
    if(std::find(minvec.begin(), minvec.end(), detail::default_val<T>()) != minvec.end()) {
        std::vector<T> tmp = minvec; // copy
        detail::densifybin(tmp);
        return harmonic_cardinality_estimate(tmp, false);
    } // Else don't worry about it, just do the thing.
    const double num = double(sizeof(T) * CHAR_BIT) / minvec.size();
    double sum = 0.;
    for(const auto v: minvec)
        sum += double(v) / num;
    return std::pow(minvec.size(), 2) / sum;
}


} // namespace detail



struct FinalBBitMinHash;
struct FinalDivBBitMinHash;

template<typename CountingType, typename=typename std::enable_if<
            std::is_arithmetic<CountingType>::value
        >::type>
struct FinalCountingBBitMinHash;



template<typename T, typename Hasher=common::WangHash>
class DivBBitMinHasher {
    std::vector<T> core_;
    uint32_t b_;
    schism::Schismatic<T> div_;
    __uint128_t M_; // Cached for fastmod64
    Hasher hf_;
public:
    using final_type = FinalDivBBitMinHash;
    template<typename... Args>
    DivBBitMinHasher(unsigned nbuckets, unsigned b, Args &&... args):
        core_(nbuckets, detail::default_val<T>()), b_(b), div_(nbuckets), hf_(std::forward<Args>(args)...)
    {
        if(b_ < 1 || b_ > 64) throw "a party";
    }
    void addh(T val) {val = hf_(val);add(val);}
    void clear() {
        std::fill(core_.begin(), core_.end(), detail::default_val<T>());
    }
    T nbuckets() const {return div_.d_;}
    INLINE void add(T hv) {
        const T bucket = div_.mod(hv);
        const T quot = div_.div(hv);
        auto &ref = core_[bucket];
#ifdef NOT_THREADSAFE
        ref = std::min(quot, ref);
#else
        while(quot < ref)
            __sync_bool_compare_and_swap(std::addressof(ref), ref, quot);
#endif
    }
    void write(const char *fn, int compression=6) const {
        finalize().write(fn, compression);
    }
    void write(const std::string &fn, int compression=6) const {write(fn.data(), compression);}
    int densify() {
        int rc = detail::densifybin(core_);
#if !NDEBUG
        switch(rc) {
            case -1: std::fprintf(stderr, "[W] Can't densify empty thing\n"); break;
            case 0: std::fprintf(stderr, "The densification, it does nothing\n"); break;
            case 1: std::fprintf(stderr, "Densifying something that needs it\n");
        }
#endif
        return rc;
    }
    double cardinality_estimate() const {
        return harmonic_cardinality_estimate(core_);
    }
    double cardinality_estimate() {
        return harmonic_cardinality_estimate(core_);
    }
    FinalDivBBitMinHash finalize(MHCardinalityMode mode=HARMONIC_MEAN) const;
};


template<typename T, typename Hasher=common::WangHash>
class BBitMinHasher {
    std::vector<T> core_;
    uint32_t b_, p_;
    Hasher hf_;
public:
    void free() {
        std::vector<T>().swap(core_);
    }
    using final_type = FinalBBitMinHash;
    template<typename... Args>
    BBitMinHasher(unsigned p, unsigned b, Args &&... args):
        core_(size_t(1) << p, detail::default_val<T>()), b_(b), p_(p), hf_(std::forward<Args>(args)...)
    {
        if(b_ + p_ > sizeof(T) * CHAR_BIT) {
            char buf[512];
            std::sprintf(buf, "[E:%s:%s:%d] Width of type (%zu) is insufficient for selected p/b parameters (%d/%d)",
                         __FILE__, __PRETTY_FUNCTION__, __LINE__, sizeof(T) * CHAR_BIT, int(b_), int(p_));
            throw std::runtime_error(buf);
        }
    }
    void read(const std::string &path) {read(path.data());}
    void read(const char *path) {
        throw NotImplementedError("NotImplemented function. This is likely an error, as you probabyl don't mean to call this.");
    }
    void addh(T val) {val = hf_(val);add(val);}
    void clear() {
        std::fill(core_.begin(), core_.end(), detail::default_val<T>());
    }
    void swap(BBitMinHasher &o) {
        std::swap_ranges((uint8_t *)this, (uint8_t *)this + sizeof(*this), (uint8_t *)std::addressof(o));
    }
    void add(T hv) {
        auto &ref = core_[hv>>(sizeof(T) * CHAR_BIT - p_)];
        hv <<= p_; hv >>= p_; // Clear top values
#ifdef NOT_THREADSAFE
        ref = std::min(ref, hv);
        assert(ref <= hv);
#else
        while(hv < ref)
            __sync_bool_compare_and_swap(std::addressof(ref), ref, hv);
#endif
    }
    void write(const char *fn, int compression=6, uint32_t b=0, MHCardinalityMode mode=HARMONIC_MEAN) const {
        finalize(b ? b: b_, mode).write(fn, compression);
    }
    void write(const std::string &fn, int compression=6, uint32_t b=0) const {write(fn.data(), compression, b);}
    void write(gzFile fp, uint32_t b=0) const {
        finalize(b?b:b_).write(fp);
    }
    int densify() {
        auto rc = detail::densifybin(core_);
#if !NDEBUG
        switch(rc) {
            case -1: std::fprintf(stderr, "[W] Can't densify empty thing\n"); break;
            case 0: std::fprintf(stderr, "The densification, it does nothing\n"); break;
            case 1: std::fprintf(stderr, "Densifying something that needs it\n");
        }
#endif
        return rc;
    }
    double cardinality_estimate(MHCardinalityMode mode=HARMONIC_MEAN) const {
        if(std::find_if(core_.begin(), core_.end(), [](auto x) {return x != detail::default_val<T>();}) == core_.end())
            return 0.; // Empty sketch
        const double num = std::ldexp(1., sizeof(T) * CHAR_BIT - p_);
        double sum;
        std::vector<T> tmp;
        const std::vector<T> *ptr = &core_;
        if(std::find(core_.begin(), core_.end(), detail::default_val<T>()) != core_.end()) { // Copy and calculate from densified.
            tmp = core_;
            detail::densifybin(tmp);
            ptr = &tmp;
        }
        switch(mode) {
        case HARMONIC_MEAN: // Dampens outliers
            return detail::harmonic_cardinality_estimate(*const_cast<std::vector<T> *>(ptr), false);
        case ARITHMETIC_MEAN: // better? Still not great.
            return std::accumulate((*ptr).begin() + 1, (*ptr).end(), num / (*ptr)[0], [num](auto x, auto y) {
                return x + num / y;
            }); // / (*ptr).size() * (*ptr).size();
        case HLL_METHOD: {
            std::array<uint64_t, 64> arr{0};
            auto diff = p_ - 1;
            for(const auto v: (*ptr))
                ++arr[v == detail::default_val<T>() ? 0: hll::clz(v) - diff];
            return hll::detail::ertl_ml_estimate(arr, p_, sizeof(T) * CHAR_BIT - p_, 0);
        }
        default: __builtin_unreachable(); // IMPOCEROUS
        }
        return sum;
    }
    FinalBBitMinHash finalize(uint32_t b=0, MHCardinalityMode mode=HARMONIC_MEAN) const;
};
template<typename T, typename Hasher=common::WangHash>
void swap(BBitMinHasher<T, Hasher> &a, BBitMinHasher<T, Hasher> &b) {
    a.swap(b);
}

template<typename T, typename CountingType, typename Hasher=common::WangHash>
class CountingBBitMinHasher: public BBitMinHasher<T, Hasher> {
    using super = BBitMinHasher<T, Hasher>;
    std::vector<CountingType> counters_;
    // TODO: consider probabilistic counting
    // TODO: consider compact_vector
    // Not threadsafe currently
    // Consider bitpacking with p_ bits for counting
public:
    using final_type = FinalCountingBBitMinHash<CountingType>;
    template<typename... Args>
    CountingBBitMinHasher(unsigned p, unsigned b, Args &&... args): super(p, b, std::forward<Args>(args)...), counters_(1ull << p) {}
    void add(T hv) {
        auto ind = hv>>(sizeof(T) * CHAR_BIT - this->p_);
        auto &ref = this->core_[ind];
        hv <<= this->p_; hv >>= this->p_; // Clear top values. We could also shift/mask, but that requires two other constants vs 1.
        if(ref < hv)
            ref = hv, counters_[ind] = 1;
        else ref += (ref == hv);
    }
    FinalCountingBBitMinHash<CountingType> finalize(uint32_t b=0, MHCardinalityMode mode=HARMONIC_MEAN) const;
};


namespace detail {

INLINE void setnthbit(uint64_t *ptr, size_t index, bool val) {
    ptr[index / 64] |= uint64_t(val) << (index % 64);
}
template<typename T> INLINE void setnthbit(T *ptr, size_t index, bool val) {
    return setnthbit(reinterpret_cast<uint64_t *>(ptr), index, val);
}

uint64_t getnthbit(const uint64_t *ptr, size_t index) {
    return (ptr[index / 64] >> (index % 64)) & 1u;
}

template<typename T> INLINE T getnthbit(const T *ptr, size_t index) {
    return T(getnthbit(reinterpret_cast<const uint8_t *>(ptr), index));
}
INLINE uint64_t getnthbit(uint64_t val, size_t index) {
    return getnthbit(&val, index);
}

#if __SSE2__
INLINE auto matching_bits(const __m128i *s1, const __m128i *s2, uint16_t b) {
     __m128i match = ~(*s1++ ^ *s2++);
     while(--b)
         match &= ~(*s1++ ^ *s2++);
     return popcount(common::vatpos(match, 0)) + popcount(common::vatpos(match, 1));
}
#else
#error("Require SSE2")
#endif

#if __AVX2__
INLINE auto matching_bits(const __m256i *s1, const __m256i *s2, uint16_t b) {
    __m256i match = ~(*s1++ ^ *s2++);
    while(--b)
        match &= ~(*s1++ ^ *s2++);
    return popcnt256(match);
}
#endif


#if HAS_AVX_512
INLINE auto matching_bits(const __m512i *s1, const __m512i *s2, uint16_t b) {
    __m512i match = ~(*s1++ ^ *s2++);
    while(--b)
        match &= ~(*s1++ ^ *s2++);
#if defined(__AVX512VPOPCNTDQ__)
    return ::_mm512_popcnt_epi64(match);
#else
    return popcnt512(match);
#endif
}
#endif

}

struct FinalBBitMinHash {
private:
    FinalBBitMinHash() {}
public:
    using value_type = uint64_t;
    double est_cardinality_;
    uint32_t b_, p_;
    std::vector<uint64_t, Allocator<uint64_t>> core_;
    FinalBBitMinHash(unsigned p, unsigned b, double est): est_cardinality_(est), b_(b), p_(p),
        core_((uint64_t(b) << p) >> 6)
    {
        std::fprintf(stderr, "Initializing finalbb with %u for b and %u for p. Number of u64s: %zu. Total nbits: %zu\n", b, p, core_.size(), core_.size() * 64);
    }
    void free() {
        decltype(core_) tmp;
        std::swap(tmp, core_);
    }
    FinalBBitMinHash(const std::string &path): FinalBBitMinHash(path.data()) {}
    FinalBBitMinHash(const char *path) {
        std::memset(this, 0, sizeof(*this));
        read(path);
    }
    FinalBBitMinHash(FinalBBitMinHash &&o) = default;
    FinalBBitMinHash(const FinalBBitMinHash &o) = default;
    template<typename T, typename Hasher=common::WangHash>
    FinalBBitMinHash(BBitMinHasher<T, Hasher> &&o): FinalBBitMinHash(std::move(o.finalize())) {
        o.free();
    }
    template<typename T, typename Hasher=common::WangHash>
    FinalBBitMinHash(const BBitMinHasher<T, Hasher> &o): FinalBBitMinHash(std::move(o.finalize())) {}
    double r() const {
        return std::ldexp(est_cardinality_, -int(sizeof(uint64_t) * CHAR_BIT - p_));
    }
    double ab() const {
        const auto _r = r();
        auto rm1 = 1. - _r;
        auto rm1p = std::pow(rm1, std::ldexp(1., b_) - 1);
        return _r * rm1p / (1. - (rm1p * rm1));
    }
    void read(gzFile fp) {
        uint32_t arr[2];
        if(gzread(fp, arr, sizeof(arr)) != sizeof(arr)) throw std::runtime_error("Could not read from file.");
        b_ = arr[0];
        p_ = arr[1];
        if(gzread(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_)) throw std::runtime_error("Could not read from file.");
        core_.resize((uint64_t(b_) << p_) >> 6);
        gzread(fp, core_.data(), sizeof(core_[0]) * core_.size());
    }
    void read(const char *path) {
        gzFile fp = gzopen(path, "rb");
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }
    void write(const std::string &path, int compression=6) const {write(path.data(), compression);}
    void write(const char *path, int compression=6) const {
        std::string mode = compression ? std::string("wb") + std::to_string(compression): std::string("wT");
        gzFile fp = gzopen(path, mode.data());
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        write(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint32_t arr[] {b_, p_};
        if(__builtin_expect(gzwrite(fp, arr, sizeof(arr)) != sizeof(arr), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, core_.data(), core_.size() * sizeof(core_[0])) != ssize_t(core_.size() * sizeof(core_[0])), 0)) throw std::runtime_error("Could not write to file");
    }
#if HAS_AVX_512
    template<typename Func1, typename Func2>
    uint64_t equal_bblocks_sub(const uint64_t *p1, const uint64_t *pe, const uint64_t *p2, const Func1 &f1, const Func2 &f2) const {
        using VT = __m512i;
        if(core_.size() * sizeof(core_[0]) < sizeof(__m512i)) {
            uint64_t sum = f2(*p1++, *p2++);
            while(p1 != pe)
                sum += f2(*p1++, *p2++);
            return sum;
        }
        const VT *vp1 = reinterpret_cast<const VT *>(p1);
        const VT *vpe = reinterpret_cast<const VT *>(pe);
        const VT *vp2 = reinterpret_cast<const VT *>(p2);
        auto sum = f1(*vp1++, *vp2++);
        while(vp1 != vpe) sum = _mm512_add_epi64(f1(*vp1++, *vp2++), sum);
        return sum_of_u64s(sum);
    }
#elif __AVX2__
    template<typename Func1, typename Func2>
    uint64_t equal_bblocks_sub(const uint64_t *p1, const uint64_t *pe, const uint64_t *p2, const Func1 &f1, const Func2 &f2) const {
        using VT = __m256i;
        if(core_.size() * sizeof(core_[0]) < sizeof(__m256i)) {
            uint64_t sum = f2(*p1++, *p2++);
            while(p1 != pe)
                sum += f2(*p1++, *p2++);
            return sum;
        }
        const VT *vp1 = reinterpret_cast<const VT *>(p1);
        const VT *vpe = reinterpret_cast<const VT *>(pe);
        const VT *vp2 = reinterpret_cast<const VT *>(p2);
        auto sum = f1(*vp1++, *vp2++);
        while(vp1 != vpe) sum = _mm256_add_epi64(f1(*vp1++, *vp2++), sum);
        return sum_of_u64s(sum);
    }
#else
    template<typename Func1, typename Func2>
    uint64_t equal_bblocks_sub(const uint64_t *p1, const uint64_t *pe, const uint64_t *p2, const Func1 &f1, const Func2 &f2) const {
        using VT = __m128i;
        uint64_t sum = 0;
        if(core_.size() * sizeof(core_[0]) >= sizeof(Space::VType)) {
            const VT *vp1 = reinterpret_cast<const VT *>(p1);
            const VT *vpe = reinterpret_cast<const VT *>(pe);
            const VT *vp2 = reinterpret_cast<const VT *>(p2);
            do {sum += f1(*vp1++, *vp2++);} while(vp1 != vpe);
            p1 = reinterpret_cast<const uint64_t *>(vp1), p2 = reinterpret_cast<const uint64_t *>(vp2);
        }
        while(p1 != pe)
            sum += f2(*p1++, *p2++);
        return sum;
    }
#endif
    uint64_t equal_bblocks(const FinalBBitMinHash &o) const {
        assert(o.core_.size() == core_.size());
        const uint64_t *p1 = core_.data(), *pe = core_.data() + core_.size(), *p2 = o.core_.data();
        assert(b_ <= 64); // b_ > 64 not yet supported, though it could be done with a larger hash
        // p_ already guaranteed to be greater than 6
        switch(p_) {
            case 6: {
                auto match = ~(*p1++ ^ *p2++);
                while(p1 != pe) match &= ~(*p1++ ^ *p2++);
                return popcount(match);
            }
            case 7: {
                const __m128i *vp1 = reinterpret_cast<const __m128i *>(p1), *vp2 = reinterpret_cast<const __m128i *>(p2), *vpe = reinterpret_cast<const __m128i *>(pe);
                __m128i match = ~(*vp1++ ^ *vp2++);
                while(vp1 != vpe)
                    match &= ~(*vp1++ ^ *vp2++);
                return popcount(common::vatpos(match, 0)) + popcount(common::vatpos(match, 1));
            }
#if __AVX2__
            case 8: return common::sum_of_u64s(detail::matching_bits(reinterpret_cast<const __m256i *>(p1), reinterpret_cast<const __m256i *>(p2), b_));
#  if HAS_AVX_512
            case 9: return common::sum_of_u64s(detail::matching_bits(reinterpret_cast<const __m512i *>(p1), reinterpret_cast<const __m512i *>(p2), b_));
            default: {
                // Process each 'b' remainder block in
                const __m512i *vp1 = reinterpret_cast<const __m512i *>(p1), *vp2 = reinterpret_cast<const __m512i *>(p2), *vpe = reinterpret_cast<const __m512i *>(pe);
                auto sum = detail::matching_bits(vp1, vp2, b_);
                for(size_t i = 1; i < (size_t(1) << (p_ - 9_)); ++i) {
                    vp1 += b_;
                    vp2 += b_;
                    sum = _mm512_add_epi64(detail::matching_bits(vp1, vp2, b_), sum);
                }
                assert((uint64_t *)vp1 == &core_[core_.size()]);
                return common::sum_of_u64s(sum);
            }
#    else /* has avx2 not not 512 */
            default: {
                const __m256i *vp1 = reinterpret_cast<const __m256i *>(p1), *vp2 = reinterpret_cast<const __m256i *>(p2);
#if !NDEBUG
                const __m256i *vpe = reinterpret_cast<const __m256i *>(pe);
#endif
                auto sum = detail::matching_bits(vp1, vp2, b_);
                for(size_t i = 1; i < 1ull << (p_ - 8u); ++i) {
                    vp1 += b_;
                    vp2 += b_;
                    sum = _mm256_add_epi64(detail::matching_bits(vp1, vp2, b_), sum);
                    assert(vp1 <= vpe);
                }
#if !NDEBUG
                auto fptr = (uint64_t *)(reinterpret_cast<const __m256i *>(p1) + (size_t(b_) << (p_ - 8u)));
                auto eptr = p1 + core_.size();
                assert(fptr == (p1 + core_.size()) || !std::fprintf(stderr, "fptr: %p. optr: %p\n", fptr, p1 + core_.size()));
#endif
                return common::sum_of_u64s(sum);
            }
#  endif /* avx512 or avx2 */
#else /* assume SSE2 */
            default: {
                // Process each 'b' remainder block in
                const __m128i *vp1 = reinterpret_cast<const __m128i *>(p1), *vp2 = reinterpret_cast<const __m128i *>(p2), *vpe = reinterpret_cast<const __m128i *>(pe);
                __m128i match = ~(*vp1++ ^ *vp2++);
                for(unsigned b = b_; --b;match &= ~(*vp1++ ^ *vp2++));
                auto sum = popcount(*(const uint64_t *)&match) + popcount(((const uint64_t *)&match)[1]);
                while(vp1 != vpe) {
                    match = ~(*vp1++ ^ *vp2++);
                    for(unsigned b = b_; --b; match &= ~(*vp1++ ^ *vp2++));
                    sum += popcount(*(const uint64_t *)&match) + popcount(((const uint64_t *)&match)[1]);
                }
                return sum;
            }
#endif
        }
    }
    double frac_equal(const FinalBBitMinHash &o) const {
        return std::ldexp(equal_bblocks(o), -int(p_));
    }
    uint64_t nmin() const {
        return uint64_t(1) << p_;
    }
    double jaccard_index(const FinalBBitMinHash &o) const {
        /*
         * reference: https://arxiv.org/abs/1802.03914.
        */
        const double b2pow = std::ldexp(1., -b_);
        double frac = frac_equal(o);
        frac -= b2pow;
        return std::max(0., frac / (1. - b2pow));
    }
    double containment_index(const FinalBBitMinHash &o) const {
        double ji = jaccard_index(o);
        double is = (est_cardinality_ + o.est_cardinality_) * ji / (1. + ji);
        return is / est_cardinality_;
    }
};

INLINE double jaccard_index(const FinalBBitMinHash &a, const FinalBBitMinHash &b) {
    return a.jaccard_index(b);
}

struct FinalDivBBitMinHash {
private:
    FinalDivBBitMinHash() {}
public:
    using value_type = uint64_t;
    double est_cardinality_;
    uint64_t nbuckets_;
    uint32_t b_;
    std::vector<uint64_t, Allocator<uint64_t>> core_;
    FinalDivBBitMinHash(unsigned nbuckets, unsigned b, double est): est_cardinality_(est), b_(b), nbuckets_(nbuckets),
        core_((uint64_t(b) * nbuckets_) / 64 + (nbuckets_ * uint64_t(b) % 64 != 0))
    {
        std::fprintf(stderr, "Initializing finalbb with %u for b and %u for p. Number of u64s: %zu. Total nbits: %zu\n", b, nbuckets, core_.size(), core_.size() * 64);
    }
    void free() {
        decltype(core_) tmp;
        std::swap(tmp, core_);
    }
    FinalDivBBitMinHash(const std::string &path): FinalDivBBitMinHash(path.data()) {}
    FinalDivBBitMinHash(const char *path) {
        std::memset(this, 0, sizeof(*this));
        read(path);
    }
    FinalDivBBitMinHash(FinalDivBBitMinHash &&o) = default;
    FinalDivBBitMinHash(const FinalDivBBitMinHash &o) = default;
    template<typename T, typename Hasher=common::WangHash>
    FinalDivBBitMinHash(DivBBitMinHasher<T, Hasher> &&o): FinalDivBBitMinHash(std::move(o.finalize())) {
        std::fprintf(stderr, "est card %lf\n", est_cardinality_);
        std::fprintf(stderr, "b: %u. nbuckets: %u. size of core: %zu\n", b_, nbuckets_, core_.size());
        o.free();
    }
    template<typename T, typename Hasher=common::WangHash>
    FinalDivBBitMinHash(const DivBBitMinHasher<T, Hasher> &o): FinalDivBBitMinHash(std::move(o.finalize())) {}
    void read(gzFile fp) {
        uint64_t arr[2];
        if(gzread(fp, arr, sizeof(arr)) != sizeof(arr)) throw std::runtime_error("Could not read from file.");
        b_ = arr[0];
        nbuckets_ = arr[1];
        if(gzread(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_)) throw std::runtime_error("Could not read from file.");
        core_.resize(b_ * nbuckets_ / 64 + (b_ * nbuckets_ % 64 != 0));
        gzread(fp, core_.data(), sizeof(core_[0]) * core_.size());
    }
    void read(const char *path) {
        gzFile fp = gzopen(path, "rb");
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }
    void write(const std::string &path, int compression=6) const {write(path.data(), compression);}
    void write(const char *path, int compression=6) const {
        std::string mode = compression ? std::string("wb") + std::to_string(compression): std::string("wT");
        gzFile fp = gzopen(path, mode.data());
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        write(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint64_t arr[] {b_, nbuckets_};
        if(__builtin_expect(gzwrite(fp, arr, sizeof(arr)) != sizeof(arr), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, core_.data(), core_.size() * sizeof(core_[0])) != ssize_t(core_.size() * sizeof(core_[0])), 0)) throw std::runtime_error("Could not write to file");
    }
    double equal_bblocks(const FinalDivBBitMinHash &o) const {
        throw NotImplementedError("FinalDivBBitMinHash not yet completed.");
    }
    double frac_equal(const FinalDivBBitMinHash &o) const {
        return double(equal_bblocks(o)) / nbuckets_;
    }
    uint64_t nmin() const {
        return nbuckets_;
    }
    double jaccard_index(const FinalDivBBitMinHash &o) const {
        /*
         * reference: https://arxiv.org/abs/1802.03914.
        */
        const double b2pow = std::ldexp(1., -b_);
        double frac = frac_equal(o);
        frac -= b2pow;
        return std::max(0., frac / (1. - b2pow));
    }
    double containment_index(const FinalDivBBitMinHash &o) const {
        double ji = jaccard_index(o);
        double is = (est_cardinality_ + o.est_cardinality_) * ji / (1. + ji);
        return is / est_cardinality_;
    }
};

template<typename T, typename Hasher>
FinalDivBBitMinHash DivBBitMinHasher<T, Hasher>::finalize(MHCardinalityMode mode) const {
#if 0
    std::vector<T> core_;
    uint32_t b_, p_;
    schism::Schismatic<T> div_;
    __uint128_t M_; // Cached for fastmod64
    Hasher hf_;
#endif
    FinalDivBBitMinHash ret(nbuckets(), b_, cardinality_estimate());
    throw NotImplementedError("BitPacking for DivBBitMinHasher not yet complete.");
}

template<typename T, typename Hasher>
FinalBBitMinHash BBitMinHasher<T, Hasher>::finalize(uint32_t b, MHCardinalityMode mode) const {
    b = b ? b: b_; // Use the b_ of BBitMinHasher if not specified; this is because we can make multiple kinds of bbit minhashes from the same hasher.
    std::vector<T> tmp;
    const std::vector<T> *ptr = &core_;
    if(std::find(core_.begin(), core_.end(), detail::default_val<T>()) != core_.end()) {
        tmp = core_;
        detail::densifybin(tmp);
        ptr = &core_;
    }
    double cest = cardinality_estimate(mode);
    const std::vector<T> &core_ref = *ptr;
    using detail::getnthbit;
    using detail::setnthbit;
    FinalBBitMinHash ret(p_, b, cest);
    std::fprintf(stderr, "size of ret vector: %zu. b_: %u, p_: %u. cest: %lf\n", ret.core_.size(), b_, p_, cest);
    using FinalType = typename FinalBBitMinHash::value_type;
    // TODO: consider supporting non-power of 2 numbers of minimizers by subsetting to the first k <= (1<<p) minimizers.
    if(b_ == 64) {
        // We've already failed for the case of b_ + p_ being greater than the width of T
        std::memcpy(ret.core_.data(), core_ref.data(), sizeof(core_ref[0]) * core_ref.size());
    } else {
        if(__builtin_expect(p_ < 6, 0))
            throw std::runtime_error("BBit minhashing requires at least p = 6 for non-power of two b currently. We could reduce this requirement using 32-bit integers.");
        // #if AVX512: if p_ >= 9
        // Pack an AVX512 element for 1 << (p_ - 9)
        // else
        // #endif
        // #if AVX2: if p_ >= 8
        // Pack an AVX512 element for 1 << (p_ - 8)
        // #else if p_ >= 7 // Assume SSE2
        // Pack an SSE2 element for each 1 << (p_ - 7)
        switch(p_) {
        case 6:
                for(size_t _b = 0; _b < b_; ++_b)
                    for(size_t i = 0; i < core_ref.size(); ++i)
                        ret.core_.operator[](i / (sizeof(T) * CHAR_BIT) * b_ + _b) |= (core_ref[i] & (FinalType(1) << _b)) << (i % (sizeof(FinalType) * CHAR_BIT));
#if !NDEBUG
                for(size_t i = 0; i < core_ref.size(); ++i) {
                    for(size_t b = 0; b < b_; ++b) {
                        assert(getnthbit(ret.core_.data() + b, i) == getnthbit(core_ref[i], b));
                    }
                }
#endif
            break;
#define DEFAULT_SET_CASE(num, type) \
        default:\
            for(size_t ov = 0, ev = 1 << (p_ - num); ov != ev; ++ov) {\
                auto main_ptr = ret.core_.data() + ov * sizeof(type) / sizeof(uint64_t) * b_;\
                auto core_ptr = core_ref.data() + ov * (sizeof(type) * CHAR_BIT);\
                for(size_t b = 0; b < b_; ++b) {\
                    auto ptr = main_ptr + (b * sizeof(type)/sizeof(uint64_t));\
                    for(size_t i = 0; i < (sizeof(type) * CHAR_BIT); ++i)\
                        setnthbit(ptr, i, getnthbit(core_ptr[i], b));\
                }\
            }\
            break
#define SET_CASE(num, type) \
        case num:\
            for(size_t b = 0; b < b_; ++b) {\
                auto ptr = ret.core_.data() + (b * sizeof(type)/sizeof(uint64_t));\
                assert(core_ref.size() == (sizeof(type) * CHAR_BIT));\
                for(size_t i = 0; i < (sizeof(type) * CHAR_BIT); ++i)\
                    setnthbit(ptr, i, getnthbit(core_ref[i], b));\
            }\
            break
        SET_CASE(7, __m128i);
#if __AVX2__
        SET_CASE(8, __m256i);
#if HAS_AVX_512
        SET_CASE(9, __m512i);

        DEFAULT_SET_CASE(9u, __m512i);
#else
        DEFAULT_SET_CASE(8u, __m256i);
#endif
#else /* no avx2 or 512 */
        DEFAULT_SET_CASE(7u, __m128i);
#endif
#undef DEFAULT_SET_CASE
#undef SET_CASE
        }
    }
    return ret;
}
template<typename CountingType, typename>
struct FinalCountingBBitMinHash: public FinalBBitMinHash {
    std::vector<CountingType> counters_;
    FinalCountingBBitMinHash(FinalBBitMinHash &&tmp, const std::vector<CountingType> &counts): FinalBBitMinHash(std::move(tmp)), counters_(counts) {}
    FinalCountingBBitMinHash(unsigned p, unsigned b, double est): FinalBBitMinHash(b, p, est), counters_(size_t(1) << this->p_) {}
    void write(gzFile fp) const {
        FinalBBitMinHash::write(fp);
        gzwrite(fp, counters_.data(), counters_.size() * sizeof(counters_[0]));
    }
    void read(gzFile fp) {
        FinalBBitMinHash::read(fp);
        counters_.resize(size_t(1) << this->p_);
        gzread(fp, counters_.data(), counters_.size() * sizeof(counters_[0]));
    }
    void write(const char *path, int compression=6) const {
        std::string mode = compression ? std::string("wb") + std::to_string(compression): std::string("wT");
        gzFile fp = gzopen(path, mode.data());
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        write(fp);
        gzclose(fp);
    }
    void read(const char *path) {
        gzFile fp = gzopen(path, "rb");
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }

    template<typename=std::enable_if_t<std::is_same<CountingType, uint32_t>::value>> // Only finished for uint32_t currently
    std::pair<uint64_t, uint64_t> histogram_sums(const FinalCountingBBitMinHash &o) const {
        assert(o.core_.size() == core_.size());
        const uint64_t *p1 = core_.data(), *pe = core_.data() + core_.size(), *p2 = o.core_.data();
        assert(b_ <= 64); // b_ > 64 not yet supported, though it could be done with a larger hash
        size_t offset = 0;
        uint64_t matched_sum = 0;
        __m128i total_sum = _mm_set1_epi32(0);
        for(size_t i = 0; i < 1u << (p_ - 6); ++i) {
            auto match = ~(*p1++ ^ *p2++);
            while(p1 != pe) match &= ~(*p1++ ^ *p2++);
            for(size_t i = 0; i < 32; ++i) {
                auto maskv = match & 0x3u;
#ifdef ENABLE_COMPUTED_GOTO
                // Can this be masked with ~(v-1) somehow?
                const void **labels = {&&zero, &&one, &&two, &&three};
                goto *labels[maskv];
                zero: goto end;
                one: matched_sum += std::min(o.counters_[i*2], counters_[i*2]); goto end;
                two: matched_sum += std::min(o.counters_[i*2 + 1], counters_[i*2 + 1]); goto end;
                three: matched_sum += std::min(o.counters_[i*2 + 1], counters_[i*2 + 1]); matched_sum += std::min(o.counters_[i*2], counters_[i*2]);
                end: ;
#else
                switch(maskv) {
                    case 0: break;
                    case 1: matched_sum += std::min(o.counters_[i*2], counters_[i*2]); break;
                    case 2: matched_sum += std::min(o.counters_[i*2 + 1], counters_[i*2 + 1]); break;
                    case 3: matched_sum += std::min(o.counters_[i*2 + 1], counters_[i*2 + 1]); matched_sum += std::min(o.counters_[i*2], counters_[i*2]); break;
                }
#endif
                total_sum = _mm_add_epi32(total_sum, __mm_max_epi32(o.counters_.data()[i*2], counters_.data()[i * 2]));
            }
        }

        const auto p = (const uint32_t *)total_sum;
        return {matched_sum, uint64_t(p[0]) + p[1] + p[2] + p[3]};
    }
};

template<typename T, typename CountingType, typename Hasher>
FinalCountingBBitMinHash<CountingType> CountingBBitMinHasher<T, CountingType, Hasher>::finalize(uint32_t b, MHCardinalityMode mode) const {
    auto bbm = BBitMinHasher<T, Hasher>::finalize(b, mode);
    return FinalCountingBBitMinHash<CountingType>(std::move(bbm), this->counters_);
}

} // minhash
namespace mh = minhash;
} // namespace sketch

#endif /* #ifndef SKETCH_BB_MINHASH_H__*/
