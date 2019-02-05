#ifndef SKETCH_BB_MINHASH_H__
#define SKETCH_BB_MINHASH_H__
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

template<typename Container>
inline int densifybin(Container &hashes, unsigned p) {
    using vtype = typename std::decay<decltype(hashes[0])>::type;
    auto it = hashes.cbegin();
    auto min = *it, max = min;
    while(++it != hashes.end()) {
        auto v = *it;
        min = std::min(min, v);
        max = std::max(max, v);
    }
    if ((vtype(1) << p) != max) return 0; // Full sketch
    if ((vtype(1) << p) == min) return -1; // Empty sketch
    const uint64_t empty_val = 1ull << (64 - p);
    for (uint64_t i = 0; i < hashes.size(); i++) {
        uint64_t j = i, nattempts = 0;
        while(hashes[j] == empty_val)
            j = twounivhash(i, ++nattempts) % hashes.size();
        hashes[i] = hashes[j];
    }
    return 1;
}

} // namespace detail
struct FinalBBitMinHash;
template<typename CountingType, typename=typename std::enable_if<
            std::is_arithmetic<CountingType>::value
        >::type>
struct FinalCountingBBitMinHash;

template<typename T, typename Hasher=common::WangHash>
class BBitMinHasher {
    std::vector<T> core_;
    uint16_t b_, p_;
    Hasher hf_;
public:
    using final_type = FinalBBitMinHash;
    template<typename... Args>
    BBitMinHasher(unsigned b, unsigned p, Args &&... args):
        core_(size_t(1) << p, T(1) << (sizeof(T) * CHAR_BIT - p)), b_(b), p_(p), hf_(std::forward<Args>(args)...)
    {
        if(b_ + p_ > sizeof(T) * CHAR_BIT) {
            char buf[512];
            std::sprintf(buf, "[E:%s:%s:%d] Width of type (%zu) is insufficient for selected p/b parameters (%d/%d)",
                         __FILE__, __PRETTY_FUNCTION__, __LINE__, sizeof(T) * CHAR_BIT, int(b_), int(p_));
            throw std::runtime_error(buf);
        }
    }
    void addh(T val) {val = hf_(val);add(val);}
    auto default_val() const {
        return T(1) << (sizeof(T) * CHAR_BIT - p_);
    }
    void add(T hv) {
        auto &ref = core_[hv>>(sizeof(T) * CHAR_BIT - p_)];
        hv <<= p_; hv >>= p_; // Clear top values
#if 1
        ref = std::min(ref, hv);
        assert(ref <= hv);
#else
        std::fprintf(stderr, "hv: %zu vs current %zu\n", size_t(hv), size_t(ref));
        while(ref < hv)
            __sync_bool_compare_and_swap(std::addressof(ref), ref, hv);
        std::fprintf(stderr, "after hv: %zu vs current %zu\n", size_t(hv), size_t(ref));
#endif
    }
    void densify() {
        auto rc = detail::densifybin(core_, p_);
#if !NDEBUG
        switch(rc) {
            case -1: std::fprintf(stderr, "[W] Can't densify empty thing\n"); break;
            case 0: std::fprintf(stderr, "The densification, it does nothing\n"); break;
            case 1: std::fprintf(stderr, "Densifying something that needs it\n");
        }
#endif
    }
    double cardinality_estimate(MHCardinalityMode mode=HARMONIC_MEAN) const {
#if 0
        for(size_t i = 0; i < core_.size(); ++i)
            std::fprintf(stderr, "value at index %zu is %zu\n", i, size_t(core_[i]));
#endif
        const double num = std::ldexp(1., sizeof(T) * CHAR_BIT - p_);
        double sum;
        switch(mode) {
        case HARMONIC_MEAN: // Dampens outliers
            sum = 0.;
            for(const auto v: core_) {
                assert(num >= v || !std::fprintf(stderr, "%lf vs %zu failure\n", num, v));
                sum += double(v) / num;
            }
            sum = std::ldexp(1. / sum, p_ * 2);
            break;
        case ARITHMETIC_MEAN:
            sum = std::accumulate(core_.begin() + 1, core_.end(), num / core_[0], [num](auto x, auto y) {
                return x + num / y;
            }); // / core_.size() * core_.size();
            break;
        case MEDIAN: {
            T *tmp = static_cast<T *>(std::malloc(sizeof(T) * core_.size()));
            if(__builtin_expect(!tmp, 0)) throw std::bad_alloc();
            std::memcpy(tmp, core_.data(), core_.size() * sizeof(T));
            common::sort::default_sort(tmp, tmp + core_.size());
            sum = 0.5 * ((num / tmp[core_.size() >> 1]) + (num / tmp[(core_.size() >> 1) - 1])) * core_.size();
            std::free(tmp);
            break;
        }
        case HLL_METHOD: {
            std::array<uint64_t, 64> arr{0};
            auto diff = p_ - 1;
            for(const auto v: core_)
                ++arr[v == default_val() ? 0: hll::clz(v) - diff];
#if !NDEBUG
            std::string s;
            for(const auto v: arr)
                s += std::to_string(v) + ',';
            s.back() = '\n';
            std::fputs(s.data(), stderr);
#endif
            sum = hll::detail::ertl_ml_estimate(arr, p_, sizeof(T) * CHAR_BIT - p_, 0);
            break;
        }
        default: __builtin_unreachable(); // IMPOCEROUS
        }
        return sum;
    }
    FinalBBitMinHash finalize(MHCardinalityMode mode=HARMONIC_MEAN) const;
};

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
    CountingBBitMinHasher(unsigned b, unsigned p, Args &&... args): super(b, p, std::forward<Args>(args)...), counters_(1ull << p) {}
    void add(T hv) {
        auto ind = hv>>(sizeof(T) * CHAR_BIT - this->p_);
        auto &ref = this->core_[ind];
        hv <<= this->p_; hv >>= this->p_; // Clear top values. We could also shift/mask, but that requires two other constants vs 1.
        if(ref < hv)
            ref = hv, counters_[ind] = 1;
        else ref += (ref == hv);
    }
    template<typename CType=CountingType>
    FinalCountingBBitMinHash<CType> finalize(MHCardinalityMode mode=HARMONIC_MEAN);
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
    return popcnt512(match);
}
#endif

}

struct FinalBBitMinHash {
    using value_type = uint64_t;
    std::vector<uint64_t, Allocator<uint64_t>> core_;
    uint16_t b_, p_;
    double est_cardinality_;
    template<typename Functor=DoNothing>
    FinalBBitMinHash(unsigned b, unsigned p, double est):
        core_((uint64_t(b) << p) >> 6, 1ull << p), b_(b), p_(p), est_cardinality_(est)
    {
        std::fprintf(stderr, "Initializing finalbb with %u for b and %u for p. Number of u64s: %zu. Total nbits: %zu\n", b, p, core_.size(), core_.size() * 64);
    }
    void densify() {
        auto rc = detail::densifybin(core_, p_);
#if !NDEBUG
        switch(rc) {
            case -1: std::fprintf(stderr, "[W] Can't densify empty thing\n"); break;
            case 0: std::fprintf(stderr, "The densification, it does nothing\n"); break;
            case 1: std::fprintf(stderr, "Densifying something that needs it\n");
        }
#endif
    }
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
        uint16_t arr[2];
        if(gzread(fp, arr, sizeof(arr)) != sizeof(arr)) throw std::runtime_error("Could not read from file.");
        b_ = arr[0];
        p_ = arr[1];
        if(gzread(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_)) throw std::runtime_error("Could not read from file.");
        core_.resize((uint64_t(b_) << p_) >> 6, 1ull << p_);
        gzread(fp, core_.data(), sizeof(core_[0]) * core_.size());
    }
    void read(const char *path) {
        gzFile fp = gzopen(path, "rb");
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }
    void write(const char *path, int compression=6) const {
        std::string mode = compression ? std::string("wb") + std::to_string(compression): std::string("wT");
        gzFile fp = gzopen(path, mode.data());
        if(!fp) throw std::runtime_error(std::string("Could not open file at ") + path);
        write(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        uint16_t arr[] {b_, p_};
        if(__builtin_expect(gzwrite(fp, arr, sizeof(arr)) != sizeof(arr), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, &est_cardinality_, sizeof(est_cardinality_)) != sizeof(est_cardinality_), 0)) throw std::runtime_error("Could not write to file");
        if(__builtin_expect(gzwrite(fp, core_.data(), core_.size() * sizeof(core_[0])) != core_.size() * sizeof(core_[0]), 0)) throw std::runtime_error("Could not write to file");
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
#if 0
        if(b_ == 1) {
#if HAS_AVX_512 || __AVX2__
            Space::VType *vp1 = (Space::VType *)p1;
            Space::VType *vp2 = (Space::VType *)p2;
            Space::VType *vpe = (Space::VType *)pe;
            auto sum = popcnt_fn(~Space::xor_fn(*vp1++, *vp2++));
            while(vp1 != vpe)
                sum = Space::add_fn(sum, popcnt_fn(~Space::xor_fn(*vp1++, *vp2++)));
            return common::sum_of_u64s(sum);
#else
            return std::accumulate(p1 + 1, pe, popcount(~(*p1 ^ *p2++)), [&](uint64_t sum, auto x) {return sum + popcount(~(x ^ *p2++));});
#endif
        }
            case 2: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
                // Based on nucleotide counting in bowtie2
                x ^= y;
                auto x0 = x ^ Space::set1(UINT64_C(0xffffffffffffffff));
                auto x1 = Space::srli(x, 1);
                auto x2 = Space::and_fn(Space::set1(UINT64_C(0x5555555555555555)), x1);
                auto x3 = Space::and_fn(x0, x2);
                return popcnt_fn(x3);
            }, [](auto x, auto y) {
                x ^= y;
                uint64_t x0 = x ^ UINT64_C(0xffffffffffffffff);
                uint64_t x1 = x0 >> 1;
                uint64_t x2 = x1 & UINT64_C(0x5555555555555555);
                uint64_t x3 = x0 & x2;
                return popcount(x3);
            });
            case 4: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
                x ^= y;
                auto x0 = x ^ Space::set1(UINT64_C(0xffffffffffffffff));
                auto x1 = Space::srli(x, 1);
                auto x2 = Space::and_fn(Space::set1(UINT64_C(0x5555555555555555)), x1);
                auto x3 = Space::and_fn(x0, x2);
                auto x4 = x3 & Space::set1(UINT64_C(0x4444444444444444));
                auto x5 = Space::srli(Space::and_fn(x3, Space::set1(UINT64_C(0x1111111111111111))), 2);
                auto x6 = Space::and_fn(x4, x5);
                return popcnt_fn(x6);
            }, [](auto x, auto y) {
                x ^= y;
                uint64_t x0 = x ^ UINT64_C(0xffffffffffffffff);
                uint64_t x1 = x0 >> 1;
                uint64_t x2 = x1 & UINT64_C(0x5555555555555555);
                uint64_t x3 = x0 & x2;
                uint64_t x4 = x3 & UINT64_C(0x4444444444444444);
                uint64_t x5 = (x3 & UINT64_C(0x1111111111111111)) >> 2;
                uint64_t x6 = x4 & x5;
                return popcount(x6);
            });
#define MMX_CVT(x) (*reinterpret_cast<const __m64 *>(std::addressof(x)))
            case 8: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu8_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi8(x, y));
#else
                return popcnt_fn(_mm_cmpeq_epi8(x, y));
#endif
            }, [](auto x, auto y) {return popcount(_mm_cmpeq_pi8(MMX_CVT(x), MMX_CVT(y))) >> 3;});
            case 16: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu16_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi16(x, y));
#else
                return popcnt_fn(_mm_cmpeq_epi16(x, y));
#endif
            }, [](auto x, auto y) {return popcount(_mm_cmpeq_pi16(MMX_CVT(x), MMX_CVT(y))) >> 4;});
            case 32: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu32_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi32(x, y));
#else
                return popcnt_fn(_mm_cmpeq_epi32(x, y));
#endif
            }, [](auto x, auto y) {return popcount(_mm_cmpeq_pi32(MMX_CVT(x), MMX_CVT(y))) >> 5;});
            case 64: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu64_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi64(x, y));
#elif __SSE4_1__
                return popcnt_fn(_mm_cmpeq_epi64(x, y));
#else
                auto tmp = _mm_cmpeq_epi32(x, y);
                auto tmp2 = _mm_slri_epi64(tmp);
                auto tmp3 = _mm_cmpeq_epi32(tmp, tmp2);
                return popcount(tmp3);
#endif
            }, std::equal_to<uint64_t>());
        }
#endif
#undef MMX_CVT
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
                for(size_t i = 1; i < (size_t(1) << (p_ - 9)); ++i) {
                    vp1 += b_;
                    vp2 += b_;
                    sum = _mm512_add_epi64(detail::matching_bits(vp1, vp2, b_), sum);
                }
                assert((uint64_t *)vp1 == &core_[core_.size()]);
                return common::sum_of_u64s(sum);
            }
#    else /* has avx2 not not 512 */
            default: {
                const __m256i *vp1 = reinterpret_cast<const __m256i *>(p1), *vp2 = reinterpret_cast<const __m256i *>(p2), *vpe = reinterpret_cast<const __m256i *>(pe);
                auto sum = detail::matching_bits(vp1, vp2, b_);
                for(size_t i = 1; i < 1ull << (p_ - 8u); ++i) {
#if !NDEBUG
                    std::fprintf(stderr, "Now incrementing matching bits for next block with i = %zu of %zu with p = %d: %zu\n", i, size_t(1) << (p_ - 8u), p_, detail::matching_bits(reinterpret_cast<const __m256i *>(p1) + i * b_, reinterpret_cast<const __m256i *>(p2) + i * b_, b_));
#endif
                    vp1 += b_;
                    vp2 += b_;
                    sum = _mm256_add_epi64(detail::matching_bits(vp1, vp2, b_), sum);
                }
#if !NDEBUG
                auto fptr = (uint64_t *)(reinterpret_cast<const __m256i *>(p1) + (size_t(b_) << (p_ - 8u)));
                auto eptr = p1 + core_.size();
                std::fprintf(stderr, "eptr - fptr: %zd. fptr - start: %zd. total distance: %zd\n", eptr - fptr, fptr - p1, &core_[core_.size()] - &core_[0]);
                assert(fptr == (p1 + core_.size()) || !std::fprintf(stderr, "fptr: %p. optr: %p\n", fptr, p1 + core_.size()));
#endif
                return common::sum_of_u64s(sum);
            }
#  endif /* avx512 or avx2 */
#else /* assume SSE2 */
            default: {
                // Process each 'b' remainder block in
                __m128i *vp1 = reinterpret_cast<__m128i *>(p1), *vp2 = reinterpret_cast<__m128i *>(p2), *vpe = reinterpret_cast<__m128i *>(pe);
                __m128i match = ~(*vp1++ ^ *vp2++);
                for(unsigned b = b_; --b;match &= ~(*vp1++ ^ *vp2++));
                auto sum = popcount(*(uint64_t *)&match) + popcount(((uint64_t *)&match)[1]);
                while(vp1 != vpe) {
                    match = ~(*vp1++ ^ *vp2++);
                    for(unsigned b = b_; --b; match &= ~(*vp1++ ^ *vp2++));
                    sum += popcount(*(uint64_t *)&match) + popcount(((uint64_t *)&match)[1]);
                }
                return sum;
            }
#endif
        }
    }
#undef common::vatpos
    double frac_equal(const FinalBBitMinHash &o) const {
        return std::ldexp(equal_bblocks(o), -int(p_));
    }
    uint64_t nmin() const {
        return uint64_t(1) << p_;
    }
    double jaccard_index(const FinalBBitMinHash &o) const {
#if 0
        auto a1b = this->ab(), a2b = o.ab();
        auto r1 = r(), r2 = o.r(), rsuminv = 1./ (r1 + r2);
        auto c1b = a1b * r2 / rsuminv, c2b = a2b * r1 * rsuminv;
        auto fe = frac_equal(o);
        return (fe - c1b) / (1. - c2b);
#endif
        auto eq = equal_bblocks(o);
        auto expected = 1ull << p_;
        expected >>= b_;
#if !NDEBUG
        std::fprintf(stderr, "expected: %zu. eq: %zu. nmin: %zu. Diff: %zu. est ji: %lf\n", size_t(expected), size_t(eq), size_t(nmin()), size_t(eq - expected), double(eq - expected) / nmin());
#endif
        return double(eq - expected) / nmin();
    }
};

template<typename T, typename Hasher>
FinalBBitMinHash BBitMinHasher<T, Hasher>::finalize(MHCardinalityMode mode) const {
#if !NDEBUG
#define __access__(x) at(x)
#else
#define __access__(x) operator[](x)
#endif
    double cest = cardinality_estimate(mode);
    using detail::getnthbit;
    using detail::setnthbit;
    FinalBBitMinHash ret(b_, p_, cest);
    std::fprintf(stderr, "size of ret vector: %zu. b_: %u, p_: %u. cest: %lf\n", ret.core_.size(), b_, p_, cest);
    using FinalType = typename FinalBBitMinHash::value_type;
    // TODO: consider supporting non-power of 2 numbers of minimizers by subsetting to the first k <= (1<<p) minimizers.
    switch(b_) {
#if 0
#define SWITCH_CASE(b) \
        case b: \
            for(size_t i = 0; i < core_.size(); ++i) {\
                ret.core_.__access__(i * b / 64) |= (core_[i] & ((UINT64_C(1) << b) - 1)) << ((i * b) % 64); \
            }\
            break;
        SWITCH_CASE(1)
        SWITCH_CASE(2)
        SWITCH_CASE(4)
        SWITCH_CASE(8)
        SWITCH_CASE(16)
        SWITCH_CASE(32)
#endif
        case 64:
            // We've already failed for the case of b_ + p_ being greater than the width of T
            std::memcpy(ret.core_.data(), core_.data(), sizeof(core_[0]) * core_.size());
            break;
        default: {
            if(__builtin_expect(p_ < 6, 0))
                throw std::runtime_error("BBit minhashing requires at least p = 6 for non-power of two b currently. We could reduce this requirement using 32-bit integers.");
            // TODO:
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
                    for(size_t _b = 0; _b < b_; ++_b) {
                        for(size_t i = 0; i < core_.size(); ++i) {
                            ret.core_.__access__(i / (sizeof(T) * CHAR_BIT) * b_ + _b) |= (core_[i] & (FinalType(1) << _b)) << (i % (sizeof(FinalType) * CHAR_BIT));
                        }
                    }
#if !NDEBUG
                    for(size_t i = 0; i < core_.size(); ++i) {
                        for(size_t b = 0; b < b_; ++b) {
                            assert(getnthbit(ret.core_.data() + b, i) == getnthbit(core_[i], b));
                        }
                    }
#endif
                break;
            case 7: {
                for(size_t b = 0; b < b_; ++b) {
                    auto ptr = ret.core_.data() + (b * 2);
                    assert(core_.size() == 128);
                    for(size_t i = 0; i < 128; ++i)
                        setnthbit(ptr, i, getnthbit(core_[i], b));
                }
                break;
            }
#if __AVX2__
            case 8:
                for(size_t b = 0; b < b_; ++b) {
                    auto ptr = ret.core_.data() + (b * 4);
                    assert(core_.size() == 256);
                    for(size_t i = 0; i < 256; ++i)
                        setnthbit(ptr, i, getnthbit(core_[i], b));
                }
#if !NDEBUG
                for(size_t i = 0; i < core_.size(); ++i) {
                    for(size_t j = 0; j < b_; ++j) {
                        uint64_t retsubb = j * 4;
                        uint64_t which_bit = i % 256;
                        assert(getnthbit(ret.core_.data() + retsubb, which_bit) == getnthbit(core_[i], j));
                    }
                }
#endif
                break;
#if HAS_AVX_512
            case 9:
                for(size_t b = 0; b < b_; ++b) {
                    auto ptr = ret.core_.data() + (b * 8);
                    assert(core_.size() == 512);
                    for(size_t i = 0; i < 512; ++i)
                        setnthbit(ptr, i, getnthbit(core_[i], b));
                }
                break;
            default:
                for(size_t ov = 0, ev = 1 << (p_ - 9); ov != ev; ++ov) {
                    auto main_ptr = ret.core_.data() + ov * sizeof(__m512i) / sizeof(uint64_t) * b_;
                    auto core_ptr = core_.data() + ov * 512;
                    for(size_t b = 0; b < b_; ++b) {
                        auto ptr = main_ptr + (b * 4);
                        for(size_t i = 0; i < 512; ++i)
                            setnthbit(ptr, i, getnthbit(core_ptr[i], b));
                    }
                }
                break;
#else
            default:
                for(size_t ov = 0, ev = 1 << (p_ - 8); ov != ev; ++ov) {
                    auto main_ptr = ret.core_.data() + ov * sizeof(__m256i) / sizeof(uint64_t) * b_;
                    auto core_ptr = core_.data() + ov * 256;
                    for(size_t b = 0; b < b_; ++b) {
                        auto ptr = main_ptr + (b * 4);
                        for(size_t i = 0; i < 256; ++i)
                            setnthbit(ptr, i, getnthbit(core_ptr[i], b));
                    }
                }
                break;
#endif
#else /* no avx2 or 512 */
            default:
                for(size_t ov = 0, ev = 1 << (p_ - 7); ov != ev; ++ov) {
                    auto main_ptr = ret.core_.data() + ov * sizeof(__m128i) / sizeof(uint64_t) * b_;
                    auto core_ptr = core_.data() + ov * 128;
                    for(size_t b = 0; b < b_; ++b) {
                        auto ptr = main_ptr + (b * 4);
                        for(size_t i = 0; i < 128; ++i)
                            setnthbit(ptr, i, getnthbit(core_ptr[i], b));
                    }
                }
#endif
            }
            break;
        }
    }
#undef __access__
    return ret;
}
template<typename CountingType, typename>
struct FinalCountingBBitMinHash: public FinalBBitMinHash {
};

} // minhash
namespace mh = minhash;
} // namespace sketch

#endif /* #ifndef SKETCH_BB_MINHASH_H__*/
