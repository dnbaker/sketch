#ifndef SKETCH_BB_MINHASH_H__
#define SKETCH_BB_MINHASH_H__
#include "common.h"
#include "hll.h"

namespace sketch {

namespace minhash {
using namespace common;

namespace detail {
// Based on content from BinDash https://github.com/zhaoxiaofei/bindash

static inline uint64_t univhash2(uint64_t s, uint64_t t) {
    uint64_t x = (1009) * s + (1000*1000+3) * t;
    return (48271 * x + 11) % ((1ULL << 31) - 1);
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
    for (uint64_t i = 0; i < hashes.size(); i++) {
        uint64_t j = i;
        uint64_t nattempts = 0;
        while (UINT64_MAX == hashes[j])
            j = univhash2(i, nattempts++) % hashes.size();
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
            for(const auto v: core_) {
                if(v == default_val())
                    ++arr[0];
                else {
                    ++arr[hll::clz(v << p_) + 1];
                }
            }
            std::string s;
            for(const auto v: arr)
                s += std::to_string(v) + ',';
            s.back() = '\n';
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
public:
    using final_type = FinalBBitMinHash;
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


struct FinalBBitMinHash {
    using value_type = uint64_t;
    std::vector<uint64_t, Allocator<uint64_t>> core_;
    uint16_t b_, p_;
    double est_cardinality_;
    template<typename Functor=DoNothing>
    FinalBBitMinHash(unsigned b, unsigned p, double est):
        core_(uint64_t(b) << p), b_(b), p_(p), est_cardinality_(est)
    {
        std::fprintf(stderr, "Initializing finalbb with %u for b and %u for p\n", b, p);
        assert((core_.size() >> p_) == b_);
    }
    int densify() {
        std::fprintf(stderr, "[W:%s:%d] densification not implemented. Results will not be made ULTRADENSE\n", __PRETTY_FUNCTION__, __LINE__);
        return 0; // Success, I guess?
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
    template<typename Func1, typename Func2>
    uint64_t equal_bblocks_sub(const uint64_t *p1, const uint64_t *pe, const uint64_t *p2, const Func1 &f1, const Func2 &f2) const {
#if __AVX512BW__
        using VT = __m512i;
#elif __AVX2__
        using VT = __m256i;
#else
        using VT = __m128i;
#endif
        uint64_t sum = 0;
        if(core_.size() * sizeof(core_[0]) >= sizeof(Space::VType)) {
            const VT *vp1 = reinterpret_cast<const VT *>(p1);
            const VT *vpe = reinterpret_cast<const VT *>(pe);
            const VT *vp2 = reinterpret_cast<const VT *>(p2);
            do {sum += f1(*vp1++, *vp2++);} while(vp1 != vpe);
            p1 = reinterpret_cast<const uint64_t *>(vp1), p2 = reinterpret_cast<const uint64_t *>(vp2);
#if __AVX512BW__
#else
            switch(b_) {
                case 8:  sum >>= 3; break; // Divide by 8 because each matching subblock's values are 8 "1"-bits,
                case 16: sum >>= 4; break; // which makes us overcount by a factor of the number of bits per operand.
                case 32: sum >>= 5; break;
                case 64: sum >>= 6; break;
            }
#endif
        }
        while(p1 != pe)
            sum += f2(*p1++, *pe++);
        return 0; // This is a lie.
    }
    uint64_t equal_bblocks(const FinalBBitMinHash &o) const {
        assert(o.core_.size() == core_.size());
        const uint64_t *p1 = core_.data(), *pe = core_.data() + core_.size(), *p2 = o.core_.data();
        switch(b_) {
            case 1: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {return popcnt_fn(~Space::xor_fn(x, y));}, [](auto x, auto y) {return popcount(~(x ^ y));});
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
            }, [](auto x, auto y) {
                return popcount(_mm_cmpeq_pi8(MMX_CVT(x), MMX_CVT(y))) >> 3;
            });
            case 16: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu16_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi16(x, y));
#else
                return popcnt_fn(_mm_cmpeq_epi16(x, y));
#endif
            }, [](auto x, auto y) {
                return popcount(_mm_cmpeq_pi16(MMX_CVT(x), MMX_CVT(y))) >> 4;
            });
            case 32: return equal_bblocks_sub(p1, pe, p2, [](auto x, auto y) {
#if __AVX512BW__
                return popcount(_mm512_cmpeq_epu32_mask(x, y));
#elif __AVX2__
                return popcnt_fn(_mm256_cmpeq_epi32(x, y));
#else
                return popcnt_fn(_mm_cmpeq_epi32(x, y));
#endif
            }, [](auto x, auto y) {
                return popcount(_mm_cmpeq_pi32(MMX_CVT(x), MMX_CVT(y))) >> 5;
            });
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
#undef MMX_CVT
        assert(b_ <= 64); // b_ > 64 not yet supported, though it could be done
        // Not special case
        throw NotImplementedError("comparisons for laterally packed minimizers not yet complete.");
        uint64_t sum = 0;
        for(size_t i = 0; i < size_t(1) << p_; ++i) {
            sum += (bbit_at_ind(i) == o.bbit_at_ind(i));
        }
        return sum;
    }
    uint64_t bbit_at_ind(size_t ind) const {
        throw NotImplementedError("Haven't implemented bbit_at_ind for extractig value for b bits at index");
        assert(b_ * ind / 64 < core_.size());
        const uint64_t *d = core_.data();
    }
    double frac_equal(const FinalBBitMinHash &o) const {
        return std::ldexp(equal_bblocks(o), -int(p_));
    }
    double jaccard_index(const FinalBBitMinHash &o) const {
        auto a1b = this->ab(), a2b = o.ab();
        auto r1 = r(), r2 = o.r(), rsuminv =1./ (r1 + r2);
        auto c1b = a1b * r2 / rsuminv, c2b = a2b * r1 * rsuminv;
        auto fe = frac_equal(o);
        return (fe - c1b) / (1. - c2b);
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
    FinalBBitMinHash ret(b_, p_, cest);
    std::fprintf(stderr, "size of ret vector: %zu. b_: %u, p_: %u. cest: %lf\n", ret.core_.size(), b_, p_);
    using FinalType = typename FinalBBitMinHash::value_type;
    switch(b_) {
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
        case 64:
            // We've already failed for the case of b_ + p_ being greater than the width of T
            std::memcpy(ret.core_.data(), core_.data(), sizeof(core_[0]) * core_.size());
            break;
        default: {
            if(__builtin_expect(p_ < 6, 0))
                throw std::runtime_error("BBit minhashing requires at least p = 6 for non-power of two b currently.");
            for(size_t _b = 0; _b < b_; ++_b)
                for(size_t i = 0; i < core_.size(); ++i)
                    ret.core_.__access__(i / (sizeof(T) * CHAR_BIT) * b_ + _b) |= (core_[i] & (FinalType(1) << _b)) << (i % (sizeof(FinalType) * CHAR_BIT));
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
