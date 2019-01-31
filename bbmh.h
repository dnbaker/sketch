#ifndef SKETCH_BB_MINHASH_H__
#define SKETCH_BB_MINHASH_H__
#include "common.h"

namespace sketch {

namespace minhash {
struct FinalBBitMinHash;

using namespace common;
template<typename T, typename Hasher=common::WangHash>
class BBitMinHasher {
    std::vector<T> core_;
    uint16_t b_, p_;
    Hasher hf_;
public:
    using FinalType = FinalBBitMinHash;
    template<typename... Args>
    BBitMinHasher(unsigned b, unsigned p, Args &&... args): core_(1ull << p), b_(b), p_(p), hf_(std::forward<Args>(args)...) {}
    void addh(T val) {val = hf_(val);add(val);}
    void add(T hv) {
        auto &ref = core_[hv>>(sizeof(T) * CHAR_BIT - p_)];
        hv <<= p_; hv >>= p_; // Clear top values
#if NOT_THREADSAFE
        ref = std::min(ref, hv);
#else
        while(ref < hv)
            __sync_bool_compare_and_swap(std::addressof(ref), ref, hv);
#endif
    }
};

struct FinalBBitMinHash {
    std::vector<uint64_t, Allocator<uint64_t>> core_;
    uint16_t b_, p_;
    double est_cardinality_;
    template<typename Functor=DoNothing>
    FinalBBitMinHash(unsigned b, unsigned p, double est, const Functor &func=Functor()):
        core_(((b * p) + (sizeof(uint64_t) * CHAR_BIT - 1)) / (sizeof(uint64_t) * CHAR_BIT)), b_(b), p_(p), est_cardinality_(est) {
        func(*this);
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
#define VT __m512i
#elif __AVX2__
#define VT __m256i
#else
#define VT __m128i
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
#undef VT
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

} // minhash
namespace mh = minhash;
} // namespace sketch

#endif /* #ifndef SKETCH_BB_MINHASH_H__*/
