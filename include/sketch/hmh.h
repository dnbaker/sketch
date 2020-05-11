#ifndef SKETCH_HMH2_H__
#define SKETCH_HMH2_H__
#include "sketch/hll.h"
#include <iostream>

namespace sketch {
namespace hmh {

static constexpr inline bool legal_regsize(unsigned regsize) {
    switch(regsize) {
        case 8: case 16: case 32: case 64: return true;
        default: return false;
    }
}


static const std::array<double, 64> INVPOWERSOFTWO = {
1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05, 3.0517578125e-05, 1.52587890625e-05, 7.62939453125e-06, 3.814697265625e-06, 1.9073486328125e-06, 9.5367431640625e-07, 4.76837158203125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 2.9802322387695312e-08, 1.4901161193847656e-08, 7.450580596923828e-09, 3.725290298461914e-09, 1.862645149230957e-09, 9.313225746154785e-10, 4.656612873077393e-10, 2.3283064365386963e-10, 1.1641532182693481e-10, 5.820766091346741e-11, 2.9103830456733704e-11, 1.4551915228366852e-11, 7.275957614183426e-12, 3.637978807091713e-12, 1.8189894035458565e-12, 9.094947017729282e-13, 4.547473508864641e-13, 2.2737367544323206e-13, 1.1368683772161603e-13, 5.684341886080802e-14, 2.842170943040401e-14, 1.4210854715202004e-14, 7.105427357601002e-15, 3.552713678800501e-15, 1.7763568394002505e-15, 8.881784197001252e-16, 4.440892098500626e-16, 2.220446049250313e-16, 1.1102230246251565e-16, 5.551115123125783e-17, 2.7755575615628914e-17, 1.3877787807814457e-17, 6.938893903907228e-18, 3.469446951953614e-18, 1.734723475976807e-18, 8.673617379884035e-19, 4.336808689942018e-19, 2.168404344971009e-19, 1.0842021724855044e-19
};

struct hmh_t {
protected:
    uint64_t rbm_;
    uint16_t p_, r_, lrszm3_;
    std::vector<uint8_t, Allocator<uint8_t>> data_;
    double alpha_;
public:
    hmh_t(unsigned p, unsigned rsize=8):
            rbm_((uint64_t(1) << (rsize - q)) - 1),
            p_(p), r_(rsize-q),
            lrszm3_(ilog2(rsize) - 3), alpha_(make_alpha(uint64_t(1) << p))
    {
        switch(rsize) {
            case 8: lrszm3_ = 0; break;  case 16: lrszm3_ = 1; break;
            case 32: lrszm3_ = 2; break; case 64: lrszm3_ = 3; break;
            default: PREC_REQ(legal_regsize(rsize), "Must have 8, 16, 32, or 64 for register size");
        }
        PREC_REQ(rsize == 8 || rsize == 16 || rsize == 32  || rsize == 64, "rsize must be 8, 16, 32, 64");
        PREC_REQ(p_ >= 3 && p < 64, "p can't be less than 3 or >= 64");
        data_.resize(rsize << (p_ - 3));
        assert(integral::is_pow2(rbm_ + 1));
        assert(std::all_of(data_.begin(), data_.end(), [](auto x) {return x == 0;}));
        if(lrszm3_ == 2) {
            std::fprintf(stderr, "Note: computation works as expected for 8, 16, and 64 bytes, but there may be an issue with 32\n");
        }
    }

    // Constants, encoding, and decoding utilities
    static constexpr unsigned q  = 6;
    static constexpr unsigned tq = 1ull << 6;
    static constexpr double C4 = 0.6796779486389564; // C * 4


    unsigned regsize() const {return r_ + q;}
    size_t num_registers() const {return size_t(1) << p_;}
    uint64_t tr() const {return rbm_ + 1;}
    unsigned max_lremainder() const {
        return 64 - p_;
    }
    uint64_t max_remainder() const {
        return (uint64_t(1) << r_) - 1;
    }
    template<typename IT>
    hmh_t &perform_merge(const hmh_t &o) {
        using Space = vec::SIMDTypes<IT>;
        using Type = typename Space::Type;
        auto d = reinterpret_cast<Type *>(data_.data());
        auto e = reinterpret_cast<const Type *>(&data_[data_.size()]);
        auto od = reinterpret_cast<const Type *>(o.data_.data());
        do {
            Space::store(d, Space::max(Space::load(d), Space::load(od)));
            ++d, ++od;
        } while(d < e);
        return *this;
    }
    hmh_t &operator+=(const hmh_t &o) {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        switch(lrszm3_) {
            case 0: return perform_merge<uint8_t> (o);
            case 1: return perform_merge<uint64_t>(o);
            case 2: return perform_merge<uint32_t>(o);
            case 3: return perform_merge<uint64_t>(o);
            default: __builtin_unreachable();
        }
    }


    template<typename IT1, typename IT2, typename IT3, typename IT4=std::common_type_t<IT1, IT2, IT3>>
    static inline constexpr IT4 encode_register(IT1 r, IT2 lzc, IT3 rem) {
        IT4 enc = (IT4(lzc) << r) | rem;
        assert(reg2lzc(enc, r) == lzc || !std::fprintf(stderr, "lzc of %ld should be decoded as %ld from %ld", static_cast<long int>(reg2lzc(enc, r)), static_cast<long int>(lzc), static_cast<long int>(enc)));
        assert((enc % (1ull << r)) == rem);
        return enc;
    }
    template<typename IT>
    static inline constexpr std::pair<IT, IT> decode_register(IT value, IT r) {
        return {value >> r, value & ((IT(1) << r) - 1)};
    }
    template<typename IT>
    static inline constexpr std::pair<IT, IT> decode_register(IT value, IT r, IT bm) {  // cache bitmask
        return {value >> r, value & bm}; // same as reg2lzc, reg2rem
    }
    template<typename IT, typename IT2>
    static inline constexpr IT reg2lzc(IT reg, IT2 r) {
        return reg >> r;
    }
    template<typename IT, typename IT2>
    static inline constexpr IT reg2rem(IT reg, IT2 bm) {
        return reg & bm;
    }
    static inline double compute_beta(double v) {
        // More stack, but better pipelining, hopefully
        const double lv = std::log1p(v);
        const double lvp = lv * lv;
        const double lvp3 = lvp * lv;
        const double lvp4 = lvp * lvp;
        const double lvp5 = lvp * lvp3;
        const double lvp6 = lvp3 * lvp3;
        const double lvp7 = lvp3 * lvp4;
        // More operations, but fewer dependencies for a superscalar processor
        return -0.370393911 * v + 0.070471823 * lv + 0.17393686 * lvp
                + 0.16339839 * lvp3 - 0.09237745 * lvp4 + 0.03738027 * lvp5
                - 0.005384159 * lvp6 + 0.00042419 * lvp7;
    }
    template<typename Func>
    void for_each_union_lzrem(const hmh_t &o, const Func &func) const {
        for_each_union_register(o, [&func,rbm=rbm_,r=r_](auto x) {
            auto lzc = reg2lzc(x, r);
            auto rem = reg2rem(x, rbm);
            func(lzc, rem);
        });
    }
    template<typename Func>
    void for_each_lzrem(const Func &func) const {
        for_each_register([&func,rbm=rbm_,r=r_](auto x) {
            auto lzc = reg2lzc(x, r);
            auto rem = reg2rem(x, rbm);
            func(lzc, rem);
        });
    }
    template<typename IT, typename Func>
    void __for_each_union_register(const hmh_t &o, const Func &func) const {
        using Space = vec::SIMDTypes<IT>;
        using Type  = typename Space::Type;
        using VType = typename Space::VType;

        const Type *d = reinterpret_cast<const Type *>(data_.data());
        const Type *e = d + ((num_registers() / Space::COUNT) * Space::COUNT);
        const Type *od = reinterpret_cast<const Type *>(o.data_.data());
        while(d < e)
            VType(Space::max(Space::load(d++), Space::load(od++))).for_each(func);
        for(const IT *w = (const IT *)d, *e = (const IT *)&data_[data_.size()]; w < e; func(*w++));
    }
    template<typename Func>
    void for_each_union_register(const hmh_t &o, const Func &func) const {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        switch(lrszm3_) {
            case 0: __for_each_union_register<uint8_t>(o, func); break;
            case 1: __for_each_union_register<uint16_t>(o, func); break;
            case 2: __for_each_union_register<uint32_t>(o, func); break;
            case 3: __for_each_union_register<uint64_t>(o, func); break;
            default: __builtin_unreachable();
        }
    }
    template<typename Func>
    void for_each_register(const Func &func) const {
        const uint8_t *const s = data_.data(), *const e = &s[data_.size()];
        switch(lrszm3_) {
            case 0:
                std::for_each(s, e, func); break;
            case 1:
                std::for_each(reinterpret_cast<const uint16_t *>(s), reinterpret_cast<const uint16_t *>(e), func);
                break;
            case 2:
                std::for_each(reinterpret_cast<const uint32_t *>(s), reinterpret_cast<const uint32_t *>(e), func);
                break;
            case 3:
                std::for_each(reinterpret_cast<const uint64_t *>(s), reinterpret_cast<const uint64_t *>(e), func);
                break;
            default: __builtin_unreachable();
        }
    }
    template<typename Func>
    void for_each_register_pair(const hmh_t &o, const Func &func) const {
        const uint8_t *const s = data_.data(), *const e = &s[data_.size()];
        auto fe = [&](auto startp, auto endp, auto op) {
            do {
                func(*startp++, *op++);
            } while(startp != endp);
        };
        switch(lrszm3_) {
            case 0:
                fe(s, e, o.data_.data()); break;
            case 1:
                fe(reinterpret_cast<const uint16_t *>(s), reinterpret_cast<const uint16_t *>(e),
                   reinterpret_cast<const uint16_t *>(o.data_.data())); break;
                break;
            case 2:
                fe(reinterpret_cast<const uint32_t *>(s), reinterpret_cast<const uint32_t *>(e),
                   reinterpret_cast<const uint32_t *>(o.data_.data())); break;
                break;
            case 3:
                fe(reinterpret_cast<const uint64_t *>(s), reinterpret_cast<const uint64_t *>(e),
                   reinterpret_cast<const uint64_t *>(o.data_.data())); break;
            default: __builtin_unreachable();
        }
    }
    template<typename Func>
    void enumerate_each_register(const Func &func) const {
        size_t i = 0;
        for_each_register([&i,&func](auto x) {func(i++, x);});
    }
    template<typename Func>
    void enumerate_each_register_pair(const hmh_t &o, const Func &func) const {
        size_t i = 0;
        for_each_register_pair([&i,&func](auto x, auto y) {func(i++, x, y);});
    }
    void add(uint64_t h1, uint64_t h2) {
        switch(lrszm3_) {
            case 0: __add<uint8_t> (h1, h2); break;
            case 1: __add<uint16_t>(h1, h2); break;
            case 2: __add<uint16_t>(h1, h2); break;
            case 3: __add<uint64_t>(h1, h2); break;
            default: __builtin_unreachable();
        }
    }
    std::array<uint32_t, 64> sum_counts() const {
        using hll::detail::SIMDHolder;
        // TODO: this
        // Note: we have whip out more complicated vectorized maxes for
        // widths of 16, 32, or 64
        std::array<uint32_t, 64> ret{0};
        if(data_.size() >= sizeof(SIMDHolder)) {
            switch(lrszm3_) {
#define CASE_U(cse, func, msk) \
                case cse: {\
                    const Space::Type mask = Space::set1(UINT64_C(msk));\
                    for(const SIMDHolder *ptr = reinterpret_cast<const SIMDHolder *>(data_.data()), *eptr = reinterpret_cast<const SIMDHolder *>(data_.data() + data_.size());\
                        ptr != eptr; ++ptr) {\
                        auto tmp = *ptr;\
                        tmp = Space::and_fn(Space::srli(*reinterpret_cast<VType *>(&tmp), r_), mask);\
                        tmp.func(ret);\
                    }\
                    break;\
                }
                CASE_U(0, inc_counts,   0x3f3f3f3f3f3f3f3f)
                CASE_U(1, inc_counts16, 0x003f003f003f003f)
                CASE_U(2, inc_counts32, 0x0000003f0000003f)
                CASE_U(3, inc_counts64, 0x000000000000003f)
#undef CASE_U
                default: goto manual_core;
            }
        } else {
            manual_core:
            for(const auto i: data_) ++ret[reg2lzc(i, r_)];
        }
        if(lrszm3_ == 2)
            for(unsigned i = 0; i < 64; ++i)
                if(ret[i]) std::fprintf(stderr, "%u: %u\n", i, int(ret[i]));
        return ret;
    }
    template<typename IT>
    void __add(uint64_t h1, uint64_t h2) {
        uint64_t bucket = h1 >> max_lremainder();
        unsigned lzc = clz(((h1 << 1)|1) << (p_ - 1)) + 1;
        uint64_t sig = h2 & rbm_;
        const IT reg = encode_register(r_, lzc, sig);
        IT &r = *reinterpret_cast<IT *>(data_.data() + sizeof(IT) * bucket);
#ifndef NOT_THREADSAFE
        if(reg > r) r = reg;
#else
        while(reg > r) __sync_bool_compare_and_swap(&r, r, reg);
#endif
        assert(r >= reg);
    }
    double estimate_hll_portion() const {
        return hll::detail::ertl_ml_estimate(this->sum_counts(), p_, 64 - p_);
    }
    double cardinality_estimate() const {
        double hest = estimate_hll_portion();
        if(hest < (1024 << p_)) return hest;
        return estimate_mh_portion();
    }
    double estimate_mh_portion() const {
        double ret = 0.;
        double maxrem = max_remainder(), mri = 1. / maxrem;
        for_each_lzrem([&](auto lzc, auto rem) {
            ret += (1. + (maxrem - rem) * mri) * INVPOWERSOFTWO[lzc];
        });
        return std::ldexp(1. / ret, int(2 * p_));
    }
    double union_size(const hmh_t &o) const {
        double ret = 0.;
        double maxrem = max_remainder(), mri = 1. / maxrem;
        for_each_union_lzrem(o, [&](auto lzc, auto rem) {
            ret += (1. + (maxrem - rem) * mri) * INVPOWERSOFTWO[lzc];
        });
        return std::ldexp(1. / ret, int(2 * p_));
    }
    double approx_ec(double n, double m, bool only_lazy=true) const {
        if(n < m) std::swap(n, m);
        double ln = std::log(n);
        if(ln > tq + r_) return std::numeric_limits<double>::max();
        if(only_lazy || ln > p_ + 5. ) {
            const auto minv = 1. / m;
            double d = n * minv / std::pow(((1.0 + n) * minv), 2);
            return std::ldexp(C4 * d, p_ - r_) + 0.5;
        }
        return expected_collisions(n, m) / p_;
    }
    double expected_collisions(double n, double m) const {
        std::fprintf(stderr, "Computing expected collisions, slowly\n");
        // Not optimized, certainly could be
        double x = 0.;
        for(size_t i = 1; i <= tq; ++i) {
            for(size_t j = 1; j <= tr(); ++j) {
                double b1, b2;
                if(i == tq) {
                    // This quantity could be precalculated and cached
                    double di = std::ldexp(1., -int(p_ + r_ + i));
                    b1 = (tr() + j) * di;
                    b2 = (tr() + j + 1.) * di;
                } else {
                    double di = std::ldexp(1., -int(p_ + r_ + i - 1.));
                    b1 = j * di;
                    b2 = (j + 1.) * di;
                }
                double prx = std::pow(1. - b2, n) - std::pow(1. - b1, n);
                double pry = std::pow(1. - b2, m) - std::pow(1. - b1, m);
                x += prx * pry;
            }
        }
        return x * p_ + 0.5;
    }
    double jaccard_index(const hmh_t &o) const {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        uint64_t cc_nc = lrszm3_ == 0
            ? __calc_cc_nc<uint8_t> (o): lrszm3_ == 1
            ? __calc_cc_nc<uint16_t> (o): lrszm3_ == 2
            ? __calc_cc_nc<uint32_t> (o): lrszm3_ == 3
            ? __calc_cc_nc<uint64_t> (o): uint64_t(-1);
        uint32_t cc = cc_nc >> 32, nc = cc_nc & 0xFFFFFFFFu;
        if(!cc) return 0.;

        auto card = cardinality_estimate();
        auto ocard = o.cardinality_estimate();
        auto ec = approx_ec(card, ocard);
        return std::max(0., cc - ec) / nc;
    }
    template<typename IT>
    static INLINE auto count_paired_1bits(IT x) {
        return popcount(((x & 0xFFFFFFFFu) >> 1) & (0x55555555u & x));
    }
    template<typename IT>
    uint64_t __calc_cc_nc(const hmh_t &o) const {
        using Space = vec::SIMDTypes<IT>;
        using Type = typename Space::Type;

        auto start = (const IT *)data_.data(), end = (const IT *)&data_[data_.size()];
        auto ostart = (const IT *)o.data_.data();
        uint32_t cc = 0, nc = 0;
        uint32_t manual_cc = 0, manual_nc = 0;

        // TODO: SIMD optimize
        if(data_.size() < sizeof(Type)) {
            do {
                cc += *start && *start == *ostart;
                nc += *start || *ostart;
                ++ostart; ++start;
            } while(start < end);
        } else {
            const Type *lhp = (const Type *)data_.data(), *lhe = (const Type *)&data_[data_.size()],
                       *rhp = (const Type *)o.data_.data();
            const Type zero = Space::set1(0);
            do { //while(lhp < lhe)
                Type lhv = Space::load(lhp++), rhv = Space::load(rhp++);
#if __AVX512F__
                auto lhnz = Space::cmpneq_mask(lhv, zero);
                auto rhnz = Space::cmpneq_mask(rhv, zero);
                auto anynz = lhnz | rhnz;
                nc += popcount(anynz);
                cc += popcount(Space::cmpeq_mask(lhv, rhv) & anynz);
#else
                Type lh_nonzero = ~Space::cmpeq(lhv, zero);
                Type rh_nonzero = ~Space::cmpeq(rhv, zero);
                Type any_nonzero = Space::or_fn(lh_nonzero, rh_nonzero);
                const Type eq_and_nonzero = Space::and_fn(any_nonzero, Space::cmpeq(lhv, rhv));
#ifndef NDEBUG
                for(uint32_t i = 0; i < sizeof(eq_and_nonzero) / sizeof(IT); ++i) {
                    if(((const IT *)&eq_and_nonzero)[i])
                        ++manual_cc;
                    if(((const IT *)&any_nonzero)[i])
                        ++manual_nc;
                }
#endif
#if __AVX2__
#  define __MOVEMASK8(x) _mm256_movemask_epi8(x)
#  define __MOVEMASK32(x) _mm256_movemask_ps((__m256)x)
#  define __MOVEMASK64(x) _mm256_movemask_pd((__m256d)x)
#elif __SSE2__
#  define __MOVEMASK8(x) _mm_movemask_epi8(x)
#  define __MOVEMASK32(x) _mm_movemask_ps((__m128)x)
#  define __MOVEMASK64(x) _mm_movemask_pd((__m128d)x)
#else
#error("NEED SSE2")
#endif

                CONST_IF(sizeof(IT) == 1) {
                    nc += popcount(__MOVEMASK8(any_nonzero));
                    cc += popcount(__MOVEMASK8(eq_and_nonzero));
                } else CONST_IF(sizeof(IT) == 2) {
                    nc += count_paired_1bits(__MOVEMASK8(any_nonzero));
                    cc += count_paired_1bits(__MOVEMASK8(eq_and_nonzero));
                } else CONST_IF(sizeof(IT) == 4) {
                    nc += popcount(__MOVEMASK32(any_nonzero));
                    cc += popcount(__MOVEMASK32(eq_and_nonzero));
                } else {
                    assert(sizeof(IT) == 8);
                    nc += popcount(__MOVEMASK64(any_nonzero));
                    cc += popcount(__MOVEMASK64(eq_and_nonzero));
                }
#undef __MOVEMASK8
#undef __MOVEMASK32
#undef __MOVEMASK64

#endif // #if AVX512 else sse/avx

            } while(lhp < lhe);
        }
#ifndef NDEBUG
        size_t occ = 0, ncc = 0;
        while(start < end) {
            occ += *start && *start == *ostart;
            ncc += *start || *ostart;
            ++ostart; ++start;
        }
        assert(ncc == nc);
        assert(occ == cc);
        assert(manual_cc == cc);
        assert(manual_nc == nc);
#endif
        return (uint64_t(cc) << 32) | nc;
    }
};

struct HyperMinHash: public hmh_t {
    static constexpr double UNSET_CARD = -std::numeric_limits<double>::max();

    mutable double card_ = UNSET_CARD;

    template<typename...Args>
    HyperMinHash(Args &&...args): hmh_t(std::forward<Args>(args)...) {}

    INLINE double getcard() const {
        if(card_ == UNSET_CARD) {
            card_ = this->cardinality_estimate();
        }
        return card_;
    }

    void unset_card() {card_ = UNSET_CARD;}

    double jaccard_index(const HyperMinHash &o) const {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        uint64_t cc_nc = this->lrszm3_ == 0
            ? this->__calc_cc_nc<uint8_t> (o): this->lrszm3_ == 1
            ? this->__calc_cc_nc<uint16_t> (o): this->lrszm3_ == 2
            ? this->__calc_cc_nc<uint32_t> (o): this->lrszm3_ == 3
            ? this->__calc_cc_nc<uint64_t> (o): uint64_t(-1);
        uint32_t cc = cc_nc >> 32, nc = cc_nc & 0xFFFFFFFFu;
        if(!cc) return 0.;

        auto card = getcard();
        auto ocard = o.getcard();
        auto ec = this->approx_ec(card, ocard);
        return std::max(0., cc - ec) / nc;
    }
};

} // namespace hmh

using hmh::hmh_t;

} // namespace sketch::hmh

#endif /* SKETCH_HMH2_H__ */
