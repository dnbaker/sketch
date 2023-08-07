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


static constexpr const std::array<double, 64> INVPOWERSOFTWO = {
1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05, 3.0517578125e-05, 1.52587890625e-05, 7.62939453125e-06, 3.814697265625e-06, 1.9073486328125e-06, 9.5367431640625e-07, 4.76837158203125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 2.9802322387695312e-08, 1.4901161193847656e-08, 7.450580596923828e-09, 3.725290298461914e-09, 1.862645149230957e-09, 9.313225746154785e-10, 4.656612873077393e-10, 2.3283064365386963e-10, 1.1641532182693481e-10, 5.820766091346741e-11, 2.9103830456733704e-11, 1.4551915228366852e-11, 7.275957614183426e-12, 3.637978807091713e-12, 1.8189894035458565e-12, 9.094947017729282e-13, 4.547473508864641e-13, 2.2737367544323206e-13, 1.1368683772161603e-13, 5.684341886080802e-14, 2.842170943040401e-14, 1.4210854715202004e-14, 7.105427357601002e-15, 3.552713678800501e-15, 1.7763568394002505e-15, 8.881784197001252e-16, 4.440892098500626e-16, 2.220446049250313e-16, 1.1102230246251565e-16, 5.551115123125783e-17, 2.7755575615628914e-17, 1.3877787807814457e-17, 6.938893903907228e-18, 3.469446951953614e-18, 1.734723475976807e-18, 8.673617379884035e-19, 4.336808689942018e-19, 2.168404344971009e-19, 1.0842021724855044e-19
};

#define SHOW_CASES(CASE_MACRO)  \
        CASE_MACRO(uint8_t, 0, 2); \
        CASE_MACRO(uint16_t, 1, 10); \
        CASE_MACRO(uint32_t, 2, 26); \
        CASE_MACRO(uint64_t, 3, 58);

/*
 * The HyperMinHash paper directs to subtract the expected collisions;
 * however, this doesn't account for the 'true' positives
 * and lower error rates are achieved by using the scaled
 * BIAS_SUB methods
 */

struct hmh_t {
protected:
    uint64_t rbm_;
    uint16_t p_, r_, lrszm3_;
    std::vector<uint8_t, Allocator<uint8_t>> data_;
    long double pmrp2_;
public:
    static constexpr long double C4 = 0.6796779486389563759078484561637623073693248443305492L;
    hmh_t(unsigned p, unsigned rsize=8):
          rbm_((uint64_t(1) << (rsize - q)) - 1),
          p_(p), r_(rsize-q),
          lrszm3_(ilog2(rsize) - 3), pmrp2_(std::ldexp(C4, p_ - r_))
    {
        PREC_REQ(rsize == 8 || rsize == 16 || rsize == 32 || rsize == 64, "Must have 8, 16, 32, or 64 for register size");
        PREC_REQ(p_ >= 3 && p_ < 64, "p can't be less than 3 or >= 64");
        PREC_REQ((uint64_t(rsize) << p_) >= 64, "We require at least 64 bits of storage");
        lrszm3_ = ilog2(rsize) - 3;
        data_.resize(rsize << (p_ - 3));
    }
    hmh_t(gzFile fp) {
        this->read(fp);
    }
    hmh_t(std::string path) {
        this->read(path);
    }
    bool operator==(const hmh_t &o) const {
        return rbm_ == o.rbm_ && p_ == o.p_ &&
            std::equal(reinterpret_cast<const uint64_t *>(&data_[0]),
                       reinterpret_cast<const uint64_t *>(&*data_.end()),
                       reinterpret_cast<const uint64_t *>(&o.data_[0])
            );
    }
    bool operator!=(const hmh_t &o) const {
        return !this->operator==(o);
    }

    // Constants, encoding, and decoding utilities
    static constexpr unsigned q  = 6;
    static constexpr unsigned tq = 1ull << 6;


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
    const IT *get_dataptr() const {
        return static_cast<const IT *>(data_.data());
    }
    template<typename IT>
    hmh_t &perform_merge(const hmh_t &o) {
#ifndef VEC_DISABLED_H__
        using Space = vec::SIMDTypes<IT>;
        using Type = typename Space::Type;
        auto d = reinterpret_cast<Type *>(data_.data());
        auto e = reinterpret_cast<const Type *>(&data_[data_.size()]);
        auto od = reinterpret_cast<const Type *>(o.data_.data());
        do {
            Space::store(d, Space::max(Space::load(d), Space::load(od)));
            ++d, ++od;
        } while(d < e);
#else
        std::transform(reinterpret_cast<IT *>(data_.data()), reinterpret_cast<IT *>(data_[data_.size()]), reinterpret_cast<IT *>(other.data_.data())
                       reinterpret_cast<IT *>(data_.data()), [](auto x, auto y) {return std::max(x, y);});
#endif
        return *this;
    }
    hmh_t &operator+=(const hmh_t &o) {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, index, __unused) case index: return perform_merge<type>(o)
            SHOW_CASES(CASE_U)
            default: return *this;
        }
    }
    hmh_t operator+(const hmh_t &o) const {
        hmh_t ret(*this);
        ret += o;
        return ret;
    }


    template<typename IT1, typename IT2, typename IT3, typename IT4=std::common_type_t<IT1, IT2, IT3>>
    static inline constexpr IT4 encode_register(IT1 r, IT2 lzc, IT3 rem) {
        static_assert(sizeof(IT4) >= std::max(std::max(sizeof(IT1), sizeof(IT2)), sizeof(IT3)), "size");
        IT4 enc = (IT4(lzc) << r) | rem;
        assert((enc >> r) == lzc);
        assert(reg2lzc(enc, r) == lzc);
        assert((enc & ((1ull << r) - 1)) == rem);
        assert((enc % (1ull << r)) == rem);
        return enc;
    }
    template<typename IT>
    static inline constexpr std::pair<IT, IT> decode_register(IT value, IT r) {
        return {value >> r, value & ((IT(1) << r) - 1)};
    }
    template<typename IT>
    static inline constexpr std::pair<IT, IT> decode_register(IT value, IT r, IT bm) {  // cache bitmask
        return {reg2lzc(value,r), reg2rem(value, bm)}; // same as reg2lzc, reg2rem
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
        DBG_ONLY(size_t i = 0;)
        for_each_union_register(o, [&func,rbm=rbm_,r=r_ DBG_ONLY(,&i,this)](auto x) {
            //DBG_ONLY(std::fprintf(stderr, "[%p] ulzrem: egister #%zu=%zu -> %zu lzc, %zu rem, with r = %d\n", (void *)this, ++i, size_t(x), size_t(reg2lzc(x, r)), size_t(reg2rem(x, rbm)), r);)
            func(reg2lzc(x, r), reg2rem(x, rbm));
        });
    }
    template<typename Func>
    void for_each_lzrem(const Func &func) const {
        DBG_ONLY(size_t i = 0;)
        for_each_register([&func,rbm=rbm_,r=r_ DBG_ONLY(,&i,this)](auto x) {
            //DBG_ONLY(std::fprintf(stderr, "[%p] lzrem: egister #%zu=%zu -> %zu lzc, %zu rem, with r = %d\n", (void *)this, ++i, size_t(x), size_t(reg2lzc(x, r)), size_t(reg2rem(x, rbm)), r);)
            func(reg2lzc(x, r), reg2rem(x, rbm));
        });
    }
    uint64_t calculate_cc_nc(const hmh_t &o) const {
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, i, __unused) case i: return __calc_cc_nc<type>(o); break
            SHOW_CASES(CASE_U)
            default: HEDLEY_UNREACHABLE();
        }
        return -1;
    }

#ifndef VEC_DISABLED_H__
    template<typename IT, typename Func>
    void __for_each_union_vector(const hmh_t &o, const Func &func) const {
        using Space = vec::SIMDTypes<IT>;
        auto d = reinterpret_cast<const typename Space::Type *>(data_.data());
        const auto e  = d + ((num_registers() / Space::COUNT) * Space::COUNT);
        auto od = reinterpret_cast<const typename Space::Type *>(o.data_.data());
        while(d != e) {
            func(Space::max(Space::load(d), Space::load(od)));
            ++d, ++od;
        }
    }
#endif
    template<typename IT, typename Func>
    void __for_each_union_register(const hmh_t &o, const Func &func) const {
        using Space = vec::SIMDTypes<IT>;
        using VType = typename Space::VType;

        auto d  = reinterpret_cast<const typename Space::Type *>(data_.data());
        auto e  = d + (num_registers() / Space::COUNT);
        auto od = reinterpret_cast<const typename Space::Type *>(o.data_.data());
        while(d < e) {
            auto lhv = Space::load(d++), rhv = Space::load(od++);
            VType maxv = Space::max(lhv, rhv);
#ifndef NDEBUG
            bool pass = true;
            for(size_t i = 0; i < sizeof(lhv) / sizeof(IT); ++i) {
                IT _lh = ((IT *)&lhv)[i],
                   _rh = ((IT *)&rhv)[i],
                   _mh = ((IT *)&maxv)[i];
                if(_mh != std::max(_lh, _rh)) {
                    std::fprintf(stderr, "lh %zu, rh %zu, maxv %zu\n", size_t(_lh), size_t(_rh), size_t(_mh));
                    pass = false;
                }
            }
            if(!pass) throw std::runtime_error("Incorrect Max calculation.");
#endif
            maxv.for_each(func);
        }
        for(const IT *w = (const IT *)d, *e = (const IT *)&data_[data_.size()]; w < e; func(*w++));
    }
    template<typename Func>
    void for_each_union_register(const hmh_t &o, const Func &func) const {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, i, __UNUSED) case i: __for_each_union_register<type>(o, func); break
        SHOW_CASES(CASE_U)
            default: HEDLEY_UNREACHABLE();
        }
    }

    template<typename Func>
    void for_each_register(const Func &func) const {
        const uint8_t *const s = data_.data(), *const e = &s[data_.size()];
        auto fe = [&](auto start, auto end) {
            SK_UNROLL_8
            while(start != end)
                func(*start++);
        };
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, index, __UNUSED) case index: fe(reinterpret_cast<const type *>(s), reinterpret_cast<const type *>(e)); break
            SHOW_CASES(CASE_U)
            default: HEDLEY_UNREACHABLE();
        }
    }
    template<typename Func>
    void for_each_register_pair(const hmh_t &o, const Func &func) const {
        const uint8_t *const s = data_.data(), *const e = &s[data_.size()];
        auto fe = [&](auto startp, auto endp, auto op) {
            SK_UNROLL_8
            do {
                func(*startp++, *op++);
            } while(startp != endp);
        };
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, index, __unused) case index: fe(reinterpret_cast<const type *>(s), reinterpret_cast<const type *>(e), reinterpret_cast<const type *>(o.data_.data())); break
            SHOW_CASES(CASE_U)
            default: HEDLEY_UNREACHABLE();
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
#undef CASE_U
#define CASE_U(type, index, __UNUSED) case index: perform_add<type>(h1, h2); break
            SHOW_CASES(CASE_U)
#undef CASE_U
            default: HEDLEY_UNREACHABLE();
        }
    }
    template<typename IT> IT access(size_t index) const {
        return reinterpret_cast<const IT *>(data_.data())[index];
    }
#ifndef _mm512_srli_epi16
#define _mm512_srli_epi16(mm, Imm) _mm512_and_si512(_mm512_set1_epi16(0xFFFFu >> Imm), _mm512_srli_epi32(mm, Imm))
#endif
#ifndef _mm512_srli_epi8
#define _mm512_srli_epi8(mm, Imm) _mm512_and_si512(_mm512_set1_epi8(0xFFu >> Imm), _mm512_srli_epi32(mm, Imm))
#endif
    template<typename IT>
    std::array<uint32_t, 64> sum_counts() const {
        using hll::detail::SIMDHolder;
        using Space = vec::SIMDTypes<IT>;
        std::array<uint32_t, 64> ret{0};
        if(data_.size() >= sizeof(SIMDHolder)) {
            const typename Space::Type mask = Space::set1(0x3Fu);
            auto update_point = [&](auto x) {
#if __AVX512BW__ || ((__AVX2__ || __SSE2__) && !(__AVX512F__))
                CONST_IF(sizeof(IT) == 1) {
                    SIMDHolder(Space::and_fn(Space::srli(x, 2), mask)).inc_counts_by_type<IT>(ret);
                } else CONST_IF(sizeof(IT) == 2) {
                    SIMDHolder(Space::and_fn(Space::srli(x, 10), mask)).inc_counts_by_type<IT>(ret);
                } else CONST_IF(sizeof(IT) == 4) {
                    SIMDHolder(Space::and_fn(Space::srli(x, 26), mask)).inc_counts_by_type<IT>(ret);
                } else {
                    SIMDHolder(Space::and_fn(Space::srli(x, 58), mask)).inc_counts_by_type<IT>(ret);
                }
#elif __AVX512F__
                CONST_IF(sizeof(IT) == 1) {
                    SIMDHolder(_mm512_and_si512(_mm512_srli_epi8(x, 2), mask)).inc_counts_by_type<IT>(ret);
                    assert(r_ == 2);
                } else CONST_IF(sizeof(IT) == 2) {
                    SIMDHolder(_mm512_and_si512(_mm512_srli_epi16(x, 10), mask)).inc_counts_by_type<IT>(ret);
                    assert(r_ == 10);
                } else CONST_IF(sizeof(IT) == 4) {
                    SIMDHolder(_mm512_and_si512(_mm512_srli_epi32(x, 26), mask)).inc_counts_by_type<IT>(ret);
                    assert(r_ == 26);
                } else {
                    SIMDHolder(_mm512_and_si512(_mm512_srli_epi64(x, 58), mask)).inc_counts_by_type<IT>(ret);
                    assert(r_ == 58);
                }
#else
            throw std::runtime_error("HMH is only supported on x86-64");
#endif
            };
            auto ptr = reinterpret_cast<const SIMDHolder *>(data_.data());
            auto eptr = reinterpret_cast<const SIMDHolder *>(&data_[data_.size()]);
            while(eptr - ptr > 8) {
                update_point(ptr[0]); update_point(ptr[1]); update_point(ptr[2]); update_point(ptr[3]);
                update_point(ptr[4]); update_point(ptr[5]); update_point(ptr[6]); update_point(ptr[7]);
                ptr += 8;
            }
            while(ptr < eptr) update_point(*ptr++);
        } else for_each_register([&](auto x) {++ret[reg2lzc(x, r_)];});
#if !NDEBUG
        std::array<uint32_t, 64> cmp{0};
        for_each_register([&](auto x) {++cmp[reg2lzc(x, r_)];});
        assert(std::equal(ret.begin(), ret.end(), cmp.begin()));
#endif
        assert(std::accumulate(ret.begin(), ret.end(), size_t(0)) == (1ull << p_));
        return ret;
    }
#undef _mm512_srli_epi16
#undef _mm512_srli_epi8
    template<typename IT>
    INLINE void perform_add(uint64_t h1, uint64_t h2) {
        const IT reg = encode_register(r_,
                                       clz(((h1 << 1)|1) << (p_ - 1)) + 1,
                                       h2 & rbm_);
        IT &r = ((IT *)data_.data())[h1 >> max_lremainder()];
        assert(&r < (IT *)&*data_.end());
#ifdef NOT_THREADSAFE
        if(reg > r) r = reg;
#else
        while(reg > r) __sync_bool_compare_and_swap(&r, r, reg);
#endif
    }
    double estimate_hll_portion() const {
        double ret;
        switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, index, __UNUSED) case index: ret = hll::detail::ertl_ml_estimate(this->sum_counts<type>(), p_, 64 - p_); break
            SHOW_CASES(CASE_U)
            default: HEDLEY_UNREACHABLE();
        }
        return std::max(ret, 0.);
    }
    double cardinality_estimate() const {
        double ret = estimate_mh_portion();
        if(ret < (1024 << p_))
            ret = estimate_hll_portion();
        return ret;
    }
    static INLINE double mhsum2ret(double ret, int p) {
        //std::fprintf(stderr, "mhsum: %g. p: %d, ret: %g\n", ret, p, std::ldexp(1. / ret, 2 * p));
        return std::ldexp(1. / ret, 2 * p);
    }
    template<size_t N, typename IT, typename OIT>
    static void __lzrem_func(IT lzc, OIT rem, double &ret) {
        static constexpr double mri = 1.L / static_cast<long double>((uint64_t(1) << N) - 1);
        static constexpr double mrx2 = 2.L * static_cast<long double>((uint64_t(1) << N) - 1);
        ret += std::ldexp((mrx2 - rem) * mri,
                           -static_cast<std::make_signed_t<IT>>(lzc));
        // TODO: Better manual intrinsics
    }
    double estimate_mh_portion() const {
        double ret = 0.;
        for_each_lzrem([&ret,r=this->r_](auto lzc, auto rem) {
            if(r == 2) __lzrem_func<2>(lzc, rem, ret);
            else if(r == 10) __lzrem_func<10>(lzc, rem, ret);
            else if(r == 26) __lzrem_func<26>(lzc, rem, ret);
            else __lzrem_func<56>(lzc, rem, ret);
            //ret += (1. + (maxrem - rem) * mri) * INVPOWERSOFTWO[lzc];
            // We substitute     (2 * maxrem - rem) * mri
            // for               (1 + (mr - rem) * mri)
            // which saves one operation per iteration
        });
        return mhsum2ret(ret, p_);
    }
    double card_ji(const hmh_t &o) const {
        double mv = this->cardinality_estimate(), ov = o.cardinality_estimate();
        double us = union_size(o);
        return std::max(0., mv + ov - us); // Inclusion-exclusion principle
    }
    double union_size(const hmh_t &o) const {
        double ret = 0.;
#ifndef VEC_DISABLED_H__
        if(data_.size() >= sizeof(vec::SIMDTypes<uint64_t>::Type)) {
            const uint64_t maxremi = max_remainder();
            const double maxrem = maxremi;
            const double mri = 1. / (maxrem);
            const double mrx2 = 2. * maxrem;
            switch(lrszm3_) {
#undef CASE_U
#define CASE_U(type, i, rshift) case i: \
            __for_each_union_vector<type>(o, [&](const auto v) noexcept { \
                using Space = vec::SIMDTypes<type>;\
                using VType = Space::VType;\
                const auto lzcs = VType(Space::srli(v, rshift));\
                const auto rems = Space::and_fn(v, Space::set1(maxremi));\
                for(unsigned j = 0; j < sizeof(VType) / sizeof(type); ++j) \
                    ret += mrx2 - double(((const type *)&rems)[j]) * mri * INVPOWERSOFTWO[((const type *)&lzcs)[j]];\
            }); break
            SHOW_CASES(CASE_U)
                default: HEDLEY_UNREACHABLE();
            }
            return mhsum2ret(ret, p_);
        }
#endif
        for_each_union_lzrem(o, [&](const auto lzc, const auto rem) noexcept {
            switch(r_) {
                case 2: __lzrem_func<2>(lzc, rem, ret); break;
                case 10: __lzrem_func<10>(lzc, rem, ret); break;
                case 26: __lzrem_func<26>(lzc, rem, ret); break;
                case 58: __lzrem_func<58>(lzc, rem, ret); break;
                default: __builtin_unreachable(); break;
            }
        });
        return mhsum2ret(ret, p_);
    }
    double approx_ec(double n, double m, int laziness=1) const {
        if(n < m) std::swap(n, m);
        const double ln = std::log(n);
        if(ln > tq + r_) return std::numeric_limits<double>::max();
        if(laziness > 1 || ln > p_ + 5. ) {
            const auto minv = 1.L / m;
            const auto nmv = static_cast<long double>(n) * minv;
            return nmv * pmrp2_ * std::pow(nmv + minv, -2);
        }
        if(laziness > 0) return hll_lazy_collision_estimate(n, m);
        return expected_collisions(n, m) / p_;
    }
    double expected_collisions(double n, double m) const {
        // Optimizations:
        // 1. Precalculate di contributions
        // 2. Use INVPOWERSOFTWO table for p + r + i < 64
        // See notes on hll_lazy_collision_estimate for
        // issues with improving its scalability
        double x = 0.;
        auto incx = [&x,n,m](auto b1, auto b2) {
            auto prx = std::pow(1. - b2, n) - std::pow(1. - b1, n);
            auto pry = std::pow(1. - b2, m) - std::pow(1. - b1, m);
            x += prx * pry;
        };
        auto get_invpow2 = [&](int x) {return x < 64 ? INVPOWERSOFTWO[x]: std::ldexp(1., -x);};
        for(size_t i = 1; i < tq; ++i) { // Note that this is now < tq, not <=
            const double di = get_invpow2(p_ + r_ + i - 1);
            double b1 = 0, b2 = di;
            for(size_t j = 1; j <= tr(); ++j) {
                b1 += di; b2 += di;
                incx(b1, b2);
            }
        }
        // We handle the last iteration (where i == tq)
        const double di = get_invpow2(p_ + r_ + tq);
        double b1 = tr() * di, b2 = b1 + di;
        for(size_t j = 0; j < tr(); b1 += di, b2 += di, incx(b1, b2), ++j);
        return x * p_ + 0.5;
    }
    double hll_lazy_collision_estimate(double n, double m) const {
        double x = 0.;
        double b1, b2, px, py;
        // Note:
        // 1. doubles run out of space at p = 10 (technically 1075 registers)
        //    This could be addressed by moving the accumulation into log-space and
        //    using the logsumexp trick when possible.
        //    long doubles run out of space at p = 14, num reg = 16645
        // 2. This is pretty inefficient -- we double the number of pow calls by repeating them for differing values of b1/b2
        //    We could save this be precalculating powers of n andm for all inverse powers of two
        //    which would allow us to save time with vectorized pow calls through.
        //    But that's only particularly useful if you can hold the values in floats, which you can't.
        // 3. This only happens hll est card is <= 5. * num registers, so it is more of an edge case anyway.
        std::vector<double> powers(this->num_registers());
        double v = 1.;
        for(size_t i = 0; i < this->num_registers(); ++i) {
            powers[i] = v; v *= .5;
        }
        SK_UNROLL_8
        for(size_t i = 0, e = this->num_registers() - 1; i < e; ++i) {
            b1 = powers[i + 1];
            b2 = powers[i];
            px = std::pow(1. - b1, n) - std::pow(1. - b2, n);
            py = std::pow(1. - b1, m) - std::pow(1. - b2, m);
            x += px * py;
        }
        int exp = (this->q - this->r_);
        if(exp < 0) return x * INVPOWERSOFTWO[-exp];
        return std::ldexp(x, exp);
    }
    double jaccard_index(const hmh_t &o) const {
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        uint64_t cc_nc = calculate_cc_nc(o);
        uint32_t cc = cc_nc >> 32, nc = cc_nc & 0xFFFFFFFFu;
        if(!cc) return 0.;

        auto card = cardinality_estimate();
        auto ocard = o.cardinality_estimate();
        assert(this != &o || card == ocard);
        auto ec = approx_ec(card, ocard);
        return (1. - double(ec) / nc) * cc;
    }
    template<typename IT>
    static INLINE auto count_paired_1bits(IT x) {
        static constexpr IT bitmask = static_cast<IT>(0x5555555555555555uLL);
        return popcount((x >> 1) & x & bitmask);
    }
    template<typename IT>
    uint64_t __calc_cc_nc(const hmh_t &o) const {

        auto start = (const IT *)data_.data(), end = (const IT *)&data_[data_.size()];
        auto ostart = (const IT *)o.data_.data();
        uint32_t cc = 0, nc = 0;
        if(data_.size() < VECTOR_WIDTH / sizeof(IT)) {
            do {
                cc += *start && *start == *ostart;
                nc += *start || *ostart;
                ++ostart; ++start;
            } while(start < end);
        }
#if __AVX512BW__ || __AVX2__ || __SSE2__
        else {
#if __AVX512BW__ // TODO: replace this with a (potentially separate) check per type
            using Type = typename vec::SIMDTypes<IT>::Type;
            const Type *lhp = (const Type *)data_.data(), *lhe = (const Type *)&data_[data_.size()],
                       *rhp = (const Type *)o.data_.data();
            const Type zero = Space::set1(0);
            SK_UNROLL_4
            do { //while(lhp < lhe)
                Type lhv = Space::load(lhp++), rhv = Space::load(rhp++);
                auto lhnz = Space::cmpneq_mask(lhv, zero);
                auto rhnz = Space::cmpneq_mask(rhv, zero);
                auto anynz = lhnz | rhnz;
                nc += popcount(anynz);
                cc += popcount(Space::cmpeq_mask(lhv, rhv) & anynz);
            } while(lhp < lhe);
#elif __AVX2__ || __SSE2__

#if __AVX2__
#  define __MOVEMASK8(x) _mm256_movemask_epi8(x)
#  define __MOVEMASK32(x) _mm256_movemask_ps((__m256)x)
#  define __MOVEMASK64(x) _mm256_movemask_pd((__m256d)x)
#  define __SETZERO() _mm256_set1_epi32(0)
#  define __CMPEQ8(x, y) _mm256_cmpeq_epi8(x, y)
#  define __CMPEQ16(x, y) _mm256_cmpeq_epi16(x, y)
#  define __CMPEQ32(x, y) _mm256_cmpeq_epi32(x, y)
#  define __CMPEQ64(x, y) _mm256_cmpeq_epi64(x, y)
#  define TYPE __m256i
#elif __SSE2__
#  define __MOVEMASK8(x) _mm_movemask_epi8(x)
#  define __MOVEMASK32(x) _mm_movemask_ps((__m128)x)
#  define __MOVEMASK64(x) _mm_movemask_pd((__m128d)x)
#  define __CMPEQ8(x, y) _mm_cmpeq_epi8(x, y)
#  define __CMPEQ16(x, y) _mm_cmpeq_epi16(x, y)
#  define __CMPEQ32(x, y) _mm_cmpeq_epi32(x, y)
#  if __SSE4_1__
#    define __CMPEQ64(x, y) _mm_cmpeq_epi64(x, y)
#  else
#    define __CMPEQ64(x, y) _mm_and_si128(_mm_cmpeq_epi32(x, y), _mm_cmpeq_epi32(_mm_srli_epi64(x, 32), _mm_srli_epi64(y, 32)))
#  endif
#  define __SETZERO() _mm_set1_epi32(0)
#  define TYPE __m128i
#endif
            const TYPE *lhp = (const TYPE *)data_.data(), *lhe = (const TYPE *)&data_[data_.size()],
                       *rhp = (const TYPE *)o.data_.data();
            const TYPE zero = __SETZERO();
            SK_UNROLL_4
            do {
                TYPE lh_nonzero, rh_nonzero, any_nonzero;
                TYPE lhv = *lhp++, rhv = *rhp++;
                CONST_IF(sizeof(IT) == 1) {
                    lh_nonzero = ~__CMPEQ8(lhv, zero);
                    rh_nonzero = ~__CMPEQ8(rhv, zero);
                    any_nonzero = lh_nonzero | rh_nonzero;
                    const TYPE eq_and_nonzero = any_nonzero & __CMPEQ8(lhv, rhv);
                    nc += popcount(__MOVEMASK8(any_nonzero));
                    cc += popcount(__MOVEMASK8(eq_and_nonzero));
                } else CONST_IF(sizeof(IT) == 2) {
                    lh_nonzero = ~__CMPEQ16(lhv, zero);
                    rh_nonzero = ~__CMPEQ16(rhv, zero);
                    any_nonzero = lh_nonzero | rh_nonzero;
                    const TYPE eq_and_nonzero = any_nonzero & __CMPEQ16(lhv, rhv);
                    nc += count_paired_1bits(__MOVEMASK8(any_nonzero));
                    cc += count_paired_1bits(__MOVEMASK8(eq_and_nonzero));
                } else CONST_IF(sizeof(IT)== 4) {
                    lh_nonzero = ~__CMPEQ32(lhv, zero);
                    rh_nonzero = ~__CMPEQ32(rhv, zero);
                    any_nonzero = lh_nonzero | rh_nonzero;
                    const TYPE eq_and_nonzero = any_nonzero & __CMPEQ32(lhv, rhv);
                    nc += popcount(__MOVEMASK32(any_nonzero));
                    cc += popcount(__MOVEMASK32(eq_and_nonzero));
                } else {
                    lh_nonzero = ~__CMPEQ64(lhv, zero);
                    rh_nonzero = ~__CMPEQ64(rhv, zero);
                    any_nonzero = lh_nonzero | rh_nonzero;
                    const TYPE eq_and_nonzero = any_nonzero & __CMPEQ64(lhv, rhv);
                    nc += popcount(__MOVEMASK64(any_nonzero));
                    cc += popcount(__MOVEMASK64(eq_and_nonzero));
                }
            } while(lhp < lhe);


#endif // #if __AVX512BW__ else sse/avx
        } // if avx2 or 512 or sse2
#endif
        return (uint64_t(cc) << 32) | nc;
    }
#undef TYPE
#undef __SETZERO
#undef __MOVEMASK8
#undef __MOVEMASK32
#undef __MOVEMASK64
#undef __CMPEQ8
#undef __CMPEQ16
#undef __CMPEQ32
#undef __CMPEQ64

    void write(gzFile fp) const {
        uint8_t buf[2];
        buf[0] = p_;
        buf[1] = lrszm3_;
        gzwrite(fp, buf, sizeof(buf));
        gzwrite(fp, data_.data(), data_.size());
    }
    void write(std::string path) const {
        gzFile fp = gzopen(path.data(), "wb");
        if(!fp) throw ZlibError(std::string("Failed to open file at ") + path + " for writing");
        write(fp);
        gzclose(fp);
    }
    void read(std::string path) {
        gzFile fp = gzopen(path.data(), "rb");
        if(!fp) throw ZlibError(std::string("Failed to open file at ") + path + " for reading");
        this->read(fp);
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint8_t buf[2];
        gzread(fp, buf, sizeof(buf));
        p_ = buf[0];
        lrszm3_ = buf[1];
        PREC_REQ(lrszm3_ <= 3, "Illegal lrszm3_");
        static constexpr std::uint8_t rlut[] {2, 10, 26, 58};
        r_ = rlut[lrszm3_];
        data_.resize(1ull << (p_ + lrszm3_));
        gzread(fp, data_.data(), data_.size());
        rbm_ = (1ull << r_) - 1;
        pmrp2_ = std::ldexp(C4, p_ - r_);
    }
    void clear() {
        std::memset(data_.data(), 0, data_.size());
    }
    void reset() {clear();}
    void free() {
        auto tmp(std::move(data_));
        rbm_ = p_ = r_ = lrszm3_ = 0;
    }
};

template<typename Hasher=hash::WangHash>
struct HyperMinHasher: public hmh_t {
    static constexpr double UNSET_CARD = -std::numeric_limits<double>::max();

    mutable double card_ = UNSET_CARD;
    Hasher hf_;


    template<typename...Args>
    HyperMinHasher(Args &&...args): hmh_t(std::forward<Args>(args)...) {}
    template<typename...Args>
    HyperMinHasher(Hasher &&hf, Args &&...args): hmh_t(std::forward<Args>(args)...), hf_(std::move(hf)) {}

    bool operator==(const HyperMinHasher &o) const {
        return hmh_t::operator==(o);
    }
    bool operator!=(const HyperMinHasher &o) const {
        return hmh_t::operator!=(o);
    }

    HyperMinHasher& operator+=(const HyperMinHasher &o) {
        hmh_t::operator+=(o);
        card_ = this->cardinality_estimate();
        return *this;
    }
    HyperMinHasher operator+(const HyperMinHasher &o) const {
        HyperMinHasher ret(*this);
        ret += o;
        return ret;
    }

    INLINE double getcard() const {
        if(card_ == UNSET_CARD) card_ = this->cardinality_estimate();
        return card_;
    }

    void unset_card() {card_ = UNSET_CARD;}


    void add(uint64_t hv) {
        uint64_t h2 = wy::wyhash64_stateless(&hv);
        hmh_t::add(hv, h2);
    }

    template<typename...Args>
    void addh(Args &&...args) {
        add(hf_(std::forward<Args>(args)...));
    }

    double containment_index(const HyperMinHasher &o) const {
        if(unlikely(this == &o)) return 1.;
        double msz = this->getcard();
        return msz ? intersection_size(o) / msz: 1.;
    }
    std::array<double, 3> full_set_comparison(const HyperMinHasher &o) const {
        auto mycard = this->getcard(), ocard = o.getcard();
        auto is = this->intersection_size(o);
        return std::array<double, 3>{std::max(mycard - is, 0.), std::max(ocard - is, 0.), is};
    }

    double jaccard_index(const HyperMinHasher &o) const {
        if(unlikely(this == &o)) return 1.;
        PREC_REQ(o.p_ == this->p_ && o.r_ == this->r_, "Must have matching parameters");
        uint64_t cc_nc = calculate_cc_nc(o);
        uint32_t cc = cc_nc >> 32, nc = cc_nc & 0xFFFFFFFFu;
        if(!cc) return 0.;
        else if(cc == nc) return 1.;
        auto ec = this->approx_ec(getcard(), o.getcard());
        return (1. - double(ec) / nc) * cc / nc;
    }
    double intersection_size(const HyperMinHasher &o) const {
        return this->union_size(o) * jaccard_index(o);
    }
    double card_ji(const HyperMinHasher &o) const {
        double mv = this->getcard(), ov = o.getcard();
        double us = union_size(o);
        double ret = std::max(0., mv + ov - us) / us;
        return ret; // Inclusion-exclusion principle
    }

    using final_type = HyperMinHasher<Hasher>;
};
#undef SHOW_CASES

} // namespace hmh

using hmh::hmh_t;
using hmh::HyperMinHasher;
using HyperMinHash = HyperMinHasher<>;

} // namespace sketch::hmh


#endif /* SKETCH_HMH2_H__ */
