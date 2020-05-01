#ifndef SKETCH_HMH_H__
#define SKETCH_HMH_H__
#include "sketch/mh.h"

namespace sketch {
inline namespace minhash {
template<typename T=uint64_t, typename Hasher=WangHash>
class HyperMinHash {
    uint64_t seeds_ [2] __attribute__ ((aligned (sizeof(uint64_t) * 2)));
    DefaultCompactVectorType core_;
    uint16_t p_, r_;
    Hasher hf_;
#ifndef NOT_THREADSAFE
    std::mutex mutex_; // I should be able to replace most of these with atomics.
#endif
public:
    static constexpr uint32_t q() {return uint32_t(std::ceil(ilog2(sizeof(T) * CHAR_BIT)));} // To hold popcount for a 64-bit integer.
    enum ComparePolicy {
        Manual,
        U8,
        U16,
        U32,
        U64,
    };
    void swap(HyperMinHash &o) {
        std::swap_ranges(reinterpret_cast<uint8_t *>(this), reinterpret_cast<uint8_t *>(this) + sizeof(*this), reinterpret_cast<uint8_t *>(&o));
    }
    auto &core() {return core_;}
    const auto &core() const {return core_;}
    uint64_t mask() const {
        return (uint64_t(1) << r_) - 1;
    }
    auto max_mhval() const {return mask();}
    template<typename... Args>
    HyperMinHash(unsigned p, unsigned r, Args &&...args):
        seeds_{0xB0BAF377C001D00DuLL, 0x430c0277b68144b5uLL}, // Fully arbitrary seeds
        core_(r + q(), 1ull << p),
        p_(p), r_(r),
        hf_(std::forward<Args>(args)...)
    {
        common::detail::zero_memory(core_); // Second parameter is a dummy for interface compatibility with STL
#if VERBOSE_AF
        std::fprintf(stderr, "p: %u. r: %u\n", p, r);
        print_params();
#endif
    }
private:
    HyperMinHash() {}
public:
    void write(gzFile fp) const {
        gzwrite(fp, this, sizeof(*this));
        gzwrite(fp, core_.get(), core_.bytes());
    }
    void write(const char *path) const {
        gzFile fp = gzopen(path, "wb");
        if(fp == nullptr) throw ZlibError(Z_ERRNO, std::string("Could not open file for writingn at ") + path);
        this->write(fp);
        gzclose(fp);
    }
    void read(gzFile fp) const {
        this->clear();
        HyperMinHash tmp;
        gzread(fp, &tmp, sizeof(tmp));
        r_ = tmp.r_;
        p_ = tmp.p_;
        core_ = DefaultCompactVectorType(tmp.r_ + tmp.q(), 1ull << tmp.p_);
        seeds_as_sse() = tmp.seeds_as_sse();
        gzread(fp, core_.get(), core_.bytes());
        std::memset(&tmp, 0, sizeof(tmp));
    }
    void clear() {
        std::memset(core_.get(), 0, core_.bytes());
    }
    HyperMinHash(const HyperMinHash &a): core_(a.r() + q(), 1ull << a.p()), p_(a.p_), r_(a.r_), hf_(a.hf_) {
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
            case 8:  return ComparePolicy::U8;
            case 16: return ComparePolicy::U16;
            case 32: return ComparePolicy::U32;
            case 64: return ComparePolicy::U64;
            default: return ComparePolicy::Manual;
        }
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
    std::array<uint32_t, 64> sum_counts() const {
        using hll::detail::SIMDHolder;
        // TODO: this
        // Note: we have whip out more complicated vectorized maxes for
        // widths of 16, 32, or 64
        std::array<uint32_t, 64> ret{0};
        if(core_.bytes() >= sizeof(SIMDHolder)) {
            switch(simd_policy()) {
#define CASE_U(cse, func, msk)\
                case cse: {\
                    const Space::Type mask = Space::set1(UINT64_C(msk));\
                    for(const SIMDHolder *ptr = reinterpret_cast<const SIMDHolder *>(core_.get()), *eptr = reinterpret_cast<const SIMDHolder *>(core_.get() + core_.bytes());\
                        ptr != eptr; ++ptr) {\
                        auto tmp = *ptr;\
                        tmp = Space::and_fn(Space::srli(*reinterpret_cast<VType *>(&tmp), r_), mask);\
                        tmp.func(ret);\
                    }\
                    break;\
                }
                CASE_U(U8,  inc_counts,   0x3f3f3f3f3f3f3f3f)
                CASE_U(U16, inc_counts16, 0x003f003f003f003f)
                CASE_U(U32, inc_counts32, 0x0000003f0000003f)
                CASE_U(U64, inc_counts64, 0x000000000000003f)
#undef CASE_U
                case Manual: default: goto manual_core;
            }
        } else {
            manual_core:
            for(const auto i: core_) {
                uint8_t lzc = get_lzc(i);
                if(__builtin_expect(lzc > 64, 0)) {std::fprintf(stderr, "Value for %d should not be %dn", int(i), int(get_lzc(i))); std::exit(1);}\
                ++ret[lzc];
            }
        }
        return ret;
    }
#undef MANUAL_CORE
    double estimate_hll_portion(double relerr=1e-2) const {
        return hll::detail::ertl_ml_estimate(this->sum_counts(), p(), q(), relerr);
    }
    double report(double relerr=1e-2) const {
        const auto csum = this->sum_counts();
#if VERBOSE_AF
        std::fprintf(stderr, "Performing estimate. Counts values: ");
        for(const auto v: csum)
            std::fprintf(stderr, "%zu,", v);
        std::fprintf(stderr, "\n");
#endif
        double est = hll::detail::ertl_ml_estimate(csum, p(), 64 - p(), relerr);
        if(est > 0. && est < static_cast<double>(core_.size() << 10)) {
#if VERBOSE_AF
            std::fprintf(stderr, "report ertl_ml_estimate %lf\n", est);
#endif
            return est;
        }
        const double mhinv = std::ldexp(1, -int(r_));
        std::fprintf(stderr, "mhinv: %lf. Manual: %lf\n", mhinv, 1./(1<<r_));
        double sum = 0.;
        for(const auto v: core_) {
            sum += std::ldexp(1. + std::ldexp(get_mhr(v), -int64_t(r_)), -int64_t(get_lzc(v)));
#if VERBOSE_AF
            std::fprintf(stderr, "sum: %lf\n", sum);
#endif
        }
        if(__builtin_expect(!sum, 0)) sum = std::numeric_limits<double>::infinity();
        else                          sum = static_cast<double>(std::pow(core_.size(), 2)) / sum;
        return sum;
    }
    auto p() const {return p_;}
    auto r() const {return r_;}
    void set_seeds(uint64_t seed1, uint64_t seed2) {
        seeds_[0] = seed1; seeds_[1] = seed2;
    }
    void set_seeds(__m128i s) {seeds_as_sse() = s;}
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
        using hll::detail::SIMDHolder;
        // This needs:
        // Vectorized maxes
        if(core_.bytes() >= sizeof(SIMDHolder)) {
            const SIMDHolder *optr = reinterpret_cast<const SIMDHolder *>(o.core_.get());
            SIMDHolder *ptr = reinterpret_cast<SIMDHolder *>(core_.get()),
                       *eptr = reinterpret_cast<SIMDHolder *>(core_.get() + core_.bytes());
            switch(simd_policy()) {
#define CASE_U(cse, op)\
                case cse:\
                    do {*ptr = SIMDHolder::op(*ptr, *optr++);} while(++ptr != eptr); break;
                CASE_U(U8, max_fn)
                CASE_U(U16, max_fn16)
                CASE_U(U32, max_fn32)
                CASE_U(U64, max_fn64)
                default: goto manual;
            }
        } else {
            manual:
            for(size_t i(0); i < core_.size(); ++i)
                if(core_[i] < o.core_[i])
                    core_[i] = o.core_[i];
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
                         lzt(integral::clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
        const uint64_t inserted_val = encode_register(lzt, reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval());
        assert(get_lzc(inserted_val) == lzt);
        assert((reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) == get_mhr(inserted_val));
        if(core_[index] < inserted_val) { // Consider other functions for specific register sizes.
            core_[index] = inserted_val;
            assert(encode_register(lzt, get_mhr(inserted_val)) == inserted_val);
            assert(lzt == get_lzc(inserted_val));
        }
    }
    double jaccard_index(const HyperMinHash &o) const {
        size_t C = 0, N = 0;
        std::fprintf(stderr, "core size: %zu\n", core_.size());
#if 0
        switch(simd_policy()) { // This can be accelerated for specific sizes
            default: //[[fallthrough]];
            //U8:      //[[fallthrough]]; // 2-bit minimizers. TODO: write this
            //U16:     //[[fallthrough]]; // 10-bit minimizers. TODO: write this
            //U32:     //[[fallthrough]]; // 26-bit minimizers. TODO: write this
            //U64:     //[[fallthrough]]; // 58-bit minimizers. TODO: write this
            Manual:
                for(size_t i = 0; i < core_.size(); ++i) {
                    C += core_[i] && (get_lzc(core_[i]) == get_lzc(o.core_[i]));
                    N += (core_[i] || o.core_[i]);
                }
            break;
        }
#else
#endif
        for(size_t i = 0; i < core_.size(); ++i) {
            C += core_[i] && (get_lzc(core_[i]) == get_lzc(o.core_[i]));
            N += (core_[i] || o.core_[i]);
        }
        const double n = this->report(), m = o.report(), ec = expected_collisions(n, m);
        std::fprintf(stderr, "C: %zu. ec: %lf. C / N: %lf\n", C, ec, static_cast<double>(C) / N);
        return std::max((C - ec) / N, 0.);
    }
    double expected_collisions(double n, double m, bool easy_way=false) const {
#if MY_WAY
        if(easy_way) {
            if(n < m) std::swap(n, m);
            auto l2n = ilog2(n);
            if(l2n > ((1 << q()) + r())) {
#if VERBOSE_AF
                std::fprintf(stderr, "Warning: too high to approximate\n");
#endif
                goto slow;
            }
            if(l2n > p() + 5) {
                const double nm = n/m;
                const double phi = 4 * nm / std::pow((1 + n) / m, 2);
                return std::ldexp(detail::HMH_C * phi, p_ - r());
            }
        }
        slow:
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
        return easy_way ? x: std::ldexp(x, p());
#else
        size_t r2 = 1ull << r(), q2 = 1ull << q();
        double x = 0;
        for(size_t i = 1; i <= q2; ++i) {
            for(size_t j = 1; j <= r2; ++j) {
                auto b1 = i != q2 ? std::ldexp(r2 + j, -int32_t(p() + r() + i)): std::ldexp(j, -int32_t(p() + r() + i - 1));
                auto b2 = i != q2 ? std::ldexp(r2 + j + 1, -int32_t(p() + r() + i)): std::ldexp(j + 1,  -int32_t(p() + r() + i - 1));
                auto prx = std::pow(1 - b2, n) - std::pow(1 - b1, n);
                auto pry = std::pow(1 - b2, m) - std::pow(1 - b1, m);
                x += prx * pry;
            }
        }
        return std::ldexp(x, p());
#endif
    }
};

template<typename T, typename Hasher>
void swap(HyperMinHash<T,Hasher> &a, HyperMinHash<T,Hasher> &b) {a.swap(b);}

} // namespace mh

} // namespace sketch

#endif
