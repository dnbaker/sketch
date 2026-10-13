#pragma once
#ifndef SKETCH_EXALOGLOG_H__
#define SKETCH_EXALOGLOG_H__
/*
 * ExaLogLog: space-efficient approximate distinct counting up to the
 * exa-scale (O. Ertl, arXiv:2402.13726, EDBT 2025).
 *
 * Generalizes the UltraLogLog data structure, replacing the geometric
 * update-value distribution with the power-of-two distribution
 *
 *     rho_update(k) = 1 / 2^(t + 1 + floor((k-1)/2^t)),
 *
 * which approximates a geometric distribution with base b = 2^(2^-t) but
 * can be generated from a 64-bit hash with a few branch-free instructions.
 * Each register stores a (6+t)-bit maximum update value plus d extra bits
 * marking the occurrences of the d next-smaller update values; registers
 * are bit-packed (6+t+d bits each).
 *
 * Recommended configurations (paper, Fig. 4): (t=2, d=20) with 28-bit
 * registers and theoretical MVP 3.67 (-43% vs 6-bit HLL), (t=2, d=24)
 * with 32-bit registers for aligned access, (t=1, d=9) with 16-bit
 * registers. Estimation is bias-reduced maximum likelihood using a
 * Newton/secant solver over a sketch-specific small set of coefficients;
 * martingale (HIP) estimation is supported for single-stream counting.
 *
 * Ported from the reference Java implementation
 * (github.com/dynatrace-research/exaloglog-paper).
 */
#include "common.h"
#include "hash.h"
#include "mle.h"

namespace sketch {
inline namespace ell {

namespace detail {

// Hurwitz zeta function via Euler-Maclaurin summation (s > 1, a > 0).
// Accurate to ~1e-12 for the parameter ranges used here; used for the
// ML bias-correction constants and the theoretical relative error.
inline double hurwitz_zeta(double s, double a) {
    static const double B2 = 1. / 6., B4 = -1. / 30., B6 = 1. / 42., B8 = -1. / 30.;
    const uint32_t N = 32;
    double sum = 0.;
    for(uint32_t k = 0; k < N; ++k) sum += std::pow(double(k) + a, -s);
    const double x = double(N) + a;
    double r = std::pow(x, 1. - s) / (s - 1.) + 0.5 * std::pow(x, -s);
    const double xp = std::pow(x, -s - 1.);
    r += (B2 / 2.) * s * xp;
    r += (B4 / 24.) * s * (s + 1.) * (s + 2.) * xp / (x * x);
    r += (B6 / 720.) * s * (s + 1.) * (s + 2.) * (s + 3.) * (s + 4.) * xp / std::pow(x, 4.);
    r += (B8 / 40320.) * s * (s + 1.) * (s + 2.) * (s + 3.) * (s + 4.) * (s + 5.) * (s + 6.) * xp / std::pow(x, 6.);
    return sum + r;
}

// First-order ML bias correction constant c(t, d), paper eq. (4),
// with b = 2^(2^-t):  c = ln(b) (1 + 2 b^-d/(b-1)) zeta(3, 1 + b^-d/(b-1))
//                     / zeta(2, 1 + b^-d/(b-1))^2
inline double ell_ml_bias_correction(uint32_t t, uint32_t d) {
    const double b = std::pow(2., std::ldexp(1., -int(t)));
    const double lnb = std::log(b);
    const double a0 = 1. + std::pow(b, -double(d)) / (b - 1.);
    return lnb * (1. + 2. * std::pow(b, -double(d)) / (b - 1.))
         * hurwitz_zeta(3., a0) / (hurwitz_zeta(2., a0) * hurwitz_zeta(2., a0));
}

} // namespace detail

template<typename HashStruct=WangHash>
class exaloglogbase_t {
    // m = 2^p registers, each 6+t+d bits, bit-packed into state_.
    uint8_t                                        t_;
    uint8_t                                        d_;
    uint32_t                                      np_;
    uint8_t                                      rb_; // register size in bits
    uint64_t                                   mask_; // register value mask
    std::vector<uint8_t, common::Allocator<uint8_t>> state_;
    HashStruct                                       hf_;
    static constexpr uint32_t MAGIC = 0x454c4c31; // "ELL1"

    static constexpr uint32_t min_p() {return 2;}
    static constexpr uint32_t max_t() {return 24;} // t + p <= 26 (reference)
public:
    using final_type = exaloglogbase_t<HashStruct>;
    using HashType = HashStruct;

    static constexpr uint32_t regbits(uint32_t t, uint32_t d) {return 6 + t + d;}
    static uint32_t max_d(uint32_t t) {return 64 - 6 - t;}
    static uint32_t max_p(uint32_t t) {return 26 - t;}
    static double base_b(uint32_t t) {return std::pow(2., std::ldexp(1., -int(t)));}

    // Theoretical asymptotic MVP (uncompressed, efficient estimator).
    static double theoretical_mvp(uint32_t t, uint32_t d) {
        const double b = base_b(t);
        const double lnb = std::log(b);
        return regbits(t, d) * lnb / detail::hurwitz_zeta(2., 1. + std::pow(b, -double(d)) / (b - 1.));
    }

    template<typename... Args>
    explicit exaloglogbase_t(uint8_t t, uint8_t d, uint32_t np, Args &&... args):
        t_(t), d_(d), np_(np), hf_(std::forward<Args>(args)...) {
        if(t > max_t()) throw std::invalid_argument(std::string("[exaloglog] t must be in [0, ") + std::to_string(max_t()) + "]");
        if(d > max_d(t)) throw std::invalid_argument(std::string("[exaloglog] d must be in [0, ") + std::to_string(max_d(t)) + "] for t=" + std::to_string(t));
        if(np < min_p() || np > max_p(t)) throw std::invalid_argument(std::string("[exaloglog] p must be in [2, ") + std::to_string(max_p(t)) + "] for t=" + std::to_string(t));
        rb_ = regbits(t_, d_);
        mask_ = (rb_ == 64) ? ~uint64_t(0): ((uint64_t(1) << rb_) - 1);
        // +16 bytes of padding so that register access can always go
        // through (unaligned, memcpy-based) 8-byte accesses, including the
        // two-word access needed when a register wider than 57 bits crosses
        // an 8-byte boundary.
        state_.assign(((size_t(m()) * rb_) >> 3) + 16, 0);
    }
    // Paper-recommended default: t=2, d=20 (28-bit packed registers, MVP 3.67).
    explicit exaloglogbase_t(uint32_t np): exaloglogbase_t(2, 20, np) {}
    exaloglogbase_t() = delete;
    // Non-template: a variadic file constructor would hijack calls whose
    // first argument is the literal 0 (a null pointer constant is an exact
    // match for gzFile and would win overload resolution).
    exaloglogbase_t(const std::string &path): exaloglogbase_t(uint8_t(2), uint8_t(20), min_p()) {
        read(path);
    }
    exaloglogbase_t(gzFile fp): exaloglogbase_t(uint8_t(2), uint8_t(20), min_p()) {
        read(fp);
    }

    uint8_t t() const {return t_;}
    uint8_t d() const {return d_;}
    uint32_t p() const {return np_;}
    uint64_t m() const {return uint64_t(1) << np_;}
    uint32_t register_bits() const {return rb_;}
    // Packed state without padding; size in bytes.
    size_t packed_size() const {return (size_t(m()) * rb_ + 7) >> 3;}
    const uint8_t *data() const {return state_.data();}
    uint8_t *mutable_data() {return state_.data();}
    double relative_error() const {
        return std::sqrt(theoretical_mvp(t_, d_) / (double(m()) * regbits(t_, d_)));
    }
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this), state_.size());
    }
    void clear() noexcept {
        std::memset(state_.data(), 0, state_.size());
    }
    bool is_empty() const {
        for(size_t i = 0; i < packed_size(); ++i) if(state_[i] != 0) return false;
        return true;
    }

    INLINE uint64_t getreg(size_t i) const {
        const size_t bitpos = size_t(i) * rb_;
        const uint8_t *const ptr = state_.data() + (bitpos >> 3);
        const uint32_t sh = bitpos & 7;
        if(sh + rb_ <= 64) { // always true for rb_ <= 57
            uint64_t v;
            std::memcpy(&v, ptr, sizeof(v));
            return (v >> sh) & mask_;
        }
        // register crosses an 8-byte boundary (only rb_ in (57, 64) with
        // sh > 0): two-word access, covered by the +16 bytes of tail padding
        uint64_t lo, hi;
        std::memcpy(&lo, ptr, sizeof(lo));
        std::memcpy(&hi, ptr + 8, sizeof(hi));
        return ((lo >> sh) | (hi << (64 - sh))) & mask_;
    }
    INLINE void setreg(size_t i, uint64_t v) {
        const size_t bitpos = size_t(i) * rb_;
        uint8_t *const ptr = state_.data() + (bitpos >> 3);
        const uint32_t sh = bitpos & 7;
        v &= mask_;
        if(sh + rb_ <= 64) {
            uint64_t old;
            std::memcpy(&old, ptr, sizeof(old));
            const uint64_t msk = mask_ << sh;
            old = (old & ~msk) | (v << sh);
            std::memcpy(ptr, &old, sizeof(old));
            return;
        }
        uint64_t lo, hi;
        std::memcpy(&lo, ptr, sizeof(lo));
        std::memcpy(&hi, ptr + 8, sizeof(hi));
        const uint64_t lo_keep = ~uint64_t(0) << sh; // writable bits [sh, 64)
        const unsigned nbits_hi = sh + rb_ - 64;     // writable bits [0, nbits_hi) of the next word
        const uint64_t hi_keep = (uint64_t(1) << nbits_hi) - 1;
        lo = (lo & ~lo_keep) | ((v << sh) & lo_keep);
        hi = (hi & ~hi_keep) | ((v >> (64 - sh)) & hi_keep);
        std::memcpy(ptr, &lo, sizeof(lo));
        std::memcpy(ptr + 8, &hi, sizeof(hi));
    }

    INLINE uint64_t hash(uint64_t val) const {return hf_(val);}
    INLINE void addh(uint64_t element) noexcept {add(hf_(element));}
    INLINE void addh(uint64_t element, mle::detail::MartingaleEstimator &martingale) noexcept {add(hf_(element), martingale);}
    void add(const uint64_t *vals, size_t n) noexcept {
        for(size_t i = 0; i < n; ++i) add(vals[i]);
    }
    template<typename It>
    void add(It b, It e) noexcept {add(&*b, std::distance(b, e));}

    // Insertion, Algorithm 2 of the paper (constant time).
    INLINE void add(uint64_t hashval) noexcept {
        const uint64_t mask = ((uint64_t(1) << t_) << np_) - 1;
        const uint32_t idx = uint32_t((hashval & mask) >> t_);
        const int nlz = clz(hashval | mask); // in {0, ..., 64-p-t}
        const uint64_t k = (uint64_t(nlz) << t_) + (hashval & ((uint64_t(1) << t_) - 1)) + 1;
        const uint64_t r_old = getreg(idx);
        const uint64_t u = r_old >> d_;
        const int64_t delta = int64_t(k) - int64_t(u);
        if(delta > 0) {
            uint64_t r_new = k << d_;
            if(delta <= d_) r_new |= ((uint64_t(1) << d_) | (r_old & ((uint64_t(1) << d_) - 1))) >> delta;
            setreg(idx, r_new);
        } else if(delta < 0 && int64_t(d_) + delta >= 0) {
            setreg(idx, r_old | (uint64_t(1) << (d_ + delta)));
        }
    }

    // Insertion with martingale (HIP) estimation support.
    INLINE void add(uint64_t hashval, mle::detail::MartingaleEstimator &martingale) noexcept {
        const uint64_t mask = ((uint64_t(1) << t_) << np_) - 1;
        const uint32_t idx = uint32_t((hashval & mask) >> t_);
        const int nlz = clz(hashval | mask);
        const uint64_t k = (uint64_t(nlz) << t_) + (hashval & ((uint64_t(1) << t_) - 1)) + 1;
        const uint64_t r_old = getreg(idx);
        const uint64_t u = r_old >> d_;
        const int64_t delta = int64_t(k) - int64_t(u);
        if(delta > 0) {
            uint64_t r_new = k << d_;
            if(delta <= d_) r_new |= ((uint64_t(1) << d_) | (r_old & ((uint64_t(1) << d_) - 1))) >> delta;
            setreg(idx, r_new);
            martingale.state_changed(mle::detail::u64_as_double(register_change_probability_scaled(r_old) -
                                                                register_change_probability_scaled(r_new)) * 0x1p-64);
        } else if(delta < 0 && int64_t(d_) + delta >= 0) {
            const uint64_t r_new = r_old | (uint64_t(1) << (d_ + delta));
            if(r_new != r_old) {
                setreg(idx, r_new);
                const int q = 63 - t_ - np_;
                martingale.state_changed(std::ldexp(1., std::max(q - nlz, 0) - 64));
            }
        }
    }

    // Register change probability scaled by 2^64 (reference:
    // ExaLogLog.getRegisterChangeProbabilityScaled == contribute(r, null, ...)).
    uint64_t register_change_probability_scaled(uint64_t r) const {
        return contribute(r, nullptr);
    }

    // Probability that a new distinct element changes the sketch state.
    double state_change_probability() const {
        const uint64_t first = register_change_probability_scaled(getreg(0));
        uint64_t sum = first;
        for(uint64_t i = 1; i < m(); ++i) sum += register_change_probability_scaled(getreg(i));
        // sum wraps to zero in two cases: all registers saturated (first == 0)
        // or the empty state, where the scaled probabilities sum to exactly 2^64.
        if(sum != 0 || first == 0) return mle::detail::u64_as_double(sum) * 0x1p-64;
        return 1.;
    }

    // Bias-reduced maximum likelihood estimate.
    double estimate() const {
        uint64_t agg = 0;
        uint32_t b[64]{0};
        for(uint64_t i = 0; i < m(); ++i) agg += contribute(getreg(i), b);
        if(agg == 0) return b[63 - t_ - np_] == 0 ? 0.: std::numeric_limits<double>::infinity();
        const double factor = std::ldexp(double(m()), t_ + 1);
        const double a = mle::detail::u64_as_double(agg) * 0x1p-64 * factor;
        return factor * mle::detail::solve_maximum_likelihood_equation(a, b, 63 - np_ - t_, 0.)
             / (1. + detail::ell_ml_bias_correction(t_, d_) / double(m()));
    }
    double report() const {return estimate();}
    double creport() const {return estimate();}
    double cardinality_estimate() const {return estimate();}

    std::string desc_string() const {
        return std::string("ExaLogLog(t=") + std::to_string(t_) + ", d=" + std::to_string(d_)
             + ", p=" + std::to_string(np_) + ", m=" + std::to_string(m())
             + ", relerr=" + std::to_string(relative_error()) + ")";
    }
    std::string to_string() const {return desc_string() + ", estimate=" + std::to_string(estimate());}

    // Reference: ExaLogLog.contribute -- coefficient computation for the ML
    // equation (optimized version of Algorithm 3 in the paper).
    // b may be null, in which case only the scaled "a" contribution is
    // returned (register change probability scaled by 2^64).
    uint64_t contribute(uint64_t r, uint32_t *b) const {
        const uint32_t t = t_, d = d_, p = np_;
        const uint32_t u = uint32_t(r >> d);
        if(u == 0) return uint64_t(1) << (64 - p);
        const int q = 63 - t - p;
        const uint32_t j = (u - 1) >> t;
        int i = int(std::min(q, int(j)));
        const uint64_t r_inv = ~r;
        const int num_bits = int(u - 1) - (i << t);
        uint64_t mask = ~uint64_t(0) << std::max(0, int(d) - num_bits);
        const uint64_t mask2 = mask & ((uint64_t(1) << d) - 1);
        uint64_t a = ((((uint64_t(i) + 2) << t) - u + uint64_t(popcount(r_inv & mask2)))) << (q - i);
        if(b) b[i] += 1 + popcount(r & mask2);
        if(t <= 5) {
            const int shift = 1 << t;
            // Java '>>' on long is an arithmetic (sign-extending) shift; the
            // sign extension is significant for the bits that ~r exposes.
            mask ^= uint64_t(int64_t(mask) >> shift);
            while(i > 0 && mask != 0) {
                i -= 1;
                a += uint64_t(popcount(mask & r_inv)) << (q - i);
                if(b) b[i] += popcount(mask & r);
                mask >>= shift;
            }
        } else if(i > 0) {
            mask = ~mask;
            i -= 1;
            a += uint64_t(popcount(mask & r_inv)) << (q - i);
            if(b) b[i] += popcount(mask & r);
        }
        return a;
    }

private:
    static uint64_t shift_right(uint64_t s, int64_t delta) {
        return delta < 64 ? (s >> delta): 0;
    }
    static uint64_t merge_register(uint64_t r1, uint64_t r2, uint32_t d) {
        const uint64_t u1 = r1 >> d, u2 = r2 >> d;
        if(u1 > u2 && u2 > 0) {
            const uint64_t x = uint64_t(1) << d;
            return r1 | shift_right(x | (r2 & (x - 1)), int64_t(u1 - u2));
        }
        if(u2 > u1 && u1 > 0) {
            const uint64_t x = uint64_t(1) << d;
            return r2 | shift_right(x | (r1 & (x - 1)), int64_t(u2 - u1));
        }
        return r1 | r2;
    }
    static uint64_t compute_downsize_threshold_u(uint32_t t, uint32_t from_p) {
        return ((uint64_t(64 - t - from_p)) << t) + 1;
    }
    static uint64_t downsize_register(uint64_t r, uint32_t t, uint32_t from_d, uint32_t to_d,
                                      uint32_t from_p, uint32_t to_p, uint64_t sub_idx,
                                      uint64_t downsize_threshold_u) {
        const uint64_t u = r >> from_d;
        r >>= (from_d - to_d);
        if(u >= downsize_threshold_u) {
            const int64_t shift = int64_t((from_p - to_p) - (63 - clz(sub_idx))) << t;
            if(shift > 0) {
                const int64_t num_bits_to_shift = int64_t(to_d) + downsize_threshold_u - u;
                if(num_bits_to_shift > 0) {
                    const uint64_t msk = ~uint64_t(0) << num_bits_to_shift;
                    r = (msk & r) | shift_right((r & ~msk), shift);
                }
                r += uint64_t(shift) << to_d;
            }
        }
        return r;
    }
public:
    // Merge. The other sketch must have the same t, d at least as large as
    // this sketch's d, and precision at least as large as this one's
    // (higher-precision sketches are downsized, which is lossless in the
    // same sense as UltraLogLog's reducibility).
    exaloglogbase_t &add(const exaloglogbase_t &other) {
        if(other.t_ != t_) throw std::invalid_argument("[exaloglog] cannot merge sketches with different t");
        if(other.d_ < d_) throw std::invalid_argument("[exaloglog] cannot merge a sketch with smaller d");
        if(other.np_ < np_) throw std::invalid_argument("[exaloglog] cannot merge a sketch with smaller precision");
        if(other.d_ == d_ && other.np_ == np_) {
            for(uint64_t i = 0; i < m(); ++i) {
                const uint64_t merged = merge_register(getreg(i), other.getreg(i), d_);
                if(merged != getreg(i)) setreg(i, merged);
            }
        } else {
            const uint64_t max_sub_index = uint64_t(1) << (other.np_ - np_);
            const uint64_t threshold = compute_downsize_threshold_u(t_, other.np_);
            for(uint64_t i = 0; i < m(); ++i) {
                uint64_t merged = downsize_register(other.getreg(i), t_, other.d_, d_, other.np_, np_, 0, threshold);
                for(uint64_t k = 1; k < max_sub_index; ++k) {
                    const uint64_t other_r = downsize_register(other.getreg(i + (k << np_)), t_, other.d_, d_, other.np_, np_, k, threshold);
                    merged = merge_register(merged, other_r, d_);
                }
                if(merged != 0) {
                    const uint64_t this_r = getreg(i);
                    const uint64_t fin = merge_register(merged, this_r, d_);
                    if(fin != this_r) setreg(i, fin);
                }
            }
        }
        return *this;
    }
    exaloglogbase_t &operator+=(const exaloglogbase_t &other) {return add(other);}
    exaloglogbase_t operator+(const exaloglogbase_t &other) const {
        exaloglogbase_t ret(*this);
        ret.add(other);
        return ret;
    }
    exaloglogbase_t downsize(uint8_t d, uint32_t np) const {
        if(np < min_p() || np > max_p(t_)) throw std::invalid_argument(std::string("[exaloglog] invalid precision parameter ") + std::to_string(np));
        if(d > max_d(t_)) throw std::invalid_argument(std::string("[exaloglog] invalid d parameter ") + std::to_string(d));
        if(np >= np_ && d >= d_) return *this;
        return exaloglogbase_t(t_, d, np).add(*this);
    }

    bool operator==(const exaloglogbase_t &o) const {
        if(t_ != o.t_ || d_ != o.d_ || np_ != o.np_) return false;
        for(uint64_t i = 0; i < m(); ++i) if(getreg(i) != o.getreg(i)) return false;
        return true;
    }
    bool operator!=(const exaloglogbase_t &o) const {return !(*this == o);}

    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[4]{MAGIC, (uint32_t(t_) << 16) | d_, np_, 0};
        CW(fp, header, sizeof(header));
        CW(fp, state_.data(), packed_size());
#undef CW
    }
    void write(const char *path) const {
        gzFile fp(gzopen(path, "wb"));
        if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at '") + path + "' for writing");
        write(fp);
        gzclose(fp);
    }
    void write(const std::string &path) const {write(path.data());}
    void read(gzFile fp) {
#define CR(fp, dst, len) \
    do {\
        if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) {\
            throw ZlibError(std::string("[E:") + __FILE__ + ':' + std::to_string(__LINE__) + ':' + __PRETTY_FUNCTION__ + "] Error reading from file");\
        }\
    } while(0)
        uint32_t header[4];
        CR(fp, header, sizeof(header));
        if(header[0] != MAGIC) throw std::runtime_error("[exaloglog] bad magic: not an ExaLogLog sketch");
        const uint8_t t = header[1] >> 16, d = header[1] & 0xFF;
        const uint32_t np = header[2];
        if(t > max_t() || d > max_d(t) || np < min_p() || np > max_p(t))
            throw std::runtime_error("[exaloglog] invalid stored parameters");
        if(t != t_ || d != d_ || np != np_) {
            // rebuild state for the stored parameters
            t_ = t; d_ = d; np_ = np;
            rb_ = regbits(t_, d_);
            mask_ = (rb_ == 64) ? ~uint64_t(0): ((uint64_t(1) << rb_) - 1);
            state_.assign(((size_t(m()) * rb_) >> 3) + 16, 0);
        }
        CR(fp, state_.data(), packed_size());
        std::memset(state_.data() + packed_size(), 0, state_.size() - packed_size());
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}
};
using exaloglog_t = exaloglogbase_t<>;

} // namespace ell
} // namespace sketch

#endif // SKETCH_EXALOGLOG_H__
