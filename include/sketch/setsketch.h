#ifndef D2_SETSKETCH_H___H__
#define D2_SETSKETCH_H___H__
#include <stdexcept>
#include <cassert>
#include "aesctr/wy.h"
#include <queue>
#include "sketch/div.h"
#include <unordered_map>
#include <memory>
#include "sketch/fy.h"
#include "sketch/count_eq.h"
#include "sketch/macros.h"
#include "sketch/hash.h"
#include "sketch/flog.h"
#include "sketch/kahan.h"
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {

namespace setsketch {

namespace detail {
    struct Deleter {
        template<typename T>
        void operator()(const T *x) const {std::free(const_cast<T *>(x));}
    };
template <class F, class T>
std::tuple<T, T, uint64_t> brent_find_minima(const F &f, T min, T max, int bits=std::numeric_limits<T>::digits, uint64_t max_iter=std::numeric_limits<uint64_t>::max()) noexcept
{
   T x, w, v, u, delta, delta2, fu, fv, fw, fx, mid, fract1, fract2;
   const T tolerance = static_cast<T>(std::ldexp(1.0, 1-bits));
   static constexpr T golden = 0.3819660;  // golden ratio, don't need too much precision here!
   x = w = v = max;
   fw = fv = fx = f(x);
   delta2 = delta = 0;
   uint64_t count = max_iter;
   do {
      mid = (min + max) / 2;
      fract1 = tolerance * std::abs(x) + tolerance / 4; fract2 = 2 * fract1;
      if(std::abs(x - mid) <= (fract2 - (max - min) / 2)) break;
      if(std::abs(delta2) > fract1) {
         T r = (x - w) * (fx - fv);
         T q = (x - v) * (fx - fw);
         T p = (x - v) * q - (x - w) * r;
         q = 2 * (q - r);
         if(q > 0) p = -p;
         else      q = -q;
         T td = delta2;
         delta2 = delta;
         if((std::abs(p) >= std::abs(q * td / 2)) || (p <= q * (min - x)) || (p >= q * (max - x)))
         {
            delta2 = (x >= mid) ? min - x : max - x;
            delta = golden * delta2;
         } else {
            delta = p / q; u = x + delta;
            if(((u - min) < fract2) || ((max- u) < fract2))
               delta = (mid - x) < 0 ? (T)-std::abs(fract1) : (T)std::abs(fract1);
         }
      } else {
         delta2 = (x >= mid) ? min - x : max - x;
         delta = golden * delta2;
      }
      u = (std::abs(delta) >= fract1) ? T(x + delta) : (delta > 0 ? T(x + std::abs(fract1)) : T(x - std::abs(fract1)));
      fu = f(u);
      if(fu <= fx) {
         if(u >= x) min = x; else max = x;
         v = w;w = x; x = u; fv = fw; fw = fx; fx = fu;
      } else {
         // Oh dear, point u is worse than what we have already,
         // even so it *must* be better than one of our endpoints:
         if(u < x) min = u;
         else      max = u;
         if((fu <= fw) || (w == x))
            v = w, w = u, fv = fw, fw = fu;
         else if((fu <= fv) || (v == x) || (v == w))
            v = u, fv = fu;
      }
   } while(--count);
   return std::make_tuple(x, fx, max_iter - count);
}

static INLINE std::pair<long double, long double> optimal_parameters(long double maxreg, long double minreg, long double q) {
    long double b = std::exp(std::log((long double)maxreg / (long double)minreg) / (long double)q);
    return {b, (long double)maxreg / b};
}

}

template<typename FT>
static inline FT jmle_simple(const uint64_t lhgt, const uint64_t rhgt, const size_t m, const FT lhest, const FT rhest, FT base) {
    if(!lhest && !rhest) return FT(0.);
    const uint64_t neq = m - (lhgt + rhgt);
    const FT sumest = lhest + rhest;
    const long double bi = 1.L / base;
    const long double lbase = std::log(static_cast<long double>(base)), lbi = 1. / lbase;
    const FT z = (1.L - bi) / (sumest);
    auto func = [neq,lhgt,rhgt,lbi,z,rhest,lhest](auto jaccard) {
        FT lhs = neq || lhgt ? FT(lbi * std::log1p((rhest * jaccard - lhest) * z)): FT(0);
        FT rhs = neq || rhgt ? FT(lbi * std::log1p((lhest * jaccard - rhest) * z)): FT(0);
        FT ret = 0;
        if(neq)  ret += neq * std::log1p(lhs + rhs);
        if(lhgt) ret += lhgt * std::log(-lhs);
        if(rhgt) ret += rhgt * std::log(-rhs);
        if(std::isnan(ret)) return std::numeric_limits<FT>::max();
        return -ret;
    };
    return std::get<0>(detail::brent_find_minima(func, FT(0), std::min(lhest, rhest) / std::max(lhest, rhest), 24));
}


    static constexpr double INVMUL64 =
#if __cplusplus >= 201703L
    0x1p-64;
#else
    5.42101086242752217e-20;
#endif

// Implementations of set sketch

template<typename FT>
class mvt_t {
    FT mv_;
    FT *data_ = nullptr;
    size_t m_;
public:
    mvt_t(size_t m, FT mv = std::numeric_limits<FT>::max()): mv_(mv), m_(m) {}

    FT mv() const {return mv_;}
    FT *data() {return data_;}
    const FT *data() const {return data_;}
    size_t getm() const {return m_;}
    size_t nelem() const {return 2 * m_ - 1;}
    FT operator[](size_t i) const {return data_[i];}
    void assign(FT *vals, size_t nvals, FT mv) {
        mv_ = mv;
        assign(vals, nvals);
    }
    void assign(FT *vals, size_t nvals) {
        data_ = vals; m_ = nvals;
        std::fill(data_, data_ + nelem(), mv_);
    }
    FT max() const {
        return data_[nelem() - 1];
    }
    FT klow() const {
        return max();
    }

    bool update(size_t index, FT x) {
        const auto sz = nelem();
        if(x < data_[index]) {
            for(;;) {
                data_[index] = x;
                if((index = m_ + (index >> 1)) >= sz) break;
                const size_t lhi = (index - m_) << 1, rhi = lhi + 1;
                x = std::max(data_[lhi], data_[rhi]);
                if(x >= data_[index]) break;
            }
            assert(max() == *std::max_element(data_, data_ + m_));
            return true;
        }
        return false;
    }
};

template<typename ResT>
struct minvt_t {
    static constexpr ResT minv_ = 0;
    ResT *data_ = nullptr;
    size_t m_;
    long double b_ = -1., explim_ = -1.;
    minvt_t(size_t m): m_(m) {}

    double explim() const {return explim_;}
    ResT *data() {return data_;}
    const ResT *data() const {return data_;}
    size_t getm() const {return m_;}
    ResT operator[](size_t i) const {return data_[i];}
    void assign(ResT *vals, size_t nvals, double b) {
        data_ = vals; m_ = nvals; b_ = b;
        std::fill(data_, data_ + (m_ << 1) - 1, minv_);
        explim_ = std::pow(b_, -min());
    }
    typename std::ptrdiff_t min() const {
        return data_[(m_ << 1) - 2];
    }
    typename std::ptrdiff_t klow() const {
        return min();
    }
    typename std::ptrdiff_t max() const {return *std::max_element(data_, &data_[(m_ << 1) - 1]);}

    bool update(size_t index, ResT x) {
        const auto sz = (m_ << 1) - 1;
        if(x > data_[index]) {
            for(;;) {
                data_[index] = x;
                if((index = m_ + (index >> 1)) >= sz) break;
                const size_t lhi = (index - m_) << 1, rhi = lhi + 1;
                x = std::min(data_[lhi], data_[rhi]);
                if(x <= data_[index]) break;
            }
            explim_ = std::pow(b_, -min());
            assert(min() == *std::min_element(data_, data_ + m_));
            return true;
        }
        return false;
    }
};

template<typename ResT>
struct LowKHelper {
    ResT *vals_;
    uint64_t natval_, nvals_;
    double b_ = -1.;
    double explim_;
    int klow_ = 0;
    LowKHelper(size_t m): nvals_(m) {}
    void assign(ResT *vals, size_t nvals, double b) {
        vals_ = vals; nvals_ = nvals;
        b_ = b;
        reset();
    }
    int klow() const {return klow_;}
    auto max() const {return *std::max_element(vals_, vals_ + nvals_);}
    double explim() const {return explim_;}
    void reset() {
        klow_ =  *std::min_element(vals_, vals_ + nvals_);
        size_t i;
        for(i = natval_ = 0; i < nvals_; ++i) natval_ += (vals_[i] == klow_);
        explim_ = std::pow(b_, -klow_);
    }
    bool update(size_t idx, ResT k) {
        if(k > vals_[idx]) {
            auto oldv = vals_[idx];
            vals_[idx] = k;
            remove(oldv);
            return true;
        }
        return false;
    }
    void remove(int kval) {
        if(kval == klow_) {
            if(--natval_ == 0) reset();
        }
    }
};

#if __AVX2__
INLINE float broadcast_reduce_sum(__m256 x) {
    const __m256 permHalves = _mm256_permute2f128_ps(x, x, 1);
    const __m256 m0 = _mm256_add_ps(permHalves, x);
    const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1 = _mm256_add_ps(m0, perm0);
    const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2 = _mm256_add_ps(perm1, m1);
    return m2[0];
}
INLINE double broadcast_reduce_sum(__m256d x) {
    __m256d m1 = _mm256_add_pd(x, _mm256_permute2f128_pd(x, x, 1));
    return _mm256_add_pd(m1, _mm256_permute_pd(m1, 5))[0];
}
#endif

static inline long double g_b(long double b, long double arg) {
    return (1.L - std::pow(b, -arg)) / (1.L - 1.L / b);
}


template<typename ResT, typename FT=double> class SetSketch; // Forward


template<typename FT=double>
class CSetSketch {
    // This uses Kahan summation for floating-point values by default
    // std::fma is expected to be accurate enough for long doubles.
    static_assert(std::is_floating_point<FT>::value, "Must float");
    // SetSketch 1
protected:
    size_t m_; // Number of registers
    std::unique_ptr<FT[], detail::Deleter> data_;
    fy::LazyShuffler ls_;
    mvt_t<FT> mvt_;
    std::vector<uint64_t> ids_;
    std::vector<uint32_t> idcounts_;
    uint64_t total_updates_ = 0;
    mutable double mycard_ = -1.;
    static FT *allocate(size_t n) {
        n = (n << 1) - 1;
        FT *ret = nullptr;
        static constexpr size_t ALN =
#if __AVX512F__
            64;
#elif __AVX2__
            32;
#else
            16;
#endif
        if(posix_memalign((void **)&ret, ALN, n * sizeof(FT))) throw std::bad_alloc();
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / static_cast<FT>(m_ - idx);
    }
public:
    const FT *data() const {return data_.get();}
    FT *data() {return data_.get();}
    CSetSketch(size_t m, bool track_ids=false, bool track_counts=false, FT maxv=std::numeric_limits<FT>::max()): m_(m), ls_(m_), mvt_(m_) {
        if(m > 0xFFFFFFFFull) {
            throw std::invalid_argument("CSetSketch's maximum sketch size is 2^32/0xFFFFFFFFu/4294967295.");
        }
        data_.reset(allocate(m_));
        mvt_.assign(data_.get(), m_, maxv);
        if(track_ids || track_counts) ids_.resize(m_);
        if(track_counts)         idcounts_.resize(m_);
        //generate_betas();
    }
    CSetSketch(const CSetSketch &o): m_(o.m_), data_(allocate(o.m_)), ls_(m_), mvt_(m_, o.mvt_.mv()), ids_(o.ids_), idcounts_(o.idcounts_) {
        mvt_.assign(data_.get(), m_, o.mvt_.mv());
        std::copy(o.data_.get(), &o.data_[2 * m_ - 1], data_.get());
        //generate_betas();
    }
    template<typename ResT=uint16_t>
    SetSketch<ResT, FT> to_setsketch(double b, double a, int64_t q=std::numeric_limits<ResT>::max() - 1) const {
        SetSketch<ResT, FT> ret(m_, b, a, q, ids_.size());
        const double logbinv = 1. / std::log1p(b - 1.);
        for(size_t i = 0; i < m_; ++i) {
            ret.lowkh().update(i, std::max(int64_t(0), std::min(int64_t(q) + 1, static_cast<int64_t>((1. - std::log(data_[i] / a) * logbinv)))));
        }
        return ret;
    }
    CSetSketch &operator=(const CSetSketch &o) {
        if(size() != o.size()) {
            if(m_ < o.m_) data_.reset(allocate(o.m_));
            m_ = o.m_;
            ls_.resize(m_);
            //generate_betas();
        }
        mvt_.assign(data_.get(), m_, o.mvt_.mv());
        std::copy(o.data(), o.data() + (2 * m_ - 1), data());
        if(o.ids_.size()) {
            ids_ = o.ids_;
            if(o.idcounts_.size()) idcounts_ = o.idcounts_;
        }
        total_updates_ = o.total_updates_;
        return *this;
    }
    CSetSketch(std::FILE *fp): ls_(1), mvt_(1) {read(fp);}
    CSetSketch(gzFile fp): ls_(1), mvt_(1) {read(fp);}
    CSetSketch(const std::string &s): ls_(1), mvt_(1) {
        read(s);
    }
    CSetSketch<FT> clone_like() const {
        return CSetSketch(m_, !ids().empty(), !idcounts().empty());
    }
    FT min() const {return *std::min_element(data(), data() + m_);}
    FT max() const {return mvt_.max();}
    size_t size() const {return m_;}
    FT &operator[](size_t i) {return data_[i];}
    const FT &operator[](size_t i) const {return data_[i];}
    void addh(uint64_t id) {update(id);}
    void add(uint64_t id) {update(id);}
    size_t total_updates() const {return total_updates_;}
    template<typename OFT, typename=typename std::enable_if<std::is_arithmetic<OFT>::value>::type>
    void update(const uint64_t id, OFT) {update(id);}
    // If a weight is passed, ignore it
    void update(const uint64_t id) {
        using fastlog::flog;
        FT kahan_carry = 0;
        mycard_ = -1.;
        ++total_updates_;
        uint64_t hid = id;
        uint64_t rv = sketch::hash::CEHasher()(id ^ uint64_t(0xb2069fc679a8da0buLL));

        FT ev;
        FT mv = max();
        CONST_IF(sizeof(FT) > 8) {
            auto lrv = __uint128_t(rv) << 64;
            const FT bv = -1. / m_;
            lrv |= wy::wyhash64_stateless(&rv);
            FT tv = static_cast<long double>((lrv >> 32) * 1.2621774483536188887e-29L);
            ev = bv * std::log(tv);
            if(ev > mv) return;
        } else {
            auto tv = rv * INVMUL64;
            const FT bv = -1. / m_;
            if(bv * flog(tv) * FT(.7) > mv) return;
            ev = bv * std::log(tv);
            if(ev > mv) return;
        }
        ls_.reset();
        ls_.seed(rv);
        uint64_t bi = 1;
        uint32_t idx;
        for(;;) {
            idx = ls_.step();
            if(mvt_.update(idx, ev)) {
                if(!ids_.empty()) {
                    ids_.operator[](idx) = id;
                    if(!idcounts_.empty()) idcounts_.operator[](idx) = 1;
                }
                mv = max();
            } else if(!idcounts_.empty()) {
                if(id == ids_.operator[](idx))
                    ++idcounts_.operator[](idx);
            }
            if(bi == m_) return;
            rv = wy::wyhash64_stateless(&hid);
            const FT bv = -getbeta(bi++);
            CONST_IF(sizeof(FT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                const FT increment = bv * std::log((lrv >> 32) * 1.2621774483536188887e-29L);
                if(kahan::update(ev, kahan_carry, increment) > mv) break;
            } else {
                const FT nv = rv * INVMUL64;
                if(bv * flog(nv) * FT(.7) + ev > mv || kahan::update(ev, kahan_carry, bv * std::log(nv)) > mv)
                    break;
            }
        }
    }
    bool operator==(const CSetSketch<FT> &o) const {
        return same_params(o) && std::equal(data(), data() + m_, o.data());
    }
    bool same_params(const CSetSketch<FT> &o) const {
        return m_ == o.m_
            && (ids().empty() == o.ids().empty())
            && (idcounts().empty() == o.idcounts().empty());
    }
    void merge(const CSetSketch<FT> &o) {
        if(!same_params(o)) throw std::runtime_error("Can't merge sets with differing parameters");
        if(ids().empty()) {
            std::transform(data(), data() + m_, o.data(), data(), [](auto x, auto y) {return std::min(x, y);});
        } else {
            for(size_t i = 0; i < size(); ++i) {
                if(!idcounts_.empty() && !ids_.empty() && ids_[i] == o.ids_[i]) {
                    idcounts_[i] += o.idcounts_[i];
                } else if(mvt_.update(i, o.data_[i])) {
                    if(!ids_.empty()) ids_[i] = o.ids_[i];
                    if(!idcounts_.empty()) idcounts_[i] = o.idcounts_[i];
                }
            }
        }
        total_updates_ += o.total_updates_;
        mycard_ = -1.;
    }
    CSetSketch &operator+=(const CSetSketch<FT> &o) {merge(o); return *this;}
    CSetSketch operator+(const CSetSketch<FT> &o) const {
        CSetSketch ret(*this);
        ret += o;
        return ret;
    }
    double jaccard_index(const CSetSketch<FT> &o) const {
        return shared_registers(o) / double(m_);
    }
    size_t shared_registers(const CSetSketch<FT> &o) const {
        CONST_IF(sizeof(FT) == 4) {
            return eq::count_eq((uint32_t *)data(), (uint32_t *)o.data(), m_);
        } else CONST_IF(sizeof(FT) == 8) {
            return eq::count_eq((uint64_t *)data(), (uint64_t *)o.data(), m_);
        } else CONST_IF(sizeof(FT) == 2) {
            return eq::count_eq((uint16_t *)data(), (uint16_t *)o.data(), m_);
        }
        auto optr = o.data();
        return std::accumulate(data(), data() + m_, size_t(0), [&optr](size_t nshared, FT x) {
            return nshared + (x == *optr++);
        });
    }
    void write(std::string s) const {
        gzFile fp = gzopen(s.data(), "w");
        if(!fp) throw ZlibError(std::string("Failed to open file ") + s + "for writing");
        write(fp);
        gzclose(fp);
    }
    void read(std::string s) {
        gzFile fp = gzopen(s.data(), "r");
        if(!fp) throw ZlibError(std::string("Failed to open file ") + s);
        read(fp);
        gzclose(fp);
    }
    void read(gzFile fp) {
        gzread(fp, &m_, sizeof(m_));
        FT mv;
        gzread(fp, &mv, sizeof(mv));
        data_.reset(allocate(m_));
        mvt_.assign(data_.get(), m_, mv);
        gzread(fp, (void *)data_.get(), m_ * sizeof(FT));
        for(size_t i = 0;i < m_; ++i) mvt_.update(i, data_[i]);
        ls_.resize(m_);
    }
    int checkwrite(std::FILE *fp, const void *ptr, size_t nb) const {
        auto ret = ::write(::fileno(fp), ptr, nb);
        if(size_t(ret) != nb) throw ZlibError("Failed to write setsketch to file");
        return ret;
    }
    int checkwrite(gzFile fp, const void *ptr, size_t nb) const {
        auto ret = gzwrite(fp, ptr, nb);
        if(size_t(ret) != nb) throw ZlibError("Failed to write setsketch to file");
        return ret;
    }
    void write(std::FILE *fp) const {
        checkwrite(fp, (const void *)&m_, sizeof(m_));
        FT m = mvt_.mv();
        checkwrite(fp, (const void *)&m, sizeof(m));
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(FT));
    }
    void write(gzFile fp) const {
        checkwrite(fp, (const void *)&m_, sizeof(m_));
        FT m = mvt_.mv();
        checkwrite(fp, (const void *)&m, sizeof(m));
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(FT));
    }
    void reset() {clear();}
    void clear() {
        mvt_.assign(data_.get(), m_, mvt_.mv());
        total_updates_ = 0;
        if(ids_.size()) {
            std::fill(ids_.begin(), ids_.end(), uint64_t(0));
            if(idcounts_.size()) std::fill(idcounts_.begin(), idcounts_.end(), uint32_t(0));
        }
        mycard_ = -1.;
    }
    const std::vector<uint64_t> &ids() const {return ids_;}
    const std::vector<uint32_t> &idcounts() const {return idcounts_;}
    double union_size(const CSetSketch<FT> &o) const {
        auto triple = alpha_beta_mu(o);
        return std::get<2>(triple);
    }
    auto alpha_beta(const CSetSketch<FT> &o) const {
        auto gtlt = eq::count_gtlt(data(), o.data(), m_);
        return std::pair<double, double>{double(gtlt.first) / m_, double(gtlt.second) / m_};
    }
    static constexpr double __union_card(double alph, double beta, double lhcard, double rhcard) {
        return std::max((lhcard + rhcard) / (2. - alph - beta), 0.);
    }
    double getcard() const {
        if(mycard_ < 0.)
            mycard_ = cardinality();
        return mycard_;
    }
    double intersection_size(const CSetSketch<FT> &o) const {
        auto triple = alpha_beta_mu(o);
        return std::max(1. - (std::get<0>(triple) + std::get<1>(triple)), 0.) * std::get<2>(triple);
    }
    std::tuple<double, double, double> alpha_beta_mu(const CSetSketch<FT> &o) const {
        const auto ab = alpha_beta(o);
        auto mycard = getcard(), ocard = o.getcard();
        if(ab.first + ab.second >= 1.) // They seem to be disjoint sets, use SetSketch (15)
            return {(mycard) / (mycard + ocard), ocard / (mycard + ocard), mycard + ocard};
        return {ab.first, ab.second, __union_card(ab.first, ab.second, mycard, ocard)};
    }

    double cardinality_estimate() const {return cardinality();}
    double cardinality() const {
        double s = 0.;
#if _OPENMP >= 201307L
        #pragma omp simd reduction(+:s)
#endif
        for(size_t i = 0; i < m_; ++i)
            s += data_[i];
        return m_ / s;
    }
    template<typename ResT=uint16_t>
    static std::pair<long double, long double> optimal_parameters(FT maxreg, FT minreg, long double q=std::numeric_limits<ResT>::max()) {
        if(maxreg < minreg) std::swap(maxreg, minreg);
        return detail::optimal_parameters(maxreg, minreg, q);
    }
    double containment_index(const CSetSketch<FT> &o) const {
        auto abm = alpha_beta_mu(o);
        auto lho = std::get<0>(abm);
        auto isf = std::max(1. - (lho + std::get<1>(abm)), 0.);
        return isf / (lho + isf);
    }
    template<typename IT=FT>
    std::vector<IT> to_sigs() const {
        std::vector<IT> ret(m_);
        if(std::is_integral<IT>::value) {
            using TmpT = std::conditional_t<(std::max(sizeof(IT), sizeof(FT)) <= 8), uint64_t, __uint128_t>;
            std::transform(data_.get(), data_.get() + m_, ret.begin(), [](auto x) {
                static_assert(sizeof(x) <= sizeof(TmpT), "Sanity check");
                TmpT t = 0;
                std::memcpy(&t, &x, sizeof(x));
                uint64_t ret = wy::wyhash64_stateless((uint64_t *)&t);
                if(sizeof(TmpT) >= 16) ret ^= wy::wyhash64_stateless((uint64_t *)&t + 1);
                return ret;
            });
        } else {
            std::copy(data_.get(), data_.get() + m_, ret.begin());
        }
        return ret;
    }
};
template<typename ResT, typename FT>
class SetSketch {
    static_assert(std::is_floating_point<FT>::value, "Must float");
    static_assert(std::is_integral<ResT>::value, "Must be integral");
    // Set sketch 1
    size_t m_; // Number of registers
    FT a_; // Exponential parameter
    FT b_; // Base
    FT ainv_;
    FT logbinv_;
    using QType = std::common_type_t<ResT, int64_t>;
    QType q_;
    std::unique_ptr<ResT[], detail::Deleter> data_;
    std::vector<uint64_t> ids_; // The IDs representing the sampled items.
                                // Only used if SetSketch is
    fy::LazyShuffler ls_;
    minvt_t<ResT> lowkh_;
    std::vector<FT> lbetas_; // Cache Beta values * 1. / a
    mutable double mycard_ = -1.;
    static ResT *allocate(size_t num_sigs) {
        const size_t n = (num_sigs << 1) - 1;
        ResT *ret = nullptr;
        static constexpr size_t ALN =
#if __AVX512F__
            64;
#elif __AVX2__
            32;
#else
            16;
#endif
#if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_ALIGNED_ALLOC)
        const size_t mem_needed = n * sizeof(ResT);
        const size_t mem_requested = mem_needed + (mem_needed % ALN ? ALN - mem_needed % ALN: 0);
        if((ret = static_cast<ResT *>(std::aligned_alloc(ALN, mem_requested))) == nullptr)
#else
        if(posix_memalign((void **)&ret, ALN, n * sizeof(ResT)))
#endif
        {
            std::fprintf(stderr, "[%s:%s:%d] Failed to allocate with nsigs = %zu, nalloc = %zu, sizef(ResT) == %zu, ALN = %zu\n", __PRETTY_FUNCTION__, __FILE__, __LINE__, num_sigs, n, sizeof(ResT), ALN);
            throw std::bad_alloc();
        }
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / (m_ - idx);
    }
public:
    FT ainv() const {return ainv_;}
    const ResT *data() const {return data_.get();}
    ResT *data() {return data_.get();}
    auto &lowkh() {return lowkh_;}
    const auto &lowkh() const {return lowkh_;}
    SetSketch(size_t m, FT b, FT a, QType q, bool track_ids = false): m_(m), a_(a), b_(b), ainv_(1./ a), logbinv_(1. / std::log1p(b_ - 1.)), q_(q), ls_(m_), lowkh_(m) {
        ResT *p = allocate(m_);
        data_.reset(p);
        std::fill(p, p + m_, static_cast<ResT>(0));
        lowkh_.assign(p, m_, b_);
        if(track_ids) ids_.resize(m_);
        lbetas_.resize(m_);
        for(size_t i = 0; i < m_; ++i) {
            lbetas_[i] = -ainv_ / (m_ - i);
        }
    }
    SetSketch(const SetSketch &o): m_(o.m_), a_(o.a_), b_(o.b_), ainv_(o.ainv_), logbinv_(o.logbinv_), q_(o.q_), ls_(m_), lowkh_(m_), lbetas_(o.lbetas_) {
        ResT *p = allocate(m_);
        data_.reset(p);
        lowkh_.assign(p, m_, b_);
        std::copy(o.data_.get(), &o.data_[2 * m_ - 1], p);
    }
    SetSketch(SetSketch &&o) = default;
    SetSketch(const std::string &s): ls_(1), lowkh_(1) {
        read(s);
    }
    size_t size() const {return m_;}
    double b() const {return b_;}
    double a() const {return a_;}
    ResT &operator[](size_t i) {return data_[i];}
    const ResT &operator[](size_t i) const {return data_[i];}
    int klow() const {return lowkh_.klow();}
    auto max() const {return lowkh_.max();}
    auto min() const {return lowkh_.min();}
    void addh(uint64_t id) {update(id);}
    void add(uint64_t id) {update(id);}
    void print() const {
        std::fprintf(stderr, "%zu = m, a %lg, b %lg, q %d\n", m_, double(a_), double(b_), int(q_));
    }
    auto explim() const {return lowkh_.explim();}
    template<typename OFT, typename=std::enable_if_t<std::is_arithmetic<OFT>::value>>
    INLINE void update(const uint64_t id, OFT) {update(id);}
    void update(const uint64_t id) {
        using GenFT = std::conditional_t<(sizeof(FT) <= 8), double, long double>;
        GenFT carry = 0.;
        mycard_ = -1.;
        uint64_t hid = id;
        size_t bi = 0;
        uint64_t rv = wy::wyhash64_stateless(&hid);
        GenFT ev = 0.;
        ls_.reset();
        ls_.seed(rv);
        for(;;) {
            const GenFT ba = lbetas_[bi];
            if(sizeof(GenFT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                kahan::update(ev, carry, GenFT(ba * std::log((lrv >> 32) * 1.2621774483536188887e-29L)));
            } else kahan::update(ev, carry, ba * std::log(rv * INVMUL64));
            if(ev > lowkh_.explim()) return;
            const QType k = std::max(QType(0), std::min(q_ + 1, static_cast<QType>((1. - std::log(ev) * logbinv_))));
            if(k <= klow()) return;
            auto idx = ls_.step();
            if(lowkh_.update(idx, k)) {
                if(!ids_.empty()) {
                    ids_[idx] = id;
                }
            }
            if(++bi == m_)
                return;
            rv = wy::wyhash64_stateless(&hid);
        }
    }
    bool operator==(const SetSketch<ResT, FT> &o) const {
        return same_params(o) && std::equal(data(), data() + m_, o.data());
    }
    bool same_params(const SetSketch<ResT,FT> &o) const {
        return std::tie(b_, a_, m_, q_) == std::tie(o.b_, o.a_, o.m_, o.q_);
    }
    double harmean(const SetSketch<ResT, FT> *ptr=static_cast<const SetSketch<ResT, FT> *>(nullptr)) const {
        static std::unordered_map<FT, std::vector<FT>> powers;
        CONST_IF(sizeof(ResT) >= 4) {
            ska::flat_hash_map<ResT, uint32_t> counts;
            if(ptr) {
                for(size_t i = 0; i < m_; ++i)
                    ++counts[std::max(data_[i], ptr->data()[i])];
            } else for(size_t i = 0; i < m_; ++counts[data_[i++]]);
            return std::accumulate(counts.begin(), counts.end(), static_cast<FT>(0.L), [b=b_](long double s, const std::pair<ResT, uint32_t> &reg) {return std::fma(reg.second, std::pow(b, -static_cast<ptrdiff_t>(reg.first)), s);});
        }
        auto it = powers.find(b_);
        if(it == powers.end()) {
            it = powers.emplace(b_, std::vector<FT>()).first;
            it->second.resize(q_ + 2);
            for(size_t i = 0; i < it->second.size(); ++i) {
                it->second[i] = std::pow(static_cast<long double>(b_), -static_cast<ptrdiff_t>(i));
            }
        }
        if(q_ <= 256) {
            std::vector<uint32_t> counts(q_ + 2);
            if(ptr) {
                for(size_t i = 0; i < m_; ++i)
                    ++counts[std::max(data_[i], ptr->data()[i])];
            } else for(size_t i = 0; i < m_; ++counts[data_[i++]]);
            return std::inner_product(&counts[lowkh_.klow()], &counts[q_ + 2], &it->second[lowkh_.klow()], 0.L);
        } else {
            ska::flat_hash_map<ResT, uint32_t> counts; counts.reserve(q_ + 2);
            if(ptr) {
                for(size_t i = 0; i < m_; ++i)
                    ++counts[std::max(data_[i], ptr->data()[i])];
            } else for(size_t i = 0; i < m_; ++counts[data_[i++]]);
            auto &ptable = it->second;
            return std::accumulate(counts.begin(), counts.end(), static_cast<FT>(0.L), [&ptable](long double s, const std::pair<ResT, uint32_t> &reg) {return std::fma(reg.second, ptable[reg.first], s);});
        }
    }
    double jaccard_by_ix(const SetSketch<ResT, FT> &o) const {
        auto us = union_size(o);
        auto mycard = getcard(), ocard = o.getcard();
        return (mycard + ocard - us) / us;
    }
    double union_size(const SetSketch<ResT, FT> &o) const {
        double num = m_ * (1. - 1. / b_) * logbinv_ * ainv_;
        return num / harmean(&o);
    }
    double cardinality_estimate() const {return cardinality();}
    double cardinality() const {
        double num = m_ * (1. - 1. / b_) * logbinv_ * ainv_;
        return num / harmean();
    }
    void merge(const SetSketch<ResT, FT> &o) {
        if(!same_params(o)) throw std::runtime_error("Can't merge sets with differing parameters");
        std::transform(data(), data() + m_, o.data(), data(), [](auto x, auto y) {return std::max(x, y);});
        mycard_ = -1.;
    }
    SetSketch &operator+=(const SetSketch<ResT, FT> &o) {merge(o); return *this;}
    SetSketch operator+(const SetSketch<ResT, FT> &o) const {
        SetSketch ret(*this);
        ret += o;
        return ret;
    }
    size_t shared_registers(const SetSketch<ResT, FT> &o) const {
        return eq::count_eq(data(), o.data(), m_);
    }
    std::pair<double, double> alpha_beta(const SetSketch<ResT, FT> &o) const {
        auto gtlt = eq::count_gtlt(data(), o.data(), m_);
        double alpha = g_b(b_, double(gtlt.first) / m_);
        double beta = g_b(b_, double(gtlt.second) / m_);
        return {alpha, beta};
    }
    static constexpr double __union_card(double alph, double beta, double lhcard, double rhcard) {
        return std::max((lhcard + rhcard) / (2. - alph - beta), 0.);
    }
    double getcard() const {
        if(mycard_ < 0.)
            mycard_ = cardinality();
        return mycard_;
    }
    double jaccard_index(const SetSketch<ResT, FT> &o) const {
        if(!same_params(o))
            throw std::invalid_argument("Parameters must match for comparison");
        auto gtlt = eq::count_gtlt(data(), o.data(), m_);
        return jmle_simple<double>(gtlt.first, gtlt.second, m_, getcard(), o.getcard(), b_);
    }
    std::tuple<double, double, double> jointmle(const SetSketch<ResT, FT> &o) const {
        auto ji = jaccard_index(o);
        const auto y = 1. / (1. + ji);
        double mycard = getcard(), ocard = o.getcard();
        return {std::max(0., mycard - ocard * ji) * y,
                std::max(0., ocard - mycard * ji) * y,
                (mycard + ocard) * ji * y};
    };
    double jaccard_index_by_card(const SetSketch<ResT, FT> &o) const {
        auto tup = jointmle(o);
        return std::get<2>(tup) / (std::get<0>(tup) + std::get<1>(tup) + std::get<2>(tup));
    }
    std::tuple<double, double, double> alpha_beta_mu(const SetSketch<ResT, FT> &o) const {
        auto gtlt = eq::count_gtlt(data(), o.data(), m_);
        double alpha = g_b(b_, double(gtlt.first) / m_);
        double beta = g_b(b_, double(gtlt.second) / m_);
        double mycard = getcard(), ocard = o.getcard();
        if(alpha + beta >= 1.) // They seem to be disjoint sets, use SetSketch (15)
            return {(mycard) / (mycard + ocard), ocard / (mycard + ocard), mycard + ocard};
        return {alpha, beta, __union_card(alpha, beta, mycard, ocard)};
    }
    void write(std::string s) const {
        gzFile fp = gzopen(s.data(), "w");
        if(!fp) throw ZlibError(std::string("Failed to open file ") + s + "for writing");
        write(fp);
        gzclose(fp);
    }
    void read(std::string s) {
        gzFile fp = gzopen(s.data(), "r");
        if(!fp) throw ZlibError(std::string("Failed to open file ") + s);
        read(fp);
        gzclose(fp);
    }
    void read(gzFile fp) {
        gzread(fp, &m_, sizeof(m_));
        gzread(fp, &a_, sizeof(a_));
        gzread(fp, &b_, sizeof(b_));
        gzread(fp, &q_, sizeof(q_));
        ainv_ = 1.L / a_;
        logbinv_ = 1.L / std::log1p(b_ - 1.);
        data_.reset(allocate(m_));
        lowkh_.assign(data_.get(), m_, b_);
        gzread(fp, (void *)data_.get(), m_ * sizeof(ResT));
        std::fill(&data_[m_], &data_[2 * m_ - 1], ResT(0));
        for(size_t i = 0;i < m_; ++i) lowkh_.update(i, data_[i]);
        ls_.resize(m_);
    }
    int checkwrite(std::FILE *fp, const void *ptr, size_t nb) const {
        auto ret = ::write(::fileno(fp), ptr, nb);
        if(size_t(ret) != nb) throw ZlibError("Failed to write setsketch to file");
        return ret;
    }
    int checkwrite(gzFile fp, const void *ptr, size_t nb) const {
        auto ret = gzwrite(fp, ptr, nb);
        if(size_t(ret) != nb) throw ZlibError("Failed to write setsketch to file");
        return ret;
    }
    void write(std::FILE *fp) const {
        checkwrite(fp, (const void *)&m_, sizeof(m_));
        checkwrite(fp, (const void *)&a_, sizeof(a_));
        checkwrite(fp, (const void *)&b_, sizeof(b_));
        checkwrite(fp, (const void *)&q_, sizeof(q_));
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(ResT));
    }
    void write(gzFile fp) const {
        checkwrite(fp, (const void *)&m_, sizeof(m_));
        checkwrite(fp, (const void *)&a_, sizeof(a_));
        checkwrite(fp, (const void *)&b_, sizeof(b_));
        checkwrite(fp, (const void *)&q_, sizeof(q_));
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(ResT));
    }
    void clear() {
        std::fill(data_.get(), &data_[m_ * 2 - 1], ResT(0));
        mycard_ = -1.;
    }
    const std::vector<uint64_t> &ids() const {return ids_;}
};

template<typename ResT, typename FT>
class CountFilteredSetSketch: public SetSketch<ResT, FT> {
    using Super = SetSketch<ResT, FT>;

    const uint32_t mc_;
    ska::flat_hash_map<uint64_t, uint32_t> potentials_;
public:
    template<typename...Args>
    CountFilteredSetSketch(int32_t mincount=1, Args &&...args): Super(std::forward<Args>(args)...), mc_(mincount) {
    }
    void reset() {
        CSetSketch<FT>::reset();
        potentials_.clear();
    }
    double getlim(const uint64_t id) const {
        using GenFT = std::conditional_t<(sizeof(FT) <= 8), double, long double>;
        uint64_t hid = id;
        uint64_t rv = wy::wyhash64_stateless(&hid);
        GenFT ev;
        const GenFT ba = -this->ainv() / this->size();
        if(sizeof(GenFT) > 8) {
            auto lrv = __uint128_t(rv) << 64;
            lrv |= wy::wyhash64_stateless(&rv);
            ev = ba * std::log((lrv >> 32) * 1.2621774483536188887e-29L);
        } else ev = ba * std::log(rv * INVMUL64);
        return ev;
    }
    bool check_can_update(const uint64_t id) const {
        return getlim(id) < this->explim();
    }
    template<typename IT, typename OFT, typename=typename std::enable_if<std::is_arithmetic<OFT>::value>::type>
    void update(const IT id, OFT) {update(id);}
    void trim_potentials() {
        const auto lim = this->explim();
        for(auto it = potentials_.begin(), eit = potentials_.end(); it != eit; ++it) {
            if(getlim(it->first) < lim) potentials_.erase(it);
        }
    }
    void update(const uint64_t id) {
        if(mc_ > 1u) {
            if((CEHasher()(id) & 0x8fffffu) == 0u) {
                trim_potentials();
            }
            if(!check_can_update(id)) return;
            auto pit = potentials_.find(id);
            if(pit == potentials_.end()) {
                potentials_.emplace(id, 1);
                return;
            }
            if(pit->second >= mc_) {
                ++pit->second; // Already added
                return;
            }
            if(++pit->second < mc_) return;
        }
        Super::update(id);
    }
};


#define CFDeclare(desttype, A, B, QC, RT, FT) \
struct desttype: SetSketch<RT, FT> {\
    static constexpr long double DEFAULT_B = A;\
    static constexpr long double DEFAULT_A = B;\
    desttype(size_t nreg, double b=DEFAULT_B, double a=DEFAULT_A): SetSketch<RT, FT>(nreg, b, a, QV) {}\
    static constexpr size_t QV = QC;\
    template<typename Arg> desttype(const Arg &arg): SetSketch<RT, FT>(arg) {}\
};\
struct CF##desttype: CountFilteredSetSketch<RT, FT> {\
    static constexpr long double DEFAULT_B = A;\
    static constexpr long double DEFAULT_A = B;\
    CF##desttype(uint32_t mincount, size_t nreg, double b=DEFAULT_B, double a=DEFAULT_A): CountFilteredSetSketch<RT, FT>(mincount, nreg, b, a, QV) {}\
    static constexpr size_t QV = QC;\
    template<typename Arg> CF##desttype(uint32_t mincount, const Arg &arg): CountFilteredSetSketch<RT, FT>(mincount, arg) {}\
}

CFDeclare(NibbleSetS, 2.7182818284590452354L, 5e-4L, 14u, uint8_t, double);
CFDeclare(SmallNibbleSetS, 4L, 1e-6L, 14u, uint8_t, double);
CFDeclare(ByteSetS, 1.2, 20., 254u, uint8_t, double);
CFDeclare(ShortSetS, 1.0005, .06, 65534u, uint16_t, long double);
struct WideShortSetS: public SetSketch<uint16_t, long double> {
    static constexpr long double DEFAULT_B = 1.0004;
    static constexpr long double DEFAULT_A = .06;
    static constexpr size_t QV = 65534u;
    WideShortSetS(size_t nreg, long double b=DEFAULT_B, long double a=DEFAULT_A): SetSketch<uint16_t, long double>(nreg, b, a, QV) {}
    template<typename...Args> WideShortSetS(Args &&...args): SetSketch<uint16_t, long double>(std::forward<Args>(args)...) {}
};
struct EShortSetS: public SetSketch<uint16_t, long double> {
    using Super = SetSketch<uint16_t, long double>;
    static constexpr long double DEFAULT_B = 1.0006;
    static constexpr long double DEFAULT_A = .06;
    static constexpr size_t QV = 65534u;
    template<typename IT, typename OFT, typename=typename std::enable_if<std::is_integral<IT>::value && std::is_floating_point<OFT>::value>::type>
    EShortSetS(IT nreg, OFT b=DEFAULT_B, OFT a=DEFAULT_A): Super(nreg, b, a, QV) {}
    EShortSetS(size_t nreg): Super(nreg, DEFAULT_B, DEFAULT_A, QV) {}
    EShortSetS(int nreg): Super(nreg, DEFAULT_B, DEFAULT_A, QV) {}
    template<typename...Args> EShortSetS(Args &&...args): Super(std::forward<Args>(args)...) {}
};
struct EByteSetS: public SetSketch<uint8_t, double> {
    static constexpr double DEFAULT_B = 1.09;
    static constexpr double DEFAULT_A = .08;
    static constexpr size_t QV = 254u;
    template<typename IT, typename=typename std::enable_if<std::is_integral<IT>::value>::type>
    EByteSetS(IT nreg, double b=DEFAULT_B, double a=DEFAULT_A): SetSketch<uint8_t, double>(nreg, b, a, QV) {}
    template<typename...Args> EByteSetS(Args &&...args): SetSketch<uint8_t, double>(std::forward<Args>(args)...) {}
};
CFDeclare(UintSetS, 1.0000000109723500835L, 19.77882586L, 0xFFFFFFFEuL, uint32_t, long double);
#undef CFDeclare

template<typename FT=double>
struct CountFilteredCSetSketch: public CSetSketch<FT> {
    using super = CSetSketch<FT>;
    const uint32_t mc_;
    ska::flat_hash_map<uint64_t, uint32_t> potentials_;
#ifdef VERBOSE_AF
    size_t numremoved = 0;
    ~CountFilteredCSetSketch() {
        std::fprintf(stderr, "%zu removed total in lifetime\n", numremoved);
    }
#endif
    template<typename...Args>
    CountFilteredCSetSketch(uint32_t mincount=1, Args &&...args): CSetSketch<FT>(std::forward<Args>(args)...), mc_(mincount) {
    }
    void reset() {
        CSetSketch<FT>::reset();
        potentials_.clear();
    }
    // If a weight is passed, ignore it
    template<typename OFT, typename=typename std::enable_if<std::is_arithmetic<OFT>::value>::type>
    void update(const uint64_t id, OFT) {update(id);}
    using super::mycard_;
    using super::total_updates_;
    using super::idcounts_;
    using super::ids_;
    using super::mvt_;
    using super::m_;
    using super::max;
    using super::ls_;
    using super::getbeta;
    long double id2ldv(uint64_t *rv, double mi) const {
        auto lrv = __uint128_t(*rv) << 64;
        lrv |= wy::wyhash64_stateless(rv);
        long double tv = (lrv >> 32) * 1.2621774483536188887e-29L;
        return mi * std::log(tv);
    }
    INLINE void erase_if(typename ska::flat_hash_map<uint64_t, uint32_t>::iterator it) {
#ifdef VERBOSE_AF
        ++numremoved;
#endif
        potentials_.erase(it);
    }
    void trim_potentials(FT mv) {
        using fastlog::flog;
        for(auto it = potentials_.begin(); it != potentials_.end(); ++it) {
            const FT mi = -1.L / m_;
            FT nv;
            uint64_t hid = it->first;
            uint64_t rv = wy::wyhash64_stateless(&hid);
            CONST_IF(sizeof(FT) > 8) {
                nv = id2ldv(&rv, mi);
                // Uses 96 bits of precision
            } else {
                auto tv = rv * INVMUL64;
                // Filter with fast log first
                nv = mi * flog(tv) * FT(.7);
                if(nv < mv) nv = mi * std::log(tv);
            }
            if(nv >= mv) {
                erase_if(it);
                continue;
            }
        }
    }
    void update(const uint64_t id) {
        using fastlog::flog;
        if(mc_ <= 1u) return CSetSketch<FT>::update(id);
        FT kahan_carry = 0;
        mycard_ = -1.;
        ++total_updates_;
        uint64_t hid = id;
        sketch::hash::CEHasher ch;
        uint64_t rv = sketch::hash::CEHasher()(id ^ uint64_t(0xb2069fc679a8da0buLL));

        FT ev;
        FT mv = max();
        if((CEHasher()(id) & 0x8fffffu) == 0u)
            trim_potentials(mv);
        CONST_IF(sizeof(FT) > 8) {
            if((ev = id2ldv(&rv, -1.L / m_)) > mv) return;
        } else {
            auto tv = rv * INVMUL64;
            const FT bv = -1. / m_;
            // Filter with fast log first
            if(bv * flog(tv) * FT(.7) > mv || (ev = bv * std::log(tv)) > mv) return;
        }
        auto pit = potentials_.find(id);
        if(pit == potentials_.end()) {
            potentials_.emplace(id, 1);
            return;
        }
        if(pit->second >= mc_) {
            ++pit->second; // Already added
            return;
        }
        if(++pit->second < mc_) return;
        // What's left now is that we have just reached the minimum count
        // We will periodically remove unnecessary k-mers as the sketch becomes filled.
        // This is done randomly as a function of the random id;
        ls_.reset();
        ls_.seed(rv);
        uint64_t bi = 1;
        uint32_t idx;
        for(;;) {
            idx = ls_.step();
            if(mvt_.update(idx, ev)) {
                if(!ids_.empty()) {
                    ids_.operator[](idx) = id;
                    if(!idcounts_.empty()) idcounts_[idx] = 1;
                }
                mv = max();
            } else if(!idcounts_.empty() && id == ids_[idx]) {
                ++idcounts_[idx];
            }
            if(bi == m_) return;
            rv = wy::wyhash64_stateless(&hid);
            const FT bv = -getbeta(bi++);
            CONST_IF(sizeof(FT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                if(kahan::update(ev, kahan_carry, bv * std::log((lrv >> 32) * 1.2621774483536188887e-29L)) > mv)
                    break;
            } else {
                const FT nv = rv * INVMUL64;
                if(bv * flog(nv) * FT(.7) + ev > mv) {
                    assert(std::fma(bv, std::log(nv), ev) > mv);
                    break;
                }
                if(kahan::update(ev, kahan_carry, bv * std::log(nv)) > mv)
                    break;
            }
        }
    }
};
template<typename FT>
static inline double intersection_size(const CSetSketch<FT> &lhs, const CSetSketch<FT> &rhs) {
    return lhs.intersection_size(rhs);
}

} // namespace setsketch
using setsketch::CSetSketch;

} // namespace sketch

#endif /*  D2_SETSKETCH_H___H__ */
