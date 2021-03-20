#ifndef EHLL_H__
#define EHLL_H__
#include <stdexcept>
#include <cassert>
#include "aesctr/wy.h"
#include <queue>
#include "sketch/div.h"
#include <unordered_map>
#include <memory>
#include "fy.h"
#include "sketch/count_eq.h"
#include "sketch/macros.h"
#include "blaze/Math.h"
#include "sketch/hash.h"
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {

namespace setsketch {

namespace detail {
    struct Deleter {
        template<typename T>
        void operator()(const T *x) const {std::free(const_cast<T *>(x));}
    };
}

#define LC_ONLY(...)



#if __cplusplus >= 201703L
    static constexpr double INVMUL64 = 0x1p-64;
#else
    static constexpr double INVMUL64 = 5.42101086242752217e-20;
#endif

// Implementations of set sketch

template<typename FT>
class mvt_t {
    FT mv_;
    FT *data_ = nullptr;
    size_t m_;
public:
    mvt_t(size_t m, FT mv = std::numeric_limits<FT>::max()): mv_(m), m_(m) {}

    FT mv() const {return mv_;}
    FT *data() {return data_;}
    const FT *data() const {return data_;}
    // Check size and max
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
    // Check size and max
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

template<typename FT>
inline double calc_card(const FT *start, const FT *end) {
    const std::ptrdiff_t n = end - start;
    return n /
        blaze::serial(blaze::sum(blaze::CustomVector<FT, blaze::aligned, blaze::unpadded>(
                const_cast<FT *>(start), n)));
}

template<typename ResT, typename FT=double> class SetSketch; // Forward


template<typename FT=double, bool FLOGFILTER=true>
class CSetSketch {
    // TODO: Add stochastically-averaged case for faster sketching.
    //      (With the approximate log trick, we're still around only twice the time as HLL)
    static_assert(std::is_floating_point<FT>::value, "Must float");
    // SetSketch 1
    size_t m_; // Number of registers
    std::unique_ptr<FT[], detail::Deleter> data_;
    fy::LazyShuffler ls_;
    mvt_t<FT> mvt_;
    std::vector<uint64_t> ids_;
    std::vector<uint32_t> idcounts_;
    blaze::DynamicVector<FT> beta_;
    uint64_t total_updates_ = 0;
    uint64_t inner_loop_updates_= 0;
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
#if 0
    // Caching these coefficients doesn't seem to make a big difference.
    void generate_betas() {
        beta_.resize(m_);
        beta_ = blaze::serial(blaze::generate(m_, [m=m_](size_t i) {return 1. / (m - i);}));
    }
#endif
    FT getbeta(size_t idx) const {
#if 0
        return beta_[idx];
#else
        return FT(1.) / static_cast<FT>(m_ - idx);
#endif
    }
public:
    const FT *data() const {return data_.get();}
    FT *data() {return data_.get();}
    CSetSketch(size_t m, bool track_ids=false, bool track_counts=false, FT maxv=std::numeric_limits<FT>::max()): m_(m), ls_(m_), mvt_(m_) {
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
            ret.lowkh().update(i, std::max(int64_t(0), std::min(int64_t(q) + 1, static_cast<int64_t>((1. - std::log(data_[i]) * logbinv)))));
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
        inner_loop_updates_ = o.inner_loop_updates_;
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
    size_t floopupdates = 0;
    size_t inner_loop_updates() const {return inner_loop_updates_;}
    long double flog(long double x) const {
        __uint128_t yi;
        std::memcpy(&yi, &x, sizeof(x));
        return yi * 3.7575583950764744255e-20L - 11356.176832703863597L;
    }
    double flog(double x) const {
        uint64_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 1.539095918623324e-16 - 709.0895657128241;
    }
    float flog(float x) const {
        uint32_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 8.2629582881927490e-8f - 88.02969186f;
    }
    void update(const uint64_t id) {
        ++total_updates_;
        uint64_t hid = id;
        uint64_t rv = wy::wyhash64_stateless(&hid);

        FT ev;
        CONST_IF(sizeof(FT) > 8) {
            auto lrv = __uint128_t(rv) << 64;
            const FT bv = -1. / m_;
            lrv |= wy::wyhash64_stateless(&rv);
            FT tv = static_cast<long double>((lrv >> 32) * 1.2621774483536188887e-29L);
            ev = bv * std::log(tv);
            if(ev >= max()) return;
        } else {
            auto tv = rv * INVMUL64;
            const FT bv = -1. / m_;
            // Filter with fast log first
            CONST_IF(FLOGFILTER) {
                if(bv * flog(tv) * FT(.7) > max()) return;
            }
            ev = bv * std::log(tv);
            if(ev >= max()) return;
        }
        LC_ONLY(++floopupdates;)
        ls_.reset();
        ls_.seed(rv);
        uint64_t bi = 1;
        uint32_t idx = ls_.step();
        FT mv = max();
        for(;;) {
            LC_ONLY(++inner_loop_updates_;)
            if(mvt_.update(idx, ev)) {
                if(!ids_.empty()) {
                    ids_.operator[](idx) = id;
                    if(!idcounts_.empty()) idcounts_.operator[](idx) = 1;
                }
                mv = max();
            } else if(!idcounts_.empty()) {
                if(id == ids_.operator[](idx)) ++idcounts_.operator[](idx);
            }
            if(bi == m_) return;
            rv = wy::wyhash64_stateless(&hid);
            const FT bv = -getbeta(bi++);
            CONST_IF(sizeof(FT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                ev = std::fma(bv, std::log((lrv >> 32) * 1.2621774483536188887e-29L), ev);
                if(ev >= mv) break;
            } else {
                const FT nv = rv * INVMUL64;
                CONST_IF(FLOGFILTER) {
                    if(bv * flog(nv) * FT(.7) + ev >= mv) break;
                }
                ev = std::fma(bv, std::log(nv), ev);
                if(ev >= mv) break;
            }
            idx = ls_.step();
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
    auto cv() {return blaze::CustomVector<FT, blaze::aligned, blaze::unpadded>(data(), m_);}
    auto cv() const {return blaze::CustomVector<const FT, blaze::aligned, blaze::unpadded>(data(), m_);}
    void merge(const CSetSketch<FT> &o) {
        if(!same_params(o)) throw std::runtime_error("Can't merge sets with differing parameters");
        if(ids().empty()) {
            cv() = blaze::serial(blaze::min(cv(), o.cv()));
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
        inner_loop_updates_ += o.inner_loop_updates_;
        total_updates_ += o.total_updates_;
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
    }
    const std::vector<uint64_t> &ids() const {return ids_;}
    const std::vector<uint32_t> &idcounts() const {return idcounts_;}
    double union_size(const CSetSketch<FT> &o) const {
        using CVT = blaze::CustomVector<FT, blaze::aligned, blaze::unpadded>;
        return blaze::serial(blaze::sum(blaze::max(
            CVT(const_cast<FT *>(data_.get()), m_), CVT(const_cast<FT *>(o.data_.get()), m_))));
    }
    auto alpha_beta(const CSetSketch<FT> &o) const {
        auto gtlt = eq::count_gtlt(data(), o.data(), m_);
        return std::pair<double, double>{double(gtlt.first) / m_, double(gtlt.second) / m_};
    }
    static constexpr double __union_card(double alph, double beta, double lhcard, double rhcard) {
        return std::max((lhcard + rhcard) / (2. - alph - beta), 0.);
    }
    double intersection_size(const CSetSketch<FT> &o, double mycard=-1., double ocard=-1.) const {
        if(mycard < 0) mycard = cardinality();
        if(ocard < 0) ocard = o.cardinality();
        auto triple = alpha_beta_mu(o, mycard, ocard);
        return std::max(1. - (std::get<0>(triple) + std::get<1>(triple)), 0.) * std::get<2>(triple);
    }
    std::tuple<double, double, double> alpha_beta_mu(const CSetSketch<FT> &o, double mycard, double ocard) const {
        const auto ab = alpha_beta(o);
        if(ab.first + ab.second >= 1.) // They seem to be disjoint sets, use SetSketch (15)
            return {(mycard) / (mycard + ocard), ocard / (mycard + ocard), mycard + ocard};
        return {ab.first, ab.second, __union_card(ab.first, ab.second, mycard, ocard)};
    }

    double cardinality_estimate() const {return cardinality();}
    double cardinality() const {
        return calc_card(data_.get(), &data_[m_]);
    }
    static std::pair<long double, long double> optimal_parameters(FT maxreg, FT minreg, size_t q) {
        long double b = std::exp(std::log((long double)maxreg / (long double)minreg) / (long double)q);
        return {FT(b), FT((long double)maxreg / b)};
    }
    template<typename ResT=uint16_t>
    static std::pair<long double, long double> optimal_parameters(FT maxreg, FT minreg) {
        if(maxreg < minreg) std::swap(maxreg, minreg);
        return optimal_parameters(maxreg, minreg, std::numeric_limits<ResT>::max());
    }
    double containment_index(const CSetSketch<FT> &o, double mycard=-1., double ocard=-1) const {
        if(mycard < 0) mycard = cardinality();
        if(ocard < 0) ocard = o.cardinality();
        auto abm = alpha_beta_mu(o, mycard, ocard);
        auto lho = std::get<0>(abm);
        auto isf = std::max(1. - (lho + std::get<1>(abm)), 0.);
        return isf / (lho + isf);
    }
};

template<typename FT>
double intersection_size(const CSetSketch<FT> &lhs, const CSetSketch<FT> &rhs) {
    return lhs.intersection_size(rhs);
}

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
    using QType = std::common_type_t<ResT, int>;
    QType q_;
    std::unique_ptr<ResT[], detail::Deleter> data_;
    std::vector<uint64_t> ids_; // The IDs representing the sampled items.
                                // Only used if SetSketch is
    fy::LazyShuffler ls_;
    minvt_t<ResT> lowkh_;
    std::vector<FT> lbetas_; // Cache Beta values * 1. / a
    static ResT *allocate(size_t n) {
        n = (n << 1) - 1;
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
        if((ret = static_cast<ResT *>(std::aligned_alloc(ALN, n * sizeof(ResT)))) == nullptr)
#else
        if(posix_memalign((void **)&ret, ALN, n * sizeof(ResT)))
#endif
            throw std::bad_alloc();
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / (m_ - idx);
    }
public:
    const ResT *data() const {return data_.get();}
    ResT *data() {return data_.get();}
    auto &lowkh() {return lowkh_;}
    const auto &lowkh() const {return lowkh_;}
    SetSketch(size_t m, FT b, FT a, int q, bool track_ids = false): m_(m), a_(a), b_(b), ainv_(1./ a), logbinv_(1. / std::log1p(b_ - 1.)), q_(q), ls_(m_), lowkh_(m) {
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
    void update(const uint64_t id) {
        uint64_t hid = id;
        size_t bi = 0;
        uint64_t rv = wy::wyhash64_stateless(&hid);
        double ev = 0.;
        ls_.reset();
        ls_.seed(rv);
        for(;;) {
            const auto ba = lbetas_[bi];
            if(sizeof(FT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                ev += ba * std::log((lrv >> 32) * 1.2621774483536188887e-29L);
            } else {
                ev += ba * std::log(rv * INVMUL64);
            }
            if(ev > lowkh_.explim()) return;
            const QType k = std::max(0, std::min(q_ + 1, static_cast<QType>((1. - std::log(ev) * logbinv_))));
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
        auto it = powers.find(b_);
        if(it == powers.end()) {
            it = powers.emplace(b_, std::vector<FT>()).first;
            it->second.resize(q_ + 2);
            for(size_t i = 0; i < it->second.size(); ++i) {
                it->second[i] = std::pow(static_cast<long double>(b_), -static_cast<ptrdiff_t>(i));
            }
        }
        std::vector<uint32_t> counts(q_ + 2);
        if(ptr) {
            for(size_t i = 0; i < m_; ++i) {
                ++counts[std::max(data_[i], ptr->data()[i])];
            }
        } else {
            for(size_t i = 0; i < m_; ++i) {
                ++counts[data_[i]];
            }
        }
        long double ret = 0.;
        for(ptrdiff_t i = lowkh_.klow(); i <= q_ + 1; ++i) {
            ret += counts[i] * it->second[i];
        }
        return ret;
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
    std::tuple<double, double, double> alpha_beta_mu(const SetSketch<ResT, FT> &o, double mycard, double ocard) const {
        const auto ab = alpha_beta(o);
        if(ab.first + ab.second >= 1.) // They seem to be disjoint sets, use SetSketch (15)
            return {(mycard) / (mycard + ocard), ocard / (mycard + ocard), mycard + ocard};
        return {ab.first, ab.second, __union_card(ab.first, ab.second, mycard, ocard)};
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
    }
    const std::vector<uint64_t> &ids() const {return ids_;}
};


#ifndef M_E
#define EULER_E 2.718281828459045
#else
#define EULER_E M_E
#endif
struct NibbleSetS: public SetSketch<uint8_t> {
    NibbleSetS(size_t nreg, double b=EULER_E, double a=5e-4): SetSketch<uint8_t>(nreg, b, a, QV) {}
    static constexpr size_t QV = 14u;
    template<typename Arg> NibbleSetS(const Arg &arg): SetSketch<uint8_t>(arg) {}
};
struct SmallNibbleSetS: public SetSketch<uint8_t> {
    SmallNibbleSetS(size_t nreg, double b=4., double a=1e-6): SetSketch<uint8_t>(nreg, b, a, QV) {}
    static constexpr size_t QV = 14u;
    template<typename Arg> SmallNibbleSetS(const Arg &arg): SetSketch<uint8_t>(arg) {}
};
struct ByteSetS: public SetSketch<uint8_t, long double> {
    using Super = SetSketch<uint8_t, long double>;
    static constexpr size_t QV = 254u;
    ByteSetS(size_t nreg, long double b=1.2, long double a=20.): Super(nreg, b, a, QV) {}
    template<typename Arg> ByteSetS(const Arg &arg): Super(arg) {}
};
struct ShortSetS: public SetSketch<uint16_t, long double> {
    static constexpr long double DEFAULT_B = 1.001;
    static constexpr long double DEFAULT_A = .25;
    static constexpr size_t QV = 65534u;
    ShortSetS(size_t nreg, long double b=DEFAULT_B, long double a=DEFAULT_A): SetSketch<uint16_t, long double>(nreg, b, a, QV) {}
    template<typename Arg> ShortSetS(const Arg &arg): SetSketch<uint16_t, long double>(arg) {}
};
struct WideShortSetS: public SetSketch<uint16_t, long double> {
    static constexpr long double DEFAULT_B = 1.0006;
    static constexpr long double DEFAULT_A = .001;
    static constexpr size_t QV = 65534u;
    WideShortSetS(size_t nreg, long double b=DEFAULT_B, long double a=DEFAULT_A): SetSketch<uint16_t, long double>(nreg, b, a, QV) {}
    template<typename...Args> WideShortSetS(Args &&...args): SetSketch<uint16_t, long double>(std::forward<Args>(args)...) {}
};
struct EShortSetS: public SetSketch<uint16_t, long double> {
    using Super = SetSketch<uint16_t, long double>;
    static constexpr long double DEFAULT_B = 1.0006;
    static constexpr long double DEFAULT_A = .001;
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

template<typename FT=double, typename KeyT=uint64_t, typename IdT=uint32_t>
struct SetSketchIndex {
    /*
     * Maintains an LSH index over a set of sketches
     *
     */
private:
    size_t m_;
    using HashMap = ska::flat_hash_map<KeyT, std::vector<IdT>>;
    using HashV = std::vector<HashMap>;
    std::vector<HashV> packed_maps_;
    std::vector<uint64_t> regs_per_reg_;
    size_t total_ids_ = 0;
public:
    size_t m() const {return m_;}
    size_t size() const {return total_ids_;}
    SetSketchIndex(size_t m, bool densified=false): m_(m) {
        uint64_t rpr = 1;
        const size_t nrpr = densified ? m: size_t(ilog2(sketch::integral::roundup(m)));
        regs_per_reg_.reserve(nrpr);
        packed_maps_.reserve(nrpr);
        for(;rpr <= m_;) {
            regs_per_reg_.push_back(rpr);
            packed_maps_.emplace_back(HashV(m_ / rpr));
            if(densified) {
                ++rpr;
            } else {
                rpr <<= 1;
            }
        }
    }
    template<typename Sketch>
    void update(const Sketch &item) {
        if(item.size() != m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        const auto my_id = total_ids_++;
        const size_t n_subtable_lists = regs_per_reg_.size();
        using ResT = typename std::decay_t<decltype(item[0])>;
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            const size_t nsubs = subtab.size();
            const size_t nelem = regs_per_reg_[i];
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = XXH3_64bits(&item[nelem * j], nelem * sizeof(ResT));
                subtab[j][myhash].push_back(my_id);
            }
        }
    }
    template<typename Sketch>
    std::pair<std::vector<IdT>, std::vector<uint32_t>>
    query_candidates(const Sketch &item, size_t maxcand) const {
        /*
         *  Returns ids matching input minhash sketches, in order from most specific/least sensitive
         *  to least specific/most sensitive
         *  Can be then used, along with sketches, to select nearest neighbors
         *  */
        using ResT = typename std::decay_t<decltype(item[0])>;
        ska::flat_hash_set<IdT> rset; rset.reserve(maxcand);
        std::vector<IdT> passing_ids;
        std::vector<uint32_t> items_per_row;
        for(std::ptrdiff_t i = regs_per_reg_.size();--i >= 0;) {
            auto &m = packed_maps_[i];
            const size_t nelem = regs_per_reg_[i];
            const size_t nsubs = m.size();
            const size_t items_before = passing_ids.size();
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = XXH3_64bits(&item[nelem * j], nelem * sizeof(ResT));
                auto it = m[j].find(myhash);
                if(it == m[j].end()) continue;
                for(const auto id: it->second) {
                    auto rit2 = rset.find(id);
                    if(rit2 == rset.end()) {
                        rset.insert(id);
                        passing_ids.push_back(id);
                    }
                }
            }
            items_per_row.push_back(passing_ids.size() - items_before);
            if(rset.size() >= maxcand) {
                std::fprintf(stderr, "Candidate set of size %zu, > maxc %zu\n", rset.size(), maxcand);
                break;
            }
        }
        return std::make_pair(passing_ids, items_per_row);
    }
};



} // namespace setsketch
using setsketch::EByteSetS;
using setsketch::ByteSetS;
using setsketch::ShortSetS;
using setsketch::EShortSetS;
using setsketch::WideShortSetS;
using setsketch::NibbleSetS;
using setsketch::SmallNibbleSetS;
using setsketch::CSetSketch;
using setsketch::SetSketch;

} // namespace sketch

#endif
