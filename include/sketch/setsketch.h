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
#include "count_eq.h"
#include "blaze/Math.h"

#ifndef NDEBUG
#include <unordered_set>
#endif

namespace sketch {


#if __cplusplus >= 201703L
    static constexpr double INVMUL64 = 0x1p-64;
#else
    static constexpr double INVMUL64 = 5.42101086242752217e-20;
#endif

// Implementations of set sketch

template<typename FT>
class mvt_t {
    static constexpr FT mv_ = std::numeric_limits<FT>::max();
    FT *data_ = nullptr;
    size_t m_;
public:
    mvt_t(size_t m): m_(m) {}

    FT *data() {return data_;}
    const FT *data() const {return data_;}
    // Check size and max
    size_t getm() const {return m_;}
    size_t nelem() const {return 2 * m_ - 1;}
    FT operator[](size_t i) const {return data_[i];}
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
    float ret = m2[0];
    std::fprintf(stderr, "sum of %g/%g/%g/%g/%g/%g/%g/%g is %g\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], ret);
    return ret;
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
    bool is_aligned = (reinterpret_cast<uint64_t>(start) %
#if __AVX512F__
        64
#elif __AVX2__
        32
#else
        1
#endif
        ) == 0;
    double sum;
    const size_t n = end - start;
    if(is_aligned) sum = blaze::serial(blaze::sum(blaze::CustomVector<FT, blaze::aligned, blaze::unpadded>(const_cast<FT *>(start), n)));
    else           sum = blaze::serial(blaze::sum(blaze::CustomVector<FT, blaze::unaligned, blaze::unpadded>(const_cast<FT *>(start), n)));
    return n / sum;
}

#if 0
template<> inline double calc_card(const float *const start, const float *const end) {
    const std::ptrdiff_t n = end - start;
    auto blzsum = blaze::sum(blaze::CustomVector<float, blaze::unaligned, blaze::unpadded>(const_cast<float *>(start), n));
    std::fprintf(stderr, "blzsum: %g\n", blzsum);
    double sum = 0.;
    size_t i;
#if __AVX512F__
    static constexpr size_t nper = sizeof(__m512i) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    for(i = 0; i < nsimd4; i += 4) {
        __m512 v0 = _mm512_loadu_ps(&start[i * nper]),
               v1 = _mm512_loadu_ps(&start[(i + 1) * nper])
               v2 = _mm512_loadu_ps(&start[(i + 2) * nper])
               v3 = _mm512_loadu_ps(&start[(i + 3) * nper]);
        sum += _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(v0, v1), _mm512_add_ps(v2, v3)));
    }
    for(i = nsimd4; i < nsimd; ++i) {
        sum += _mm512_reduce_add_ps(_mm512_loadu_ps(&start[i * nper]));
    }
    i = nsimd * nper;
#elif __AVX2__
    static constexpr size_t nper = sizeof(__m256i) / sizeof(float);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    i = 0;
#if 0
    for(; i < nsimd4; i += 4) {
        __m256 v0 = _mm256_loadu_ps(&start[i * nper]),
               v1 = _mm256_loadu_ps(&start[(i + 1) * nper]),
               v2 = _mm256_loadu_ps(&start[(i + 2) * nper]),
               v3 = _mm256_loadu_ps(&start[(i + 3) * nper]);
        sum += broadcast_reduce_sum(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)));
    }
#endif
    for(; i < nsimd; ++i) {
        sum += broadcast_reduce_sum(_mm256_loadu_ps(&start[i * nsimd]));
    }
    i = nsimd * nper;
#endif
    for(;i < n; ++i) {
        sum += start[i];
    }
    std::fprintf(stderr, "blzsum: %g. mysum:%g\n", blzsum, sum);
    return n / sum;
}
template<> inline double calc_card(const double *start, const double *end) {
    const std::ptrdiff_t n = end - start;
    auto blzsum = blaze::sum(blaze::CustomVector<double, blaze::unaligned, blaze::unpadded>(const_cast<double *>(start), n));
    std::fprintf(stderr, "blzsum: %g\n", blzsum);
    double sum = 0.;
    size_t i;
#if __AVX512F__
    static constexpr size_t nper = sizeof(__m512i) / *start;
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    __m512d vsum = _mm512_setzero_pd();
    for(i = 0; i < nsimd4; i += 4) {
        vsum = _mm512_add_pd(vsum,
                _mm512_add_pd(
                    _mm512_add_pd(_mm512_loadu_pd(&start[i * nper]), _mm512_loadu_pd(&start[(i + 1) * nper])),
                    _mm512_add_pd(_mm512_loadu_pd(&start[(i + 2) * nper]), _mm512_loadu_pd(&start[(i + 3) * nper]))
        ));
    }
    switch(nsimd - i) {
        case 3: vsum = _mm512_add_pd(vsum, _mm512_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        case 2: vsum = _mm512_add_pd(vsum, _mm512_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        case 1: vsum = _mm512_add_pd(vsum, _mm512_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        case 0: ;
    }
    sum = _mm512_reduce_add_pd(vsum);
    i = nsimd * nper;
#elif __AVX2__
    static constexpr size_t nper = sizeof(__m256i) / sizeof(*start);
    const size_t nsimd = n / nper;
    const size_t nsimd4 = (nsimd / 4) * 4;
    __m256d vsum = _mm256_setzero_pd();
    for(i = 0; i < nsimd4; i += 4) {
        vsum = _mm256_add_pd(vsum, _mm256_add_pd(
                    _mm256_add_pd(_mm256_loadu_pd(&start[i * nper]), _mm256_loadu_pd(&start[(i + 1) * nper])),
                    _mm256_add_pd(_mm256_loadu_pd(&start[(i + 2) * nper]), _mm256_loadu_pd(&start[(i + 3) * nper]))
        ));
    }
    switch(nsimd - i) {
        case 3: vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        case 2: vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        case 1: vsum = _mm256_add_pd(vsum, _mm256_loadu_pd(&start[i++ * nsimd])); [[fallthrough]];
        default: [[fallthrough]];
        case 0: ;
    }
    sum = broadcast_reduce_sum(vsum);
    i = nsimd * nper;
#endif
    for(;i < n; ++i) sum += start[i];
    std::fprintf(stderr, "mysum: %0.20g. osum: %0.20g\n", sum, blzsum);
    return n / sum;
}
#endif


template<typename FT=double>
class CSetSketch {
    static_assert(std::is_floating_point<FT>::value, "Must float");
    // Set sketch 1
    size_t m_; // Number of registers
    std::unique_ptr<FT[]> data_;
    std::vector<uint64_t> ids_;
    fy::LazyShuffler ls_;
    mvt_t<FT> mvt_;
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
#if __cplusplus >= 201703L && defined(_GLIBCXX_HAVE_ALIGNED_ALLOC)
        if((ret = static_cast<FT *>(std::aligned_alloc(ALN, n * sizeof(FT)))) == nullptr)
#else
        if(posix_memalign((void **)&ret, ALN, n * sizeof(FT)))
#endif
            throw std::bad_alloc();
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / (m_ - idx);
    }
public:
    const FT *data() const {return data_.get();}
    FT *data() {return data_.get();}
    CSetSketch(size_t m, bool track_ids=false): m_(m), ls_(m_), mvt_(m) {
        FT *p = allocate(m_);
        data_.reset(p);
        std::fill(p, p + m_, std::numeric_limits<FT>::max());
        mvt_.assign(p, m_);
        if(track_ids) ids_.resize(m_);
    }
    CSetSketch(const CSetSketch &o): m_(o.m_), ls_(m_), mvt_(m_) {
        FT *p = allocate(m_);
        data_.reset(p);
        mvt_.assign(p, m_);
        std::copy(o.data_.get(), &o.data_[2 * m_ - 1], p);
    }
    CSetSketch(const std::string &s): ls_(1), mvt_(1) {
        read(s);
    }
    FT min() const {return *std::min_element(data(), data() + m_);}
    FT max() const {return mvt_.max();}
    size_t size() const {return m_;}
    FT &operator[](size_t i) {return data_[i];}
    const FT &operator[](size_t i) const {return data_[i];}
    void addh(uint64_t id) {update(id);}
    void add(uint64_t id) {update(id);}
    void update(const uint64_t id) {
        uint64_t hid = id;
        size_t bi = 0;
        uint64_t rv = wy::wyhash64_stateless(&hid);
        FT ev = 0.;
        ls_.reset();
        ls_.seed(rv);
        for(;;) {
            if(sizeof(FT) > 8) {
                auto lrv = __uint128_t(rv) << 64;
                lrv |= wy::wyhash64_stateless(&rv);
                ev += -getbeta(bi) * std::log(static_cast<long double>((lrv >> 32) * 1.2621774483536188887e-29L));
            } else {
                ev += -getbeta(bi) * std::log(rv * INVMUL64);
            }
            if(ev >= mvt_.max()) break;
            auto idx = ls_.step();
            if(mvt_.update(idx, ev)) {
                if(!ids_.empty()) ids_[idx] = id;
            }
            if(++bi == m_)
                break;
            rv = wy::wyhash64_stateless(&hid);
        }
    }
    bool operator==(const CSetSketch<FT> &o) const {
        return same_params(o) && std::equal(data(), data() + m_, o.data());
    }
    bool same_params(const CSetSketch<FT> &o) const {
        return m_ == o.m_;
    }
    void merge(const CSetSketch<FT> &o) {
        if(!same_params(o)) throw std::runtime_error("Can't merge sets with differing parameters");
        std::transform(data(), data() + m_, o.data(), data(), [](auto x, auto y) {return std::min(x, y);});
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
        } else CONST_IF(sizeof(FT) = 2) {
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
        data_.reset(new FT[2 * m_ - 1]);
        mvt_.assign(data_.get(), m_);
        gzread(fp, (void *)data_.get(), m_ * sizeof(FT));
        std::fill(&data_[m_], &data_[2 * m_ - 1], std::numeric_limits<FT>::max());
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
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(FT));
    }
    void write(gzFile fp) const {
        checkwrite(fp, (const void *)&m_, sizeof(m_));
        checkwrite(fp, (const void *)data_.get(), m_ * sizeof(FT));
    }
    void clear() {
        std::fill(data_.get(), &data_[m_ * 2 - 1], std::numeric_limits<FT>::max());
    }
    const std::vector<uint64_t> &ids() const {return ids_;}
    double cardinality() const {
        return calc_card(data_.get(), &data_[m_]);
    }
    static std::pair<FT, FT> optimal_parameters(FT maxreg, FT minreg, size_t q) {
        FT b = std::exp(std::log(minreg / maxreg) / q);
        return {b, minreg / b};
    }
    template<typename ResT=uint16_t>
    static std::pair<FT, FT> optimal_parameters(FT maxreg, FT minreg) {
        static constexpr unsigned long long q = sizeof(ResT) = 1 ? 254ull : sizeof(ResT) == 2 ? 65534ull: sizeof(ResT) == 4 ? 4294967294ull: 18446744073709551614ull;
        return optimal_parameters(maxreg, minreg, q);
    }
};

template<typename ResT, typename FT=double>
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
    std::unique_ptr<ResT[]> data_;
    std::vector<uint64_t> ids_; // The IDs representing the sampled items.
                                // Only used if SetSketch is
    std::vector<FT> lbetas_; // Cache Beta values * 1. / a
    fy::LazyShuffler ls_;
    minvt_t<ResT> lowkh_;
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
        if((ret = static_cast<FT *>(std::aligned_alloc(ALN, n * sizeof(FT)))) == nullptr)
#else
        if(posix_memalign((void **)&ret, ALN, n * sizeof(FT)))
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
                ev += ba * std::log(static_cast<long double>((lrv >> 32) * 1.2621774483536188887e-29L));
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
        data_.reset(new ResT[2 * m_ - 1]);
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

struct NibbleSetS: public SetSketch<uint8_t> {
    NibbleSetS(size_t nreg, double b=16., double a=1.): SetSketch<uint8_t>(nreg, b, a, QV) {}
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
    static constexpr long double DEFAULT_B = 1.0006;
    static constexpr long double DEFAULT_A = .001;
    static constexpr size_t QV = 65534u;
    template<typename IT, typename OFT, typename=typename std::enable_if<std::is_integral<IT>::value && std::is_floating_point<OFT>::value>::type>
    EShortSetS(IT nreg, OFT b=DEFAULT_B, OFT a=DEFAULT_A): SetSketch<uint16_t, long double>(nreg, b, a, QV) {}
    template<typename...Args> EShortSetS(Args &&...args): SetSketch<uint16_t, long double>(std::forward<Args>(args)...) {}
};
struct EByteSetS: public SetSketch<uint8_t, double> {
    static constexpr double DEFAULT_B = 1.09;
    static constexpr double DEFAULT_A = .08;
    static constexpr size_t QV = 254u;
    template<typename IT, typename=typename std::enable_if<std::is_integral<IT>::value>::type>
    EByteSetS(IT nreg, double b=DEFAULT_B, double a=DEFAULT_A): SetSketch<uint8_t, double>(nreg, b, a, QV) {}
    template<typename...Args> EByteSetS(Args &&...args): SetSketch<uint8_t, double>(std::forward<Args>(args)...) {}
};


} // namespace sketch

#endif
