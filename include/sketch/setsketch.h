#ifndef EHLL_H__
#define EHLL_H__
#include <stdexcept>
#include <cassert>
#include "aesctr/wy.h"
#include <queue>
#include <div.h>
#include <unordered_map>
#include <memory>
#include "fy.h"
#include "count_eq.h"

#ifndef NDEBUG
#include <unordered_set>
#endif

namespace sketch {

// Implementations of set sketch

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
    typename std::make_signed<ResT>::type min() const {
        return data_[m_ << 1];
    }
    typename std::make_signed<ResT>::type klow() const {
        return min();
    }
    

    bool update(size_t index, ResT x) {
        const auto sz = (m_ << 1) - 1;
        if(x > data_[index]) {
            for(;;) {
                data_[index] = x;
                if((index = m_ + (index >> 1)) >= sz) break;
                const size_t lhi = (index - m_) << 1, rhi = lhi + 1;
                if((x = std::min(data_[lhi], data_[rhi])) <= data_[index])
                    break;
            }
            explim_ = std::pow(b_, -min());
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

static inline long double g_b(long double b, long double arg) {
    return (1.L - std::pow(b, -arg)) / (1.L - 1.L / b);
}
template<typename ResT, typename FT=double, bool ScanHelper=false>
struct SetSketch {
private:
    static_assert(std::is_floating_point<FT>::value, "Must float");
    static_assert(std::is_integral<ResT>::value, "Must be integral");
    // Set sketch 1
    size_t m_; // Number of registers
    const FT a_; // Exponential parameter
    const FT b_; // Base
    const FT ainv_;
    const FT logbinv_;
    int q_;
    std::unique_ptr<ResT[]> data_;
    fy::LazyShuffler ls_;
    typename std::conditional<true,
                              LowKHelper<ResT>, minvt_t<ResT>
    >::type lowkh_;
    static ResT *allocate(size_t n) {
        if(!ScanHelper) n = (n << 1) - 1;
        ResT *ret = nullptr;
        static constexpr size_t ALN =
#if __AVX512F__
            64;
#elif __AVX2__
            32;
#else
            16;
#endif
        if(posix_memalign((void **)&ret, ALN, n * sizeof(ResT))) throw std::bad_alloc();
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / (m_ - idx);
    }
public:
    const ResT *data() const {return data_.get();}
    SetSketch(size_t m, FT b, FT a, int q): m_(m), a_(a), b_(b), ainv_(1./ a), logbinv_(1. / std::log1p(b_ - 1.)), q_(q), ls_(m_), lowkh_(m) {
        ResT *p = allocate(m_);
        data_.reset(p);
        std::fill(p, p + m_, static_cast<ResT>(0));
        lowkh_.assign(p, m_, b_);
    }
    size_t size() const {return m_;}
    double b() const {return b_;}
    double a() const {return a_;}
    ResT &operator[](size_t i) {return data_[i];}
    const ResT &operator[](size_t i) const {return data_[i];}
    int klow() const {return lowkh_.klow();}
    void update(uint64_t id) {
#ifndef NDEBUG
        //auto mys = std::accumulate(lowkh_.data(), lowkh_.data() + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());});
#endif
        size_t bi = 0;
        uint64_t rv = wy::wyhash64_stateless(&id);
        double ev = 0.;
        ls_.reset();
        ls_.seed(rv);
#ifndef NDEBUG
        std::unordered_set<uint64_t> idxs;
#endif
        for(;;) {
            static constexpr double mul = 1. / (1ull << 52);
            ev += -getbeta(bi) * ainv_ * std::log((rv >> 12) * mul);
            if(ev > lowkh_.explim()) return;
            const int k = std::max(0, std::min(q_ + 1, static_cast<int>((1. - std::log(ev) * logbinv_))));
            if(k <= klow()) return;
            auto idx = ls_.step();
#ifndef NDEBUG
            assert(idxs.find(idx) == idxs.end()); idxs.emplace(idx);
#endif
            lowkh_.update(idx, k);
            if(++bi == m_) {
                return;
            }
            assert(bi == idxs.size());
            //assert(lowkh_.natval_ == std::accumulate(data_.get(), data_.get() + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());}));
            rv = wy::wyhash64_stateless(&id);
        }
    }
    bool same_params(const SetSketch<ResT,FT> &o) {
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
};

struct NibbleSetS: public SetSketch<uint8_t> {
    NibbleSetS(int nreg, double b=16., double a=1.): SetSketch<uint8_t>(nreg, b, a, 14) {}
};
struct SmallNibbleSetS: public SetSketch<uint8_t> {
    SmallNibbleSetS(int nreg, double b=4., double a=1e-6): SetSketch<uint8_t>(nreg, b, a, 14) {}
};
struct ByteSetS: public SetSketch<uint8_t, long double> {
    using Super = SetSketch<uint8_t, long double>;
    ByteSetS(int nreg, long double b=1.2, long double a=20.): Super(nreg, b, a, 254) {}
};
struct ShortSetS: public SetSketch<uint16_t, long double> {
    ShortSetS(int nreg, long double b=1.001, long double a=.25): SetSketch<uint16_t, long double>(nreg, b, a, 65534u) {}
};
struct WideShortSetS: public SetSketch<uint16_t, long double> {
    WideShortSetS(int nreg, long double b=1.00095, long double a=.03): SetSketch<uint16_t, long double>(nreg, b, a, 65534u) {}
};


} // namespace sketch

#endif
