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

#ifndef NDEBUG
#include <unordered_set>
#endif

// Implementations of set sketch

template<typename ResT>
struct LowKHelper {
    const ResT *vals_;
    uint64_t natval_, nvals_;
    int klow_ = 0;
    void assign(const ResT *vals, size_t nvals) {
        vals_ = vals; nvals_ = nvals;
        reset();
    }
    int klow() const {return klow_;}
    void reset() {
        klow_ =  *std::min_element(vals_, vals_ + nvals_);
        size_t i;
        for(i = natval_ = 0; i < nvals_; ++i) natval_ += (vals_[i] == klow_);
    }
    void remove(int kval) {
        if(kval == klow_) {
            if(--natval_ == 0) reset();
        }
    }
};

template<typename ResT, typename FT=double>
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
    LowKHelper<ResT> lowkh_;
    static ResT *allocate(size_t n) {
        ResT *ret = nullptr;
        if(posix_memalign((void **)&ret, 64, n * sizeof(ResT))) throw std::bad_alloc();
        return ret;
    }
    FT getbeta(size_t idx) const {
        return FT(1.) / (m_ - idx);
    }
public:
    const ResT *data() const {return data_.get();}
    SetSketch(size_t m, FT b, FT a, int q): m_(m), a_(a), b_(b), ainv_(1./ a), logbinv_(1. / std::log1p(b_ - 1.)), q_(q), ls_(m_) {
        ResT *p = allocate(m_);
        data_.reset(p);
        std::fprintf(stderr, "base %g, a = %g\n", b, a);
        std::fill(p, p + m_, static_cast<ResT>(0));
        lowkh_.assign(p, m_);
    }
    size_t size() const {return m_;}
    double b() const {return b_;}
    double a() const {return a_;}
    ResT &operator[](size_t i) {return data_[i];}
    const ResT &operator[](size_t i) const {return data_[i];}
    int klow() const {return lowkh_.klow();}
    void update(uint64_t id) {
        assert(lowkh_.nvals_ == m_);
        auto mys = std::accumulate(lowkh_.vals_, lowkh_.vals_ + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());});
        assert(lowkh_.natval_ == mys);
        assert(lowkh_.natval_ == std::accumulate(lowkh_.vals_, lowkh_.vals_ + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());}));
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
            auto randv = -getbeta(bi) * ainv_ * std::log((rv >> 12) * mul);
            ev += randv;
            if(ev > std::pow(b_, -klow())) return;
            const int k = std::max(0, std::min(q_ + 1, static_cast<int>((1. - std::log(ev) * logbinv_))));
            //if(b_ >= 2.) std::fprintf(stderr, "ev: %g. k: %u. klow: %u\n", ev, k, klow());
            if(k <= klow()) return;
            auto idx = ls_.step();
#ifndef NDEBUG
            assert(idxs.find(idx) == idxs.end()); idxs.emplace(idx);
#endif
            if(k > data_[idx]) {
                auto oldv = data_[idx];
                data_[idx] = k;
                lowkh_.remove(oldv);
                assert(lowkh_.natval_ == std::accumulate(data_.get(), data_.get() + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());}));
            }
            if(++bi == m_) {
                assert(lowkh_.natval_ == std::accumulate(data_.get(), data_.get() + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());}));
                std::fprintf(stderr, "[%g] bi: %zu = m. Current klow: %u with %zu. Current %g. ev limit: %g\n", b_, bi, klow(), size_t(lowkh_.natval_), ev, std::pow(b_, -klow()));
                return;
            }
            assert(bi == idxs.size());
            //std::fprintf(stderr, "Current klow: %u, with %zu at\n", klow(), lowkh_.natval_);
            assert(lowkh_.natval_ == std::accumulate(data_.get(), data_.get() + m_, size_t(0), [&](size_t ret, auto x) {return ret + (x == lowkh_.klow());}));
            rv = wy::wyhash64_stateless(&id);
        }
    }
};

struct NibbleSetS: public SetSketch<uint8_t> {
    NibbleSetS(int nreg): SetSketch<uint8_t>(nreg, 4., 1., 14) {}
};
struct ByteSetS: public SetSketch<uint8_t> {
    ByteSetS(int nreg): SetSketch<uint8_t>(nreg, 1.2, 20., 254) {}
};
struct ShortSetS: public SetSketch<uint16_t> {
    ShortSetS(int nreg): SetSketch<uint16_t>(nreg, 1.001, 30., 65534) {}
};

#endif
