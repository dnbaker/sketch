#ifndef SKETCH_BAGMINHASH_H__
#define SKETCH_BAGMINHASH_H__
#include <stdexcept>
#include <cassert>
#include "aesctr/wy.h"
#include <queue>
#include <unordered_map>
#include "fy.h"
#include "macros.h"
#include "div.h"

namespace sketch {


namespace wmh {

enum Sketchers {
    S_BMH1,
    S_BMH2,
    S_PMH1,
    S_PMH1A,
    S_PMH2
};
static constexpr const char *mh2str(Sketchers s) {
    switch(s) {
#define __C(x) case x: return #x;
        __C(S_BMH1)
        __C(S_BMH2)
        __C(S_PMH2)
        __C(S_PMH1)
        __C(S_PMH1A)
#undef __C
    }
    return "unknown";
}



template<typename FT>
struct mvt_t {
    // https://arxiv.org/pdf/1802.03914v2.pdf, algorithm 5,
    // and https://arxiv.org/pdf/1911.00675.pdf, algorithm 4
    std::vector<FT> data_;
    mvt_t(size_t m, const FT maxv=std::numeric_limits<FT>::max()): data_((m << 1) - 1, maxv)
    {
        assert(getm() == m);
    }


    FT *data() {return data_.data();}
    const FT *data() const {return data_.data();}
    // Check size and max
    size_t getm() const {return (data_.size() >> 1) + 1;}
    FT max() const {return data_.back();}
    FT operator[](size_t i) const {return data_[i];}
    void reset() {
        std::fill(data_.begin(), data_.end(), std::numeric_limits<FT>::max());
    }

    bool update(size_t index, FT x) {
        const auto sz = data_.size();
        const auto mv = getm();
        if(x < data_[index]) {
            do {
                data_[index] = x;
                index = mv + (index >> 1);
                if(index >= sz) break;
                size_t lhi = (index - mv) << 1;
                size_t rhi = lhi + 1;
                x = std::max(data_[lhi], data_[rhi]);
            } while(x < data_[index]);
            return true;
        }
        return false;
    }
};

template<typename FT>
using DefIT = std::conditional_t<sizeof(FT) == 4, uint32_t,
               std::conditional_t<sizeof(FT) == 8, uint64_t,
               std::conditional_t<sizeof(FT) == 2, uint16_t,
               std::conditional_t<sizeof(FT) == 1, uint8_t,
               std::conditional_t<sizeof(FT) == 16, __uint128_t,
               void>>>>>;

template<typename FT>
struct wd_t {
    using IT = DefIT<FT>;
    static_assert(std::is_integral<IT>::value || std::is_same<IT, __uint128_t>::value, "Sanity check");

    union ITFTU {
        IT i_; FT f_;
        ITFTU(): i_(0) {}
    };
    static constexpr IT ft2it(FT val=std::numeric_limits<FT>::max()) {
        ITFTU tmp;
        tmp.f_ = val;
        return tmp.i_;
    }
    static constexpr FT it2ft(IT val) {
        ITFTU tmp;
        tmp.i_ = val;
        return tmp.f_;
    }
    template<typename OIT, typename=std::enable_if_t<std::is_integral<OIT>::value>>
    static constexpr FT cvt(OIT val) {return it2ft(val);}
    template<typename OFT, typename=std::enable_if_t<std::is_floating_point<OFT>::value>>
    static constexpr IT cvt(OFT val) {return ft2it(val);}
    static constexpr FT maxv = std::numeric_limits<FT>::max();
    static const IT maxi = cvt(maxv);
    using IntType = IT;
};

template<typename FT, typename IT=DefIT<FT>>
struct poisson_process_t {
    static_assert(std::is_arithmetic<FT>::value, "Must be arithmetic");
    static_assert(std::is_integral<IT>::value, "Must be intgral");
    // Algorithm 4
    FT x_, weight_, minp_, maxq_;
    IT idx_ = std::numeric_limits<IT>::max();
    uint64_t wyv_; // RNG state
    using wd = wd_t<FT>;
public:
    poisson_process_t(FT x, FT w, FT p, FT q, uint64_t seed):
        x_(x), weight_(w), minp_(p), maxq_(q), wyv_(seed)
    {
        assert(minp_ < maxq_);
    }
    poisson_process_t& operator=(poisson_process_t &&) = default;
    poisson_process_t& operator=(const poisson_process_t &) = default;
    poisson_process_t(const poisson_process_t &o) = default;
    poisson_process_t(poisson_process_t &&o) = default;
    poisson_process_t(IT id, FT w): x_(0.), weight_(w), minp_(0.), maxq_(std::numeric_limits<FT>::max()), wyv_(id) {

    }
    IT widxmax() const {
        return wd::cvt(maxq_);
    }
    IT widxmin() const {
        return wd::cvt(minp_);
    }
    bool partially_relevant() const {
        return wd::cvt(widxmin() + 1) <= weight_;
    }
    bool fully_relevant() const {
        return wd::cvt(widxmax()) <= weight_;
    }
    bool can_split() const {
        return widxmax() > widxmin() + 1;
    }
    size_t nsteps_ = 0;
    // Note: > is reversed, for use in pq
    bool operator>(const poisson_process_t &o) const {return x_ < o.x_;}
    bool operator<(const poisson_process_t &o) const {return x_ > o.x_;}
    template<typename OIT>
    void step(const schism::Schismatic<OIT> &fastmod) {
        // Top 52-bits as U01 for exponential with weight of q - p,
        // bottom logm bits for index
        uint64_t xi = wy::wyhash64_stateless(&wyv_);
        x_ += -std::log((xi >> 12) * 2.220446049250313e-16) / (maxq_ - minp_);
        //assert(fastmod.mod(static_cast<IT>(xi)) == (static_cast<IT>(xi) % fastmod.d()) || !std::fprintf(stderr, "lhs: %zu. rhs: %zu. xi: %u. d(): %u\n", size_t(fastmod.mod(static_cast<IT>(xi))), size_t(static_cast<IT>(xi) % fastmod.d()), static_cast<IT>(xi), int(fastmod.d())));
        idx_ = fastmod.mod(xi);
    }
    void step(size_t m) {
        uint64_t xi = wy::wyhash64_stateless(&wyv_);
        x_ += -std::log((xi >> 12) * 2.220446049250313e-16) / (maxq_ - minp_);
        idx_ = xi % m;
    }
    poisson_process_t split() {
        uint64_t midpoint = (uint64_t(widxmin()) + widxmax()) / 2;
        double midval = wd::cvt(midpoint);
        uint64_t xval = wd::cvt(x_) ^ wy::wyhash64_stateless(&midpoint);
        const double p = (midval - minp_) / (maxq_ - minp_);
        const double rv = (wy::wyhash64_stateless(&xval) * 5.421010862427522e-20);
        //auto mynsteps = static_cast<size_t>(-1);
        const bool goleft = rv < p;
        auto oldmaxq = maxq_;
        auto oldminp = minp_;
        poisson_process_t ret(x_, weight_, goleft ? midval: oldminp, goleft ? oldmaxq: midval, xval);
        if(goleft) {
            maxq_ = midval;
        } else {
            minp_ = midval;
        }
        nsteps_ = -1;
        return ret;
    }
};


template<typename FT>
static inline uint64_t reg2sig(FT v) {
    uint64_t t = 0;
    std::memcpy(&t, &v, std::min(sizeof(v), sizeof(uint64_t)));
    t ^= 0xcb1eb4b41a93fe67uLL;
    return wy::wyhash64_stateless(&t);
}

template<typename FT=double>
struct bmh_t {
    using wd = wd_t<FT>;
    using IT = typename wd::IntType;
    using PoissonP = poisson_process_t<FT, IT>;

    struct pq_t: public std::priority_queue<PoissonP, std::vector<PoissonP>> {
        auto &getc() {return this->c;}
        const auto &getc() const {return this->c;}
        void clear() {
            this->c.clear();
            assert(this->size() == 0);
        }
    };
    uint64_t total_updates_ = 0;
    pq_t heap_;
    mvt_t<FT> hvals_;
    schism::Schismatic<IT> div_;
    auto m() const {return hvals_.getm();}

    uint64_t total_updates() const {return total_updates_;}
    bmh_t(size_t m): hvals_(m), div_(m) {
        heap_.getc().reserve(m);
    }
    void update_2(IT id, FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        PoissonP p(id, w);
        p.step(div_);
        if(p.fully_relevant()) hvals_.update(p.idx_, p.x_);
        auto &tmp = heap_.getc();
        const size_t offset = tmp.size();
        //size_t mainiternum = 0, subin  =0;
        while(p.x_ < hvals_.max()) {
            //++mainiternum;
            //std::fprintf(stderr, "x: %g. max: %g\n", p.x_, hvals_.max());
            while(p.can_split() && p.partially_relevant()) {
                //std::fprintf(stderr, "min %g max %g, splitting!\n", p.minp_, p.maxq_);
                auto pp = p.split();
                if(p.fully_relevant())
                    hvals_.update(p.idx_, p.x_);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hvals_.update(pp.idx_, pp.x_);
                    if(pp.partially_relevant()) {
                        tmp.emplace_back(std::move(pp));
                        std::push_heap(tmp.begin() + offset, tmp.end());
                    }
                }
                //std::fprintf(stderr, "Finishing subloop at %zu/%zu\n", mainiternum, subin);
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hvals_.update(p.idx_, p.x_);
                if(p.x_ <= hvals_.max()) {
                    tmp.emplace_back(std::move(p));
                    std::push_heap(tmp.begin() + offset, tmp.end());
                }
            }
            if(tmp.size() == offset) break;
            std::pop_heap(tmp.begin() + offset, tmp.end());
            p = std::move(tmp.back());
            tmp.pop_back();
        }
        auto bit = tmp.begin() + offset;
        for(;bit != tmp.begin() && tmp.front().x_ > hvals_.max();--bit) {
            std::pop_heap(tmp.begin(), bit, std::greater<>());
        }
        for(auto hit = tmp.begin() + offset; hit != tmp.end(); ++hit) {
            if(hit->x_ <= hvals_.max()) {
                *bit++ = std::move(*hit);
                std::push_heap(tmp.begin(), bit, std::greater<>());
            }
        }
        tmp.erase(bit, tmp.end());
    }
    void finalize_2() {
        auto &tmp = heap_.getc();
        std::make_heap(tmp.begin(), tmp.end());
        while(tmp.size()) {
            std::pop_heap(tmp.begin(), tmp.end());
            auto p = std::move(tmp.back());
            tmp.pop_back();
            if(p.x_ > hvals_.max()) break;
            while(p.can_split() && p.partially_relevant()) {
                auto pp = p.split();
                if(p.fully_relevant()) hvals_.update(p.idx_, p.x_);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hvals_.update(pp.idx_, pp.x_);
                    if(pp.x_ <= hvals_.max()) {
                        tmp.emplace_back(std::move(pp));
                        std::push_heap(tmp.begin(), tmp.end());
                    }
                }
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hvals_.update(p.idx_, p.x_);
                if(p.x_ <= hvals_.max()) {
                    tmp.emplace_back(std::move(p));
                    std::push_heap(tmp.begin(), tmp.end());
                }
            }
        }
    }
    void update_1(IT id, FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        PoissonP p(id, w);
        p.step(div_);
        if(p.fully_relevant()) hvals_.update(p.idx_, p.x_);
        //size_t mainiternum = 0, subin  =0;
        //std::fprintf(stderr, "Updating key %zu and w %g\n", size_t(id), double(w));
        //std::fprintf(stderr, "Current max: %g\n", hvals_.max());
        while(p.x_ < hvals_.max()) {
            //++mainiternum;
            VERBOSE_ONLY(std::fprintf(stderr, "x: %0.20g. max: %0.20g\n", p.x_, hvals_.max());)
            while(p.can_split() && p.partially_relevant()) {
                VERBOSE_ONLY(std::fprintf(stderr, "min %0.20g max %0.20g, splitting!\n", p.minp_, p.maxq_);)
                auto pp = p.split();
                if(p.fully_relevant())
                    hvals_.update(p.idx_, p.x_);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hvals_.update(pp.idx_, pp.x_);
                    if(pp.partially_relevant()) heap_.push(std::move(pp));
                }
                //std::fprintf(stderr, "Finishing subloop at %zu/%zu\n", mainiternum, subin);
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hvals_.update(p.idx_, p.x_);
                if(p.x_ <= hvals_.max()) heap_.push(std::move(p));
            }
            if(heap_.empty()) break;
            p = std::move(heap_.top());
            heap_.pop();
            //std::fprintf(stderr, "heap size: %zu\n", heap_.size());
        }
        heap_.clear();
    }
    template<typename IT=FT>
    void write(std::FILE *fp) const {
        auto sigs = to_sigs<IT>();
        std::fwrite(sigs.data(), sizeof(IT), sigs.size(), fp);
    }
    void write(std::string path) const {
        std::FILE *fp = std::fopen(path.data(), "wb");
        write(fp);
        std::fclose(fp);
    }
    template<typename IT=FT>
    std::vector<IT> to_sigs() const {
        std::vector<IT> ret(m());
        if(std::is_integral<IT>::value) {
            std::transform(hvals_.data(), hvals_.data() + m(), ret.begin(), reg2sig<FT>);
        } else {
            std::copy(hvals_.data(), hvals_.data() + m(), ret.begin());
        }
        return ret;
    }
};
template<typename FT>
struct BagMinHash1: bmh_t<FT> {
    template<typename...Args> BagMinHash1(Args &&...args): bmh_t<FT>(std::forward<Args>(args)...) {}
    template<typename IT>
    void add(IT id, FT w) {
        this->update_1(id, w);
    }
    template<typename IT> void update(IT id, FT w) {add(id, w);}
    void finalize() {}
};
template<typename FT>
struct BagMinHash2: bmh_t<FT> {
    using S = bmh_t<FT>;
    BagMinHash2(size_t m): S(m) {}
    template<typename IT>
    void add(IT id, FT w) {
        S::update_2(id, w);
    }
    template<typename IT> void update(IT id, FT w) {add(id, w);}
    void finalize() {S::finalize_2();}
};


template<typename FT=double>
struct pmh1_t {
    using wd = wd_t<FT>;
    using IT = typename wd::IntType;

    mvt_t<FT> hvals_;
    schism::Schismatic<IT> div_;
    std::vector<IT> res_;
    pmh1_t(size_t m): hvals_(m), div_(m), res_(m) {}
    uint64_t total_updates_ = 0;

    uint64_t total_updates() const {return total_updates_;}
    void finalize() const {}
    void update(const IT id, const FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        const FT wi = 1. / w;
        uint64_t hi = id;
        uint64_t xi = wy::wyhash64_stateless(&hi);
        for(auto hv = -std::log(xi * 5.421010862427522e-20) * wi; hv < hvals_.max();) {
            auto idx = div_.mod(xi);
            if(hvals_.update(idx, hv)) {
                res_[idx] = id;
                if(hv >= hvals_.max()) break;
            }
            xi = wy::wyhash64_stateless(&hi);
            hv += -std::log(xi * 5.421010862427522e-20) * wi;
        }
    }
    void add(const IT id, const FT w) {update(id, w);}
    size_t m() const {return res_.size();}
    template<typename IT=FT>
    std::vector<IT> to_sigs() const {
        std::vector<IT> ret(m());
        if(std::is_integral<IT>::value) {
            std::transform(hvals_.data(), hvals_.data() + m(), ret.begin(), reg2sig<FT>);
        } else {
            std::copy(hvals_.data(), hvals_.data() + m(), ret.begin());
        }
        return ret;
    }
};

namespace fastlog {
    static inline long double flog(long double x) {
        __uint128_t yi;
        std::memcpy(&yi, &x, sizeof(x));
        return yi * 3.7575583950764744255e-20L - 11356.176832703863597L;
    }
    static inline double flog(double x) {
        uint64_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 1.539095918623324e-16 - 709.0895657128241;
    }
    static inline float flog(float x) {
        uint32_t yi;
        std::memcpy(&yi, &x, sizeof(yi));
        return yi * 8.2629582881927490e-8f - 88.02969186f;
    }
}

template<typename FT=double, typename IdxT=uint32_t>
struct pmh2_t {
    using wd = wd_t<FT>;
    using IT = typename wd::IntType;

    uint64_t total_updates_ = 0;
    mvt_t<FT> hvals_;
    schism::Schismatic<IdxT> div_;
    std::vector<IT> res_;
    fy::LazyShuffler ls_;
    pmh2_t(size_t m): hvals_(m), div_(m), res_(m), ls_(m) {
        if(m > std::numeric_limits<IdxT>::max()) throw std::invalid_argument("pmh2 requires a larger integer type to sketch.");
    }

    void finalize() const {}
    static constexpr FT beta(size_t idx, size_t m) {
        const double rs = m;
        return rs / (rs - idx + 1.);
    }
    uint64_t total_updates() const {return total_updates_;}
    FT getbeta(size_t idx) const {return beta(idx, ls_.size());}
    void update(const IT id, const FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        uint64_t hi = id;
        const FT wi = 1. / w;
        size_t i = 0;
        uint64_t rv = wy::wyhash64_stateless(&hi);
        auto maxv = hvals_.max();
        FT hv;
        CONST_IF(sizeof(FT) <= 8) {
            FT frv = rv * 5.421010862427522e-20;
            if(fastlog::flog(frv) * wi * FT(0.7) > maxv) return;
            hv = -std::log(frv) * wi;
        } else {
            hv = -std::log(((__uint128_t(rv) << 64) | hi) * 2.9387358770557187699e-39L) * wi;
        }
        if(hv >= maxv) return;
        ls_.reset();
        ls_.seed(rv);
        do {
            auto idx = ls_.step();
            if(hvals_.update(idx, hv)) {
                maxv = hvals_.max();
                res_[idx] = id;
                if(hv >= maxv) return;
            }
            CONST_IF(sizeof(FT) <= 8) {
                auto bv = getbeta(i);
                const FT wbv = bv * wi; // weight inverse * beta
                const FT frv = wy::wyhash64_stateless(&hi) * 5.421010862427522e-20;
                if(hv + fastlog::flog(frv) * 0.7 * wbv > maxv) return;
                hv += -std::log(frv) * wbv;
            } else {
                hv += -std::log(((__uint128_t(rv) << 64) | hi) * 2.9387358770557187699e-39L) * wi * getbeta(i);
            }
            ++i;
        } while(hv < maxv);
    }
    void add(const IT id, const FT w) {update(id, w);}
    size_t m() const {return res_.size();}
    template<typename IT=FT>
    std::vector<IT> to_sigs() const {
        std::vector<IT> ret(m());
        if(std::is_integral<IT>::value) {
            std::transform(hvals_.data(), hvals_.data() + m(), ret.begin(), reg2sig<FT>);
        } else {
            std::copy(hvals_.data(), hvals_.data() + m(), ret.begin());
        }
        return ret;
    }
    template<typename IT=FT>
    void write(std::FILE *fp) const {
        auto sigs = to_sigs<IT>();
        std::fwrite(sigs.data(), sizeof(IT), sigs.size(), fp);
    }
    void write(std::string path) const {
        std::FILE *fp = std::fopen(path.data(), "wb");
        write(fp);
        std::fclose(fp);
    }
};


} // namespace wmh
using wmh::mh2str;
using wmh::pmh2_t;
using wmh::BagMinHash1;
using wmh::BagMinHash2;

} // namespace sketch

#endif // #ifndef SKETCH_BAGMINHASH_H__
