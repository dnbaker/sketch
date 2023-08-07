#ifndef SKETCH_BAGMINHASH_H__
#define SKETCH_BAGMINHASH_H__
#include <stdexcept>
#include <cassert>
#include <queue>
#include "sketch/macros.h"
#include "sketch/fy.h"
#include "aesctr/wy.h"
#include "sketch/div.h"
#include "sketch/flog.h"
#include "sketch/kahan.h"
#include "sketch/hash.h"
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {

namespace kahan_detail {
    template<typename T>
    INLINE T kahan_update(T &sum, T &carry, T increment) {
        increment -= carry;
        T tmp = sum + increment;
        carry = (tmp - sum) - increment;
        return sum = tmp;
    }
}


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

    int update(size_t index, FT x) {
        const auto sz = data_.size();
        const auto mv = getm();
        if(x == data_[index]) return -1;
        if(x < data_[index]) {
            do {
                data_[index] = x;
                index = mv + (index >> 1);
                if(index >= sz) break;
                size_t lhi = (index - mv) << 1;
                size_t rhi = lhi + 1;
                x = std::max(data_[lhi], data_[rhi]);
            } while(x < data_[index]);
            return 1;
        }
        return 0;
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
    static_assert(std::is_integral<IT>::value || sizeof(FT) > 8, "Sanity check");


    static IT ft2it(FT val) noexcept {
        IT ret = 0;
        std::memcpy(&ret, &val, sizeof(FT));
        return ret;
     }
    static constexpr IT ft2it() noexcept {
        return maxv;
    }
    static FT it2ft(IT val) noexcept {
        FT ret = 0;
        std::memcpy(&ret, &val, sizeof(FT));
        return ret;
    }
    template<typename OIT, typename=std::enable_if_t<std::is_integral<OIT>::value || std::is_same<__uint128_t, OIT>::value>>
    static FT cvt(OIT val) noexcept {return it2ft(val);}
    template<typename OFT, typename=std::enable_if_t<std::is_floating_point<OFT>::value>>
    static IT cvt(OFT val) noexcept {return ft2it(val);}
    static constexpr FT maxv = std::numeric_limits<FT>::max();
    static constexpr IT maxi = cvt(maxv);
    using IntType = IT;
};

template<typename FT, typename IT=DefIT<FT>>
struct poisson_process_t {
    static_assert(std::is_arithmetic<FT>::value, "Must be arithmetic");
    //static_assert(std::is_integral<IT>::value, "Must be intgral");
    // Algorithm 4
    FT x_, weight_, minp_, maxq_, sum_carry_;
    IT idx_ = std::numeric_limits<IT>::max();
    uint64_t wyv_; // RNG state
    uint64_t id_;
    using wd = wd_t<FT>;
public:
    poisson_process_t(FT x, FT w, FT p, FT q, uint64_t seed, IT id, FT sum_carry):
        x_(x), weight_(w), minp_(p), maxq_(q), sum_carry_(sum_carry), wyv_(seed), id_(id)
    {
        assert(minp_ < maxq_);
    }
    poisson_process_t& operator=(poisson_process_t &&) = default;
    poisson_process_t& operator=(const poisson_process_t &) = default;
    poisson_process_t(const poisson_process_t &o) = default;
    poisson_process_t(poisson_process_t &&o) = default;
    poisson_process_t(IT id, FT w): x_(0.), weight_(w), minp_(0.), maxq_(std::numeric_limits<FT>::max()), sum_carry_(0.), wyv_(id), id_(id) {
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
        return maxq_ <= weight_;
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
        kahan::update(x_, sum_carry_, -std::log((xi >> 12) * FT(2.220446049250313e-16)) / (maxq_ - minp_));
        idx_ = fastmod.mod(xi);
    }
    void step(size_t m) {
        uint64_t xi = wy::wyhash64_stateless(&wyv_);
        kahan::update(x_, sum_carry_, -std::log((xi >> 12) * FT(2.220446049250313e-16)) / (maxq_ - minp_));
        idx_ = xi % m;
    }
    poisson_process_t split() {
        using MT = std::conditional_t<(sizeof(IT) < 8), uint64_t, IT>;
        MT midpoint = (widxmin() + widxmax()) / 2;
        FT midval = wd::cvt(midpoint);
        uint64_t rval = wy::wyhash64_stateless((uint64_t *)&midpoint);
        if(sizeof(midpoint) > 8) {
            rval ^= wy::wyhash64_stateless((uint64_t *)&midpoint + 1);
            rval += wy::wyhash64_stateless((uint64_t *)&midpoint);
        }
        uint64_t xval = wd::cvt(x_) ^ rval;
        const FT p = (midval - minp_) / (maxq_ - minp_);
        const FT rv = (wy::wyhash64_stateless(&xval) * 5.421010862427522e-20);
        //auto mynsteps = static_cast<size_t>(-1);
        const bool goleft = rv < p;
        auto oldmaxq = maxq_;
        auto oldminp = minp_;
        poisson_process_t ret(x_, weight_, goleft ? midval: oldminp, goleft ? oldmaxq: midval, xval, id_, sum_carry_);
        if(goleft)
            maxq_ = midval;
        else
            minp_ = midval;
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
    FT total_weight() const {return total_weight_;}
    uint64_t total_updates_ = 0;
    FT total_weight_ = 0., total_weight_carry_ = 0.;
    pq_t heap_;
    mvt_t<FT> hvals_;
    using IDType = uint64_t;
    std::vector<IDType> track_ids_;
    std::vector<FT> idcounts_;
    schism::Schismatic<std::conditional_t<(sizeof(FT) <= 8), IT, uint64_t>> div_;
    FT *data() {return hvals_.data();}
    const FT *data() const {return hvals_.data();}
    auto m() const {return hvals_.getm();}
    size_t size() const {return m();}

    uint64_t total_updates() const {return total_updates_;}
    std::vector<FT> &idcounts() {return idcounts_;}
    const std::vector<FT> &idcounts() const {return idcounts_;}
    std::vector<IDType> &ids() {return track_ids_;}
    const std::vector<IDType> &ids() const {return track_ids_;}
    bmh_t(size_t m, bool track_ids = false, bool track_idcounts = false): hvals_(m), div_(m) {
        heap_.getc().reserve(m);
        if(!track_ids && track_idcounts) {
            std::fprintf(stderr, "track idcounts implies track ids\n");
            track_ids = true;
        }
        if(track_ids)
            track_ids_.resize(m);
        if(track_idcounts)
            idcounts_.resize(m);
    }
    void hv_update(const PoissonP &pt) {
        if(hvals_.update(pt.idx_, pt.x_) && !track_ids_.empty()) {
            track_ids_[pt.idx_] = pt.id_;
            if(!idcounts_.empty())
                idcounts_[pt.idx_] = pt.weight_;
        }
    }
    void update_2(IDType id, FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        kahan::update(total_weight_, total_weight_carry_, w);
        PoissonP p(id, w);
        p.step(div_);
        if(p.maxq_ <= p.weight_)
            hv_update(p);
        auto &tmp = heap_.getc();
        const size_t offset = tmp.size();
        while(p.x_ < hvals_.max()) {
            while(p.can_split() && p.partially_relevant()) {
                auto pp = p.split();
                if(p.fully_relevant()) hv_update(p);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hv_update(pp);
                    if(pp.partially_relevant()) {
                        tmp.emplace_back(std::move(pp));
                        std::push_heap(tmp.begin() + offset, tmp.end());
                    }
                }
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hv_update(p);
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
                if(p.fully_relevant()) hv_update(p);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hv_update(pp);
                    if(pp.x_ <= hvals_.max()) {
                        tmp.emplace_back(std::move(pp));
                        std::push_heap(tmp.begin(), tmp.end());
                    }
                }
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hv_update(p);
                if(p.x_ <= hvals_.max()) {
                    tmp.emplace_back(std::move(p));
                    std::push_heap(tmp.begin(), tmp.end());
                }
            }
        }
    }
    void update_1(IDType id, FT w) {
        if(w <= 0.) return;
        ++total_updates_;
        kahan::update(total_weight_, total_weight_carry_, w);
        PoissonP p(id, w);
        p.step(div_);
        if(p.fully_relevant()) hv_update(p);
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
                    hv_update(p);
                if(pp.partially_relevant()) {
                    pp.step(div_);
                    if(pp.fully_relevant()) hv_update(pp);
                    if(pp.partially_relevant()) heap_.push(std::move(pp));
                }
                //std::fprintf(stderr, "Finishing subloop at %zu/%zu\n", mainiternum, subin);
            }
            if(p.fully_relevant()) {
                p.step(div_);
                hv_update(p);
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
    void reset() {
        hvals_.reset();
        heap_.clear();
        total_updates_ = 0;
        total_weight_carry_ = total_weight_ = 0.;
    }
};
template<typename FT>
struct BagMinHash1: public bmh_t<FT> {
    template<typename...Args> BagMinHash1(Args &&...args): bmh_t<FT>(std::forward<Args>(args)...) {}
    FT *data() {return this->hvals_.data();}
    const FT *data() const {return this->hvals_.data();}
    template<typename IT>
    void add(IT id, FT w) {
        this->update_1(id, w);
    }
    template<typename IT> void update(IT id, FT w) {add(id, w);}
    void finalize() {}
};
template<typename FT>
struct BagMinHash2: public bmh_t<FT> {
    using S = bmh_t<FT>;
    FT *data() {return this->hvals_.data();}
    const FT *data() const {return this->hvals_.data();}
    BagMinHash2(size_t m, bool track_ids=false, bool track_counts=false): S(m, track_ids, track_counts) {}
    template<typename IT>
    void add(IT id, FT w) {
        this->update_2(id, w);
    }
    template<typename IT> void update(IT id, FT w) {add(id, w);}
    void finalize() {S::finalize_2();}
};


template<typename FT=double>
struct pmh1_t {
    using wd = wd_t<FT>;
    using IT = typename wd::IntType;

    double total_weight_ = 0., total_weight_carry_ = 0.;
    double total_weight() const {return total_weight_;}
    mvt_t<FT> hvals_;
    schism::Schismatic<IT> div_;
    std::vector<IT> res_;
    pmh1_t(size_t m): hvals_(m), div_(m), res_(m) {}
    uint64_t total_updates_ = 0;

    uint64_t total_updates() const {return total_updates_;}
    void finalize() const {}
    void reset() {
        hvals_.reset();
        std::fill(res_.begin(), res_.end(), IT(0));
        total_weight_ = total_weight_carry_ = 0.;
    }
    void update(const IT id, const FT w) {
        if(w <= 0.) return;
        kahan::update(total_weight_, total_weight_carry_, w);
        FT carry = 0.;
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
            kahan::update(hv, carry, -std::log(xi * FT(5.421010862427522e-20)) * wi);
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

template<typename FT=long double>
static inline FT compute_truncexp_lambda(size_t m) {
    return static_cast<FT>(std::log1p(1.L / (m - size_t(1))));
}
template<typename FT=long double>
static inline std::array<FT, 5> compute_truncexp_constants(size_t m) noexcept {
    const long double lambda = compute_truncexp_lambda<long double>(m);
    const long double c1 = (std::exp(lambda) - 1.L) / lambda;
    const long double c2 = std::log(2.L / (1.L + std::exp(-lambda))) / lambda;
    const long double c3 = (1.L - std::exp(-lambda)) / lambda;
    const long double c4 = c1 * lambda;
    return std::array<FT, 5>{FT(lambda), FT(c1), FT(c2), FT(c3), FT(c4)};
}

template<typename FT> INLINE FT truncexp_update(uint64_t& state) noexcept {
    return static_cast<std::common_type_t<FT, double>>(0x1p-64) * wy::wyhash64_stateless(&state);
}
template<typename FT>
static INLINE FT truncexpsamplestepped(uint64_t rngstate, const FT lambda, const FT c1, const FT c2, const FT c3, const FT c4) noexcept {
    // Func should update the x and return the random value sampled
    static constexpr FT one = static_cast<FT>(1);
    FT x = (0x1p-64 * rngstate) * c1;
    if(x >= one) {
        for(;;) {
            if((x = truncexp_update<FT>(rngstate)) < c2) break;
            FT yhat = static_cast<FT>(0.5) * truncexp_update<FT>(rngstate);
            if(const FT one_minus_x = one - x;yhat > one_minus_x) {
                x = one_minus_x; yhat = one - yhat;
            }
            const FT one_minus_x = one - x;
            if(x <= c3 * (one - yhat) || (yhat * c1 <= one_minus_x)) break;
            if(std::fma(yhat, c4, one) <= std::exp(lambda * one_minus_x)) break;
        }
    }
    return x;
}
template<typename FT>
static INLINE FT truncexpsample(const uint64_t &src, const FT lambda, const FT c1, const FT c2, const FT c3, const FT c4) noexcept {
    uint64_t src0(src);
    return truncexpsamplestepped(wy::wyhash64_stateless(&src0), lambda, c1, c2, c3, c4);
}
template<typename FT>
static INLINE FT truncexpsample(uint64_t &src, const FT lambda, const FT c1, const FT c2, const FT c3, const FT c4) noexcept {
    return truncexpsamplestepped(wy::wyhash64_stateless(&src), lambda, c1, c2, c3, c4);
}
template<typename FT>
static INLINE FT truncexpsample(const uint64_t &rngstate, const std::array<FT, 5> &c) noexcept {
    return truncexpsample(rngstate, c[0], c[1], c[2], c[3], c[4]);
}
template<typename FT>
static INLINE FT truncexpsample(uint64_t &rngstate, const std::array<FT, 5> &c) noexcept {
    return truncexpsample(rngstate, c[0], c[1], c[2], c[3], c[4]);
}


template<typename FT=double, typename IdxT=uint32_t>
struct pmh2_t {
    using wd = wd_t<FT>;
    using IT = typename wd::IntType;

    uint64_t total_updates_ = 0;
    FT total_weight_ = 0., total_weight_carry_ = 0.;
    mvt_t<FT> hvals_;
    schism::Schismatic<IdxT> div_;
    using IDType = uint64_t;
    std::vector<IDType> res_;
    std::vector<FT> resweights_;
    fy::LazyShuffler ls_;
    FT total_weight() const {return total_weight_;}
    pmh2_t(size_t m, bool track_counts=true): hvals_(m), div_(m), res_(m), ls_(m) {
        if(m > std::numeric_limits<IdxT>::max()) throw std::invalid_argument("pmh2 requires a larger integer type to sketch.");
        if(track_counts) resweights_.resize(m);
    }
    std::vector<FT> &idcounts() {return resweights_;}
    const std::vector<FT> &idcounts() const {return resweights_;}
    void reset() {
        hvals_.reset();
        std::fill(res_.begin(), res_.end(), IT(0));
        std::fill(resweights_.begin(), resweights_.end(), FT(0));
        total_updates_ = 0;
        total_weight_carry_ = total_weight_ = 0.;
    }
    const std::vector<IDType> &ids() const {return res_;}
    std::vector<IDType> &ids() {return res_;}

    void finalize() const {}
    static constexpr FT beta(size_t idx, size_t m) {
        const double rs = m;
        return rs / (rs - idx + 1.);
    }
    FT *data() {return hvals_.data();}
    const FT *data() const {return hvals_.data();}
    uint64_t total_updates() const {return total_updates_;}
    FT getbeta(size_t idx) const {return beta(idx, ls_.size());}
    void update(const uint64_t id, const FT w) {
        using fastlog::flog;
        if(w <= 0.) return;
        kahan::update(total_weight_, total_weight_carry_, w);
        FT carry = 0.;
        ++total_updates_;
        uint64_t hi = id;
        const FT wi = 1. / w, m_double = ls_.size();
        size_t i = 0;
        uint64_t rv = wy::wyhash64_stateless(&hi);
        auto maxv = hvals_.max();
        FT hv;
        CONST_IF(sizeof(FT) <= 8) {
            FT frv = rv * 5.421010862427522e-20;
            if(flog(frv) * wi * FT(0.7) > maxv) return;
            hv = -std::log(frv) * wi;
        } else {
            hv = -std::log(((__uint128_t(rv) << 64) | hi) * 2.9387358770557187699e-39L) * wi;
        }
        if(hv >= maxv) return;
        for(ls_.reset(), ls_.seed(rv);hv < maxv;) {
            auto idx = ls_.step();
            if(hvals_.update(idx, hv)) {
                maxv = hvals_.max();
                res_[idx] = id;
                resweights_[idx] = w;
                if(hv >= maxv) return;
            }
            // Beta = m_double / (m_double - i + 1)
            // const auto wb = getbeta(i++) * wi;
            const FT wb = m_double / (m_double - i++ + 1.);
            CONST_IF(sizeof(FT) <= 8) {
                const FT frv = wy::wyhash64_stateless(&hi) * 5.421010862427522e-20;
                kahan::update(hv, carry, -std::log(frv) * wb);
            } else {
                kahan::update(hv, carry, static_cast<FT>(-std::log(((__uint128_t(rv) << 64) | hi) * 2.9387358770557187699e-39L) * wb));
            }
        }
    }
    pmh2_t &operator+=(const pmh2_t &o) {
        if(size() != o.size()) throw std::invalid_argument("Mismatched sizes");
        if(!resweights_.empty() != o.resweights_.empty()) throw std::invalid_argument("Mismatched counting");
        const bool use_counts = !resweights_.empty();
        total_weight_ += o.total_weight_;
        total_weight_carry_ += o.total_weight_carry_;
        for(size_t i = 0; i < m(); ++i) {
            if(hvals_[i] == o.hvals_[i]) {
                if(use_counts) resweights_[i] += o.resweights_[i];
            } else if(hvals_.update(i, o.hvals_[i])) {
                if(use_counts) resweights_[i] = o.resweights_[i];
            }
        }
        return *this;
    }
    void add(const IT id, const FT w) {update(id, w);}
    size_t m() const {return res_.size();}
    size_t size() const {return m();}
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
template<typename FT=double, typename IdxT=uint32_t>
struct pmh3_t: public pmh2_t<FT, IdxT> {
    using base_t = pmh2_t<FT, IdxT>;
    const std::array<FT, 5> CONSTANTS;
    template<typename...Args> pmh3_t(Args &&...args): base_t(std::forward<Args>(args)...), CONSTANTS(compute_truncexp_constants<FT>(this->size())) {
        if(this->size() < 2)
            throw std::invalid_argument("ProbMinHash3 (pmh3_t) requires M >= 2");
    }
    INLINE FT steptrunc(uint64_t *src) const {
        return steptrunc(wy::wyhash64_stateless(src));
    }
    INLINE FT steptrunc(uint64_t src) const {
        return truncexpsamplestepped(src, CONSTANTS[0], CONSTANTS[1], CONSTANTS[2], CONSTANTS[3], CONSTANTS[4]);
    }
    void update(const uint64_t id, const FT w) {
        using fastlog::flog;
        if(w <= 0.) return;
        kahan::update(this->total_weight_, this->total_weight_carry_, w);
        ++this->total_updates_;
        uint64_t hi = id;
        const FT wi = 1. / w;
        size_t i = 0;
        uint64_t rv = wy::wyhash64_stateless(&hi);
        auto maxv = this->hvals_.max();
        FT hv = wi * steptrunc(rv); // Since we already generated the point, don't use this.
        if(hv >= maxv) return;
        for(this->ls_.reset(), this->ls_.seed(rv);hv < maxv;) {
            const auto idx = this->ls_.step();
            if(this->hvals_.update(idx, hv)) {
                this->res_[idx] = id;
                this->resweights_[idx] = w;
                maxv = this->hvals_.max();
                if(hv >= maxv) return;
            }
            if((hv = wi * ++i) > maxv) return;
            hv = std::fma(wi, steptrunc(&rv), hv);
        }
    }
};
template<typename FT, typename IdxT> using ProbMinHash = pmh3_t<FT, IdxT>;

template<typename T, int64_t N, int64_t NUM=1, int64_t DENOM=1>
static constexpr std::array<T, N> generate_power_lut() {
    const long double MUL = static_cast<long double>(NUM) / DENOM;
    std::array<T, N> ret{};
    long double current = MUL;
    for(int32_t i = 1; i < N; ++i) {
        ret[i] = current;
        current *= MUL;
    }
    return ret;
}

template<typename T, int64_t N>
static constexpr std::array<T, N> generate_poisson_cdf() {
    std::array<T, N> ret{};
    long double pdf = 0.367879441171442321610162567991L;
    long double cdf = pdf;
    for(int64_t i = 0; i < N; i++) {
        ret[i] = cdf;
        pdf = pdf/(i + 1);
        cdf += pdf;
    }
    return ret;
}


// Based on dartminhash https://github.com/tobc/dartminhash
// Advantages: fast weighted sketching [up to 64x faster, typically 16-30x than bagminhash]
// Disadvantages: not streaming. Needs up-front total weight calculation to select RHO value.
//                Additionally, it is a Las Vegas algorithm; IE, it is repeated until success.
//                This means there is no way around iterating through the data multiple times.
template<typename Hash=hash::CEHasher>
class DartHash {

    uint64_t t;
    // Seeds
    uint64_t T_nu, T_rho, T_w, T_r;
    uint64_t T_i, T_p, T_q, F, M;
    // Hasher
    const Hash hasher;

    // Compile-time generate to simplify.
    // Standard double precision ranges from around +-2^1024.
    static constexpr std::array<double, 1024> powers_of_two = generate_power_lut<double, 1024, 2LL, 1LL>();
    static constexpr std::array<double, 1024> negative_powers_of_two = generate_power_lut<double, 1024, 1LL, 2LL>();
    static constexpr std::array<double, 128> poisson_cdf = generate_poisson_cdf<double, 128>();


    uint64_t hash(const uint64_t seed, const uint64_t seed2) const noexcept {
        return hasher(hasher(seed) ^ seed2);
    }

public:
    template<typename...Args>
    DartHash(uint64_t seed, uint64_t t, Args &&...args) : t(t), hasher(std::forward<Args>(args)...) {
        wy::WyRand rng(seed);
        T_nu = rng();
        T_rho = rng();
        T_r = rng();
        T_i = rng();
        T_p = rng();
        T_q = rng();
        F = rng();
        M = rng();
    }

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& x, double weight_sum=-1., const double theta = 1.0) const {
        std::vector<std::pair<uint64_t, double>> darts;
        darts.reserve(2*t);
        if(weight_sum < 0) {
            weight_sum = std::accumulate(std::cbegin(x), std::cend(x), 1e-40, [](double sum, const auto& pair) {return sum += pair.second;});
        }
        const double max_rank = theta / weight_sum; // Add 1e-40 to the weight sum to avoid dividing by 0.
        const double t_inv = 1.0/t;
        const int32_t RHO = (int32_t)std::floor(std::log2(1.0 + max_rank));

        for(const std::pair<uint64_t, double>& element : x) {
            uint64_t i = element.first;
            double xi = element.second;
            uint64_t i_hash = hash(T_i, i);
            const int32_t NU = std::floor(std::log2(1.0 + t*xi));
            for(int32_t nu = 0; nu <= NU; nu++) {
                uint64_t nu_hash = hash(T_nu, nu);
                for(int32_t rho = 0; rho <= RHO; rho++) {
                    uint64_t region_hash = nu_hash ^ hash(T_rho, rho);
                    const double two_nu = powers_of_two[nu];
                    const double two_rho = powers_of_two[rho];
                    const double W = (two_nu - 1)*t_inv;
                    const double R = two_rho - 1;
                    const double delta_nu = two_nu*t_inv*negative_powers_of_two[rho];
                    const double delta_rho = two_rho*negative_powers_of_two[nu];
                    double w0 = W;
                    double w0_carry = 0.;
                    uint32_t w_max = (1ul << std::min(rho, 31));
                    for(uint32_t w = 0; w < w_max; w++) {
                        if(xi < w0) break;
                        const uint64_t w_hash = hash(T_w, w);
                        double r0 = R;
                        double r0_carry = 0.;
                        uint32_t r_max = 1ul << std::min(31, nu);
                        for(uint32_t r = 0; r < r_max; r++) {
                            if(max_rank < r0) break;
                            // Get area fingerprint to speed up subsequent hashing
                            const uint64_t area_hash = w_hash ^ hash(T_r, r);
                            const uint64_t z = i_hash ^ region_hash ^ area_hash;

                            // Draw from Poisson distribution
                            const double p_z = 0x1p-64 * hash(T_p, z);
                            const uint8_t p = std::find_if_not(poisson_cdf.begin(), poisson_cdf.end(), [p_z](const auto x) {return p_z > x;}) - poisson_cdf.begin();

                            uint64_t q = 0;
                            while(q < p) {
                                // Layer the q-values over z to create a unique key with a strong hash value
                                const uint64_t z_q = z ^ (q << 56) ^ (q << 48) ^ (q << 40) ^ (q << 32) ^ (q << 24) ^ (q << 16) ^ (q << 8) ^ q;
                                const uint64_t uniform_weight_hash = hash(T_q, z_q);
                                const double first_weight_rank = (uniform_weight_hash >> 32) * 0x1p-32;
                                const double second_weight_rank = (uniform_weight_hash & 0xFFFFFFFF) * 0x1p-32;
                                const double weight = std::fma(delta_nu, first_weight_rank, w0);
                                const double rank = std::fma(delta_rho, second_weight_rank, r0);
                                if(weight < xi && rank < max_rank) {
                                    darts.push_back({hash(F, z_q), rank});
                                }
                                q++;
                            }

                            r0 += delta_rho;

                            kahan::update(r0, r0_carry, delta_rho);
                        }
                        kahan::update(w0, w0_carry, delta_nu);
                    }
                }
            }
        }
        return darts;
    }


    // Convert the t darts to k minhashes by hashing the darts to k buckets and keeping the minimum from each bucket
    std::vector<std::pair<uint64_t, double>> minhash(const std::vector<std::pair<uint64_t, double>>& x, uint64_t k) {
        auto darts = (*this)(x);
        std::vector<std::pair<uint64_t, double>> minhashes(k, {0, std::numeric_limits<double>::max()});
        for(auto& dart : darts) {
            uint64_t j = M(dart.first) % k;
            if(dart.second < minhashes[j].second) {
                minhashes[j] = dart;
            }
        }
        return minhashes;
    }
};

template<typename RNG=wy::WyRand<>>
class DartMinHash {
    const uint64_t size_;
    const uint64_t seed_;
    DartHash<RNG> D;

public:
    // Set t = k ln(k) + 2k so the probability of failing on the first run is at most exp(-2)
    DartMinHash(const uint64_t seed, const int64_t size) : size_(size), seed_(seed), D(seed, std::ceil(size*std::log(size) + 2*size)) {};

    std::vector<std::pair<uint64_t, double>> operator()(const std::vector<std::pair<uint64_t, double>>& x) {
        bool all_minhashed = false;
        double theta = 1.0;
        std::vector<std::pair<uint64_t, double>> minhashes(size_, {0, std::numeric_limits<double>::max()});
        schism::Schismatic<uint64_t> div(size_);
        while(!all_minhashed) {
            std::vector<bool> minhashed(size_, false);
            auto darts = D(x, theta);
            // Place darts into buckets
            for(auto& dart : darts) {
                uint64_t j = div.mod(T(dart.first));
                minhashed[j] = true;
                if(dart.second < minhashes[j].second) {
                    minhashes[j] = dart;
                }
            }
            // Verify whether all minhashes were computed
            all_minhashed = std::accumulate(minhashed.begin(), minhashed.end(), true, std::logical_and<>{});

            theta = theta + 0.5;
        }
        return minhashes;
    }
};


} // namespace wmh
using wmh::mh2str;
using wmh::pmh2_t;
using wmh::pmh3_t;
using wmh::BagMinHash1;
using wmh::BagMinHash2;
using wmh::ProbMinHash;
using wmh::compute_truncexp_lambda;
using wmh::compute_truncexp_constants;
using wmh::truncexpsample;

namespace omh {
template<typename FT=double>
struct OMHasher {
private:
    size_t m_, l_;
    std::vector<uint64_t> indices;
    std::vector<FT>          vals;
    wmh::mvt_t<FT>            mvt;
    ska::flat_hash_map<uint64_t, uint32_t> counter;
    fy::LazyShuffler ls_;

    bool sub_update(const uint64_t pos, const FT value, const uint64_t element_idx) {
        const auto start = l_ * pos;
        const auto stop = start + l_ - 1;
        if(value >= vals[stop]) {
            return false;
        }
        uint64_t ix = stop;
        for(ix = stop;ix > start && value < vals[ix - 1]; --ix) {
            vals[ix] = vals[ix - 1];
            indices[ix] = indices[ix - 1];
        }
        vals[ix] = value;
        indices[ix] = element_idx;
        return mvt.update(pos, vals[stop]);
    }

    void update(const uint64_t item, const uint64_t item_index) {
        FT carry = 0.;
        uint64_t rng = item;
        uint64_t hv = wy::wyhash64_stateless(&rng);
        auto it = counter.find(hv);
        if(it == counter.end()) it = counter.emplace(hv, 1).first;
        else ++it->second;
        rng ^= it->second;
        uint64_t rv = wy::wyhash64_stateless(&rng); // RNG with both item and count

        FT f = -std::log(rv * 0x1p-64);
        ls_.reset();
        ls_.seed(rv);
        uint32_t n = 0;
        for(;;) {
            const uint32_t idx = ls_.step();
            sub_update(idx, f, item_index);
            if(f >= mvt.max()) {
                std::fprintf(stderr, "f: %g. max: %g\n", f, mvt.max());
                break;
            }
            if(n == m_ + 1) {
                // This should never happen
                std::fprintf(stderr, "n = m * 2: %u/%u\n", n, int(m_));
                std::exit(1);
                throw std::runtime_error("Failed to update sketch properly.\n");
                break;
            }
            ++n;
            const uint64_t rv = wy::wyhash64_stateless(&rng);
            const double rd = (rv + 1) * 0x1p-64;
            const FT inc = -std::log(rd) * (static_cast<double>(m_) / (m_ - n));
            kahan::update(f, carry, inc);
            if(f >= mvt.max()) {
                break;
            }
        }
    }
public:
    OMHasher(size_t m, size_t l)
        : m_(m), l_(l), indices(m_ * l_), vals(m_ * l_), mvt(m), ls_(m)
    {
        reset();
    }
    template<typename...ARG>
    decltype(auto) operator()(ARG &&...args) {return hash(std::forward<ARG>(args)...);}

    template<typename T>
    std::vector<uint64_t> hash(const T *ptr, size_t n) {
        reset();
        for(size_t i = 0; i < n; ++i) {
            update(ptr[i], i);
        }
        return finalize(ptr);
    }
    template<typename T, typename Traits, typename Alloc>
    std::vector<uint64_t> hash(const std::basic_string<T, Traits, Alloc> &o) {return hash(o.data(), o.size());}
    template<typename T, typename Alloc>
    std::vector<uint64_t> hash(const std::vector<T, Alloc> &ptr) {
        return hash(ptr.data(), ptr.size());
    }

    size_t m() const {return m_;}
    size_t l() const {return l_;}

    template<typename T>
    std::vector<uint64_t> finalize(const T *data) const {
        assert(indices.size() == m_ * l_);
        if(*std::max_element(indices.begin(), indices.end()) == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("Cannot complete an OrderMinHash with infinite indexes selected. This is either a bug or you are finalizing an item built from an empty string.");
        }
        std::vector<uint64_t> ret(m_);
        std::vector<T> up(l_);
        XXH3_state_t state;
        for(size_t i = 0; i < m_; ++i) {
            auto indptr = &indices[l_ * i];
            std::copy(indptr, indptr + l_, up.begin());
            std::sort(up.begin(), up.end());
            XXH3_64bits_reset(&state);
            for(size_t j = 0; j < l_; ++j) {
                XXH3_64bits_update(&state, data + up[j], sizeof(T));
            }
            ret[i] = XXH3_64bits_digest(&state);
        }
        return ret;
    }

    void reset() {
        std::fill(vals.begin(), vals.end(), std::numeric_limits<FT>::max());
        std::fill(indices.begin(), indices.end(), std::numeric_limits<uint64_t>::max());
        mvt.reset();
        counter.clear();
    }

};

}

} // namespace sketch

#endif // #ifndef SKETCH_BAGMINHASH_H__
