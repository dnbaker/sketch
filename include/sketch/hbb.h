#ifndef HYPERBITBIT_H__
#define HYPERBITBIT_H__
#include "sketch/common.h"
#include "sketch/hash.h"

namespace sketch {

inline namespace hbb {

/**
 *
 * HyperBitBit algorithm (c/o Sedgewick) from
 * https://www.cs.princeton.edu/~rs/talks/AC11-Cardinality.pdf
 * Based on https://github.com/thomasmueller/tinyStats/blob/master/src/main/java/org/tinyStats/cardinality/HyperBitBit.java
 */
template<typename HashStruct=hash::WangHash>
class HyperBitBit {

    uint32_t    logn_;
    uint64_t s1_, s2_;
    HashStruct    hf_;
public:
    uint64_t hash(uint64_t item) const {return hf_(item);}
    template<typename...Args>
    HyperBitBit(Args &&...args): logn_(5), s1_(0), s2_(0), hf_(std::forward<Args>(args)...) {}

    void addh(uint64_t item) {add(hash(item));}
    void add(uint64_t hv) {
        unsigned r = ctz(hv);
        if(r > logn_) {
            const auto k = (hv >> (sizeof(hv) * CHAR_BIT - 6));
            const auto bit = 1uL << k;
            s1_ |= bit;
            if (r > logn_ + 1u) s2_ |= bit;
            if(popcount(s1_) > 31)
                s1_ = s2_, s2_ = 0, ++logn_;
        }
    }

    double cardinality_estimate() const {
        //std::fprintf(stderr, "pcsum for this: %g\n", logn_ + 5.15 + popcount(s1_) / 32.);
        return std::pow(2., (logn_ + 5.8 + popcount(s1_) / 32.));
    }
    double report() const {return cardinality_estimate();}
};

struct HyperHyperBitBitSimple {
    std::vector<HyperBitBit<>> data_;
    std::vector<uint64_t> seeds_;
    HyperHyperBitBitSimple(size_t n): data_(n), seeds_(n) {
        std::mt19937_64 mt(n);
        for(auto &i: seeds_) i = mt();
    }
    void addh(uint64_t x) {
        add(x);
    }
    void add(uint64_t x) {
        data_[x % data_.size()].addh(hash::WangHash()(x ^ seeds_[x % data_.size()]));
    }
    double report() {
        double estsum = 0.;
        double harmestsum = 0.;
        std::vector<double> v;
        for(const auto &i: data_) {
            auto r = i.report();
            estsum += r;
            harmestsum += 1. / r;
            v.push_back(r);
        }
        std::sort(v.begin(), v.end());
        double oret = .5 * (v[v.size() / 2] + v[(v.size() - 1) / 2]);
        fprintf(stderr, "median: %g\n", oret);
        harmestsum = data_.size() / harmestsum;
        std::fprintf(stderr, "total sum: %g. harmestsum: %g\n", estsum, harmestsum);
        return estsum;
    }
};
struct HyperHyperBitBit {
    using lntype = uint32_t;
    uint32_t nelem_;
    std::unique_ptr<lntype[]> logns_;
    std::unique_ptr<uint64_t[]> s1s_, s2s_;
    HyperHyperBitBit(size_t n): nelem_(n), logns_(new lntype[n]), s1s_(new uint64_t[nelem_]()), s2s_(new uint64_t[nelem_]()) {
        std::fill_n(logns_.get(), nelem_, uint32_t(5));
        std::fill_n(s1s_.get(), nelem_, uint64_t(0));
        std::fill_n(s2s_.get(), nelem_, uint64_t(0));
    }
    void addh(uint64_t x) {
        wy::wyhash64_stateless(&x);
        return add(x);
    }
    void add(uint64_t v) {
        auto idx = v % nelem_;
        v /= nelem_;
        auto r = ctz(v);
        auto &logn = logns_[idx];
        if(r > logn) {
            auto bit = uint64_t(1) << ((v>>(r + 1))%64);
            auto &sketch = s1s_[idx], sketch2 = s2s_[idx];
            sketch |= bit;
            if(r > logn + 1) {
                sketch2 |= bit;
            }
            if(popcount(sketch) > 31) {
                sketch = sketch2;
                sketch2 = 0;
                ++logn;
            }
        }
    }
    double report() const {
        double pcsum = 0;
        double est_sums = 0, ies = 0., hes = 0.;
        for(size_t i = 0; i < nelem_; ++i) {
            double cinc = popcount(s1s_[i]) / 32. + 6.43 + logns_[i];
            pcsum += cinc;
            est_sums += std::pow(2., cinc);
            ies += 1. / std::pow(2., cinc);
            hes += 1. / cinc;
        }
        ies = nelem_ * nelem_ / ies;
        hes = nelem_ / hes;
        std::fprintf(stderr, "est sum: %g. iestsum: %g. hestsum: %g\n", est_sums, ies, std::pow(2., hes));
        std::fprintf(stderr, "pcsum before: %g\n", pcsum);
        pcsum /= nelem_;
        std::fprintf(stderr, "pcsum: %g\n", pcsum);
        return ies;
    }
};

} // hbb
} // sketch

#endif /* HYPERBITBIT_H__ */
