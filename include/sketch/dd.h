#ifndef DDSKETCH_H__
#define DDSKETCH_H__
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <cmath>

#include "macros.h"

namespace sketch {


inline namespace dd {

using std::size_t;
using std::int64_t;
using std::uint64_t;

// Based on implementation from https://raw.githubusercontent.com/DataDog/sketches-py/master/ddsketch/ddsketch.py
// Accessed 9/6/19

template<typename IntegerType=std::int64_t, size_t initial_nbins=128, size_t grow_left_by=128>
struct Store {

    const size_t maxbins_;
    std::vector<IntegerType> bins_; // TODO: consider using std::deque; this depends on how often it needs to shift
    size_t count_;
    uint64_t mink_, maxk_;

    using Type = IntegerType;

    Store(size_t maxnbins): maxbins_(maxnbins), bins_(initial_nbins, 0), count_(0), mink_(0), maxk_(0) {}
    Store(const Store &o) = default;
    Store(Store &&o) = default;

    Store& operator=(const Store &o) = default;
    Store& operator=(Store &&o) = default;

    auto begin() {return bins_.begin();}
    auto end()   {return bins_.end();}
    auto begin() const {return bins_.begin();}
    auto end()   const {return bins_.end();}
    IntegerType &operator[](size_t i) {return bins_[i];}
    const IntegerType &operator[](size_t i) const {return bins_[i];}

    Store &operator+=(const Store &o) {
        if(o.count_ == 0) return *this;
        if(count_ == 0)
            std::copy(o.begin(), o.end(), begin());
        const bool minklt = o.mink_ < mink_;
        if(maxk_ > o.maxk_) {
            if(minklt) grow_left(o.mink_);

            for(size_t i = std::max(mink_, o.mink_), e = o.maxk_ + 1;i < e; ++i)
                bins_[i - mink_] += o.bins_[i - o.mink_];

            if(!minklt)
                bins_.front() += std::accumulate(o.begin(), o.begin() + (mink_ - o.mink_), IntegerType(0));
        } else {
            if(minklt) {
                auto tmp = o.bins_;
                for(size_t i = mink_; i < maxk_ + 1; ++i)
                    tmp[i - o.mink_] += bins_[i - mink_];
                bins_ = std::move(tmp);
                maxk_ = o.maxk_;
                mink_ = o.mink_;
            } else {
                grow_right(maxk_);
                for(size_t i = o.mink_; i < o.maxk_ + 1; ++i)
                    bins_[i - mink_] += o.bins_[i - o.mink_];
            }
        }
        count_ += o.count_;
        return *this;
    }
    Store operator+(const Store &o) {
        auto ret = *this;
        ret += o;
        return ret;
    }

    void addh(uint64_t key) {
        if(unlikely(count_ == 0))
            mink_ = maxk_ = key;
        else {
            if(key < mink_) grow_left(key);
            else grow_right(key);
        }
        ++bins_[std::max(ssize_t(0), ssize_t(key - mink_))];
        ++count_;
    }
    size_t size() const {return bins_.size();}
    void grow_left(uint64_t key) {
        if(mink_ < key || size() >= maxbins_) return;
        uint64_t tmpmink;
        if(maxk_ - key >= maxbins_) tmpmink = maxk_ - maxbins_ + 1;
        else {
            tmpmink = mink_;
            while(tmpmink > key) tmpmink -= grow_left_by;
        }
        bins_.insert(bins_.begin(), mink_ - key, IntegerType(0));
        mink_ = tmpmink;
    }
    void grow_right(uint64_t key) {
        if(key < maxk_) return;
        if(key - maxk_ >= maxbins_) {
            bins_.resize(maxbins_);
            std::fill(std::begin(bins_) + 1, std::end(bins_), 0);
            bins_.front() = count_;
            maxk_ = key;
            mink_ = key - maxbins_ + 1;
        } else if(key - mink_ >= maxbins_) {
            auto tmpmink = key - maxbins_ + 1;
            auto keydiff = tmpmink - mink_;
            auto beg = std::begin(bins_);
            auto endit = beg + keydiff - 1;
            auto n = std::accumulate(beg + 1, endit, *beg);
            bins_.erase(beg, endit);
            bins_.insert(bins_.end(), key - maxk_, IntegerType(0));
            maxk_ = key;
            mink_ = tmpmink;
            bins_.front() += n;
        } else {
            bins_.insert(bins_.end(), key - maxk_, IntegerType(0));
            maxk_ = key;
        }
        throw std::runtime_error("NotImplemented");
    }
    auto key_at_rank(uint64_t rank) const {
        // If this was managed better [e.g., prefix scan] we'd look it up
        // in log time, but whether that matters depends on the number of bins
        auto it = std::cbegin(bins_);
        const auto e = std::cend(bins_);
        uint64_t n = *it++;
        do {
            if(n >= rank) return it - std::cbegin(bins_) + mink_;
            n += *it++;
        } while(it != e);
        return maxk_;
    }
};

template<typename FType=float, typename StoreT=Store<>>
class DDSketch {
    const size_t maxbins_;
    FType alpha_;
    FType gamma_;
    FType lgamma_;
    FType mv_;
    double sum_;
    uint64_t count_;
    FType lowest_, highest_;
    int32_t offset_;
    StoreT store_;
public:
    DDSketch(DDSketch &&o) = default;
    DDSketch(const DDSketch &o) = default;
    DDSketch(FType alpha=1e-2, size_t max_bins=2048, FType min_value=1e-9):
        maxbins_(max_bins),
        alpha_(alpha), gamma_(1. + 2.*alpha/(1-alpha)), lgamma_(std::log1p(2.*alpha/(1.-alpha))),
        mv_(min_value), sum_(0), count_(0),
        lowest_(std::numeric_limits<FType>::min()), highest_(std::numeric_limits<FType>::max()),
        offset_(-static_cast<int32_t>(std::ceil(std::log(mv_)/lgamma_)) + 1)
    {
    }
    int64_t get_key(FType val) const  {
        if(val < -mv_)
            return -static_cast<int64_t>(std::ceil(std::log(-val)/lgamma_)) - offset_;
        if(val > mv_)
            return static_cast<int64_t>(std::ceil(std::log(val)/lgamma_)) + offset_;
        return 0;
    }
    void addh(FType x) {
        auto k = get_key(x);
        store_.addh(k);
        ++count_;
        sum_ += x;
        if(x < lowest_) lowest_ = x;
        if(x > highest_) highest_ = x;
    }
}; // class DDSketch

using ddf = DDSketch<float>;
using ddd = DDSketch<double>;

} // namespace dd

}

#endif
