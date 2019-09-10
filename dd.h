#ifndef DDSKETCH_H__
#define DDSKETCH_H__
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cmath>

#ifndef unlikely
#  ifdef __GNUC__ || __clang__
#    define unlikely(x) __builtin_expect((x), 0)
#  else
#    define unlikely(x) (x)
#  endif
#endif


namespace sketch {


namespace dd {

using std::size_t;
using std::int64_t;
using std::uint64_t;

// Based on implementation from https://raw.githubusercontent.com/DataDog/sketches-py/master/ddsketch/ddsketch.py
// Accessed 9/6/19

template<typename IntegerType=std::int64_t, size_t initial_nbins=128>
struct Store {

    const size_t maxbins_;
    std::vector<IntegerType> bins_;
    size_t count_;
    uint64_t mink_, maxk_;
    Store(const Store &o) = default;
    Store(Store &&o) = default;

    using Type = IntegerType;
    Store(size_t maxnbins): maxbins_(maxnbins), bins_(initial_nbins, 0), count_(0), mink_(0), maxk_(0) {}
    Store(const Store &o) = default;
    Store(Store &&o) = default;
    Store& operator=(const Store &o) = default;
    Store& operator=(Store &&o) = default;
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
        throw std::runtime_error("NotImplemented");
    }
    void grow_right(uint64_t key) {
        throw std::runtime_error("NotImplemented");
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
using namespace dd;

}

#endif
