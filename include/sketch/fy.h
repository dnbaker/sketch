#ifndef WMH_FY_H__
#define WMH_FY_H__
#include "sketch/div.h"
#include <vector>
#include <cstdlib>
#include <memory>
#include "aesctr/wy.h"
#include <cassert>

namespace fisher_yates {

using std::size_t;

struct LazyShuffler {
    // Algorithm 6, https://arxiv.org/pdf/1911.00675.pdf
    // Uses 32-bit integers for cheaper modulo reductions,
    // and uses the fastmod https://arxiv.org/abs/1902.01961 trick
    using IT = uint32_t;
private:
    std::vector<IT> data_;
    wy::WyRand<uint32_t, 2> rng_;
    size_t i_ = 0, c_ = 0, sz_;
    std::vector<schism::Schismatic<IT>> divs_;


    IT &getg(size_t i) {return data_[i << 1];}
    IT &getv(size_t i) {return data_[(i << 1) + 1];}
public:
    LazyShuffler(size_t n, uint64_t seed=0): data_(n * 2), rng_(seed), sz_(n) {
        divs_.reserve(n);
        for(size_t i = 0; i < n; ++i)
            divs_.emplace_back(n - i);
        reset();
    }
    size_t size() const {return sz_;}
    IT step() {
        IT samp = divs_[i_].mod(rng_());
        IT j = i_ + samp;
        assert(j < size());
        IT &gj = getg(j);
        const IT &gi = getg(i_);
        IT &vj = getv(j);
        const IT k  = vj  == c_ ? gj: j;
        gj = getv(i_) == c_ ? gi: i_;
        vj = c_;
        if(++i_ == divs_.size()) i_ = 0;
        return k;
    }
    bool has_next() {return i_ < sz_;}
    void seed(uint64_t seed) {
        rng_.seed(seed);
    }
    void resize(size_t newsize, uint64_t seed=0) {
        data_.resize(newsize * 2);
        std::fill(data_.begin(), data_.end(), IT(0));
        divs_.clear();
        for(size_t i = 0; i < newsize; ++i)
            divs_.emplace_back(newsize - i);
        rng_.seed(seed);
        reset();
    }
    void reset() {
        i_ = 0;
        ++c_;
    }
};

} // namespace fisher_yates

namespace fy = fisher_yates;

#endif
