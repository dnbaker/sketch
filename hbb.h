#ifndef HYPERBITBIT_H__
#define HYPERBITBIT_H__
#include "common.h"

namespace sketch {

namespace hbb {
using namespace ::sketch::common;

/**
 * 
 * HyperBitBit algorithm (c/o Sedgewick) from
 * https://www.cs.princeton.edu/~rs/talks/AC11-Cardinality.pdf
 * Based on https://github.com/thomasmueller/tinyStats/blob/master/src/main/java/org/tinyStats/cardinality/HyperBitBit.java
 */
template<typename HashStruct>
class HyperBitBit {

    uint16_t    logn_;
    uint64_t s1_, s2_;
    HashStruct    hf_;
public:
    uint64_t hash(uint64_t item) const {return hf_(item);}
    template<typename...Args>
    HyperBitBit(Args &&...args): logn_(5), s1_(0), s2_(0), hf_(std::forward<Args>(args)...) {}

    void addh(uint64_t item) {add(hash(item));}
    void add(uint64_t hv) {
        auto r = __builtin_ctz(hv);
        if(r > logn_) {
            const auto k = (hv >> (sizeof(hv) * CHAR_BIT - 6));
            s1_ |= 1L << k;
            if (r > logn_ + 1) s2_ |= 1L << k;
            if(popcount(s1_) > 31) {
                s1_ = s2_; s2_ = 0; ++logn_;
            }
        }
    }

    double cardinality_estimate() const {return std::pow(2., (logn_ + 5.15 + popcount(s1_) / 32.));}

};

} // hbb
} // sketch

#endif /* HYPERBITBIT_H__ */
