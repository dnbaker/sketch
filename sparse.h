#pragma once
#include "hll.h"

namespace sketch {

namespace sparse {

// For HLL

struct SparseEncoding {
    static constexpr uint32_t get_index(uint32_t val);
    static constexpr uint8_t get_value(uint32_t val);
    static constexpr uint32_t encode_value(uint32_t index, uint8_t val) ;
};

struct SparseHLL32: public SparseEncoding {
    static constexpr uint32_t get_index(uint32_t val) {
        return val >> 8;
    }
    static constexpr uint8_t get_value(uint32_t val) {return static_cast<uint8_t>(val);} // Implicitly truncated, but we add a cast for clarity
    static constexpr encode_value(uint32_t index, uint8_t val) {
        return (index << 8) | val;
    }
};
template<typename HashStruct>
std::array<double, 3> get_counts(const hll::hllbase_t<HashStruct> &h) {
    auto sc = hll::detail::sum_counts(h.core());
}

template<typename HashStruct>
std::array<double, 3> sparse_query(const hll::hllbase_t<HashStruct> &h, size_t nvals, uint32_t *vi, std::array<uint32_t, 64> *a) {
    if(h.p() > 24)
        throw common::NotImplementedError("Error: sparse query doesn't support HLLs with prefix > 24");
    bool del;
    if(a == nullptr) {
        a = new std::array<uint32_t, 64>;
        *a = hll::detail::sum_counts(h.core());
        del = true;
    } else del = false;
    throw common::NotImplementedError("The core comparison for sparse has not yet been implemented. However, this isn't too far from it.");

    if(del) delete a;
}

} // sparse

} // sketch
