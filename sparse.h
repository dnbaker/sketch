#pragma once
#include "common.h"

namespace sketch {

namespace sparse {

// For HLL

struct SparseEncoding {
    uint32_t get_index(uint32_t val) const;
    uint8_t get_value(uint32_t val) const;
    uint32_t encode_value(uint32_t index, uint8_t val) const;
};

struct SparseHLL32: public SparseEncoding {
    uint32_t get_index(uint32_t val) const {
        return val >> 8;
    }
    uint8_t get_value(uint32_t val) const {return val;} // Implicitly truncated.
    uint32_t encode_value(uint32_t index, uint8_t val) const {
        index <<= 8;
        index |= val;
        return index;
    }
};

} // sparse

} // sketch
