#pragma once
#include "blaze/Math.h"
#include "div.h"
#include "common.h"
#if 0
#if __cplusplus <= 201703L
#error("Need c++20")
#endif
#include <concepts>
#endif

namespace sketch {

template<typename C>
auto cs_compress(const C &in, size_t newdim, const KWiseHasherSet<4> &hf) {
    if(newdim > in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    using FT = std::decay_t<decltype(*std::begin(in))>;
    std::vector<FT> ret(newdim * ns);
    for(unsigned j = 0; j < ns; ++j) {
        for(size_t i = 0; i < in.size(); ++i) {
            const auto v = in[i];
            auto hv = hf(i, j);
            ret[div.mod(hv >> 1) + in.size() * j] += v * (hv & 1 ? 1: -1);
        }
    }
    return ret;
}
template<typename C>
auto cs_decompress(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf) {
    if(newdim < in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    using FT = std::decay_t<decltype(*std::begin(in))>;
    std::vector<FT> ret(newdim);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 8> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            tmp[j] = in[div.mod(hv >> 1) + j * olddim] * (hv & 1 ? 1: -1);
        }
        std::sort(tmp, tmp + hf.size());
        ret[i] = (tmp[ns >> 1] + tmp[(ns - 1) >> 1]) * .5; 
    }
    return ret;
}

} // sketch
