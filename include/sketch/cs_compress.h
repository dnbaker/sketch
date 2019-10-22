#pragma once
#include "blaze/Math.h"
#include "div.h"
#include "common.h"
#include <queue>
#if 0
#if __cplusplus <= 201703L
#error("Need c++20")
#endif
#include <concepts>
#endif

namespace sketch {

//template<typename FT, bool SO>
//blaze::Dynamic

// TODO: write a generic method compatible with pytorch/ATen

template<typename C>
auto cs_compress(const C &in, size_t newdim, const KWiseHasherSet<4> &hf) {
    //using FT = std::decay_t<decltype(*std::begin(in))>;
    if(newdim > in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    C ret(newdim * ns);
    for(unsigned j = 0; j < ns; ++j) {
        for(unsigned i = 0; i < in.size(); ++i) {
            const auto v = in[i];
            auto hv = hf(i, j);
            auto ind = div.mod(hv >> 1) * ns + j;
            ret.operator[](ind) += v * (hv & 1 ? 1: -1);
        }
    }
    return ret;
}
template<typename C>
auto cs_decompress(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf) {
    const size_t ns = hf.size();
    //if(newdim < in.size())
    schism::Schismatic<uint32_t> div(olddim);
    //using FT = std::decay_t<decltype(*std::begin(in))>;
    C ret(newdim);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 8> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            tmp[j] = in[div.mod(hv >> 1) * ns + j] * (hv & 1 ? 1: -1);
        }
        common::sort::insertion_sort(tmp, tmp + hf.size());
        ret[i] = (tmp[ns >> 1] + tmp[(ns - 1) >> 1]) * .5; 
    }
    return ret;
}

struct AbsMax {
    template<typename T>
    bool operator()(T x, T y) const {return std::abs(x) > std::abs(y);}
};

template<typename C, typename Functor=std::greater<void>>
auto top_indices_from_compressed(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf, unsigned k) {
    //if(newdim < in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    using FT = std::decay_t<decltype(*std::begin(in))>;
    std::priority_queue<std::pair<FT, unsigned>, std::vector<std::pair<FT, unsigned>>, Functor> pq;
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 8> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            tmp[j] = in.operator[](div.mod(hv >> 1) * ns + j) * (hv & 1 ? 1: -1);
        }
        common::sort::insertion_sort(tmp, tmp + hf.size());
        std::pair<FT, unsigned> pair = std::make_pair((tmp[ns >> 1] + tmp[(ns - 1) >> 1]) * .5, unsigned(i));
        OMP_PRAGMA("omp critical")
        {
            pq.push(pair);
            if(pq.size() > k) pq.pop();
        }
    }
    std::pair<std::vector<FT>, std::vector<unsigned>> ret;
    ret.first.reserve(k);
    ret.second.reserve(k);
    for(unsigned i = 0; i < k; ++i) {
        const auto &r(pq.top());
        ret.first.push_back(r.first);
        ret.second.push_back(r.second);
        pq.pop();
    }
    return ret;
}


} // sketch
