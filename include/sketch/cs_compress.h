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

/***
   Utilities for compressing and decompressing
  */

//template<typename FT, bool SO>
//blaze::Dynamic

// TODO: write a generic method compatible with pytorch/ATen
//       (this actually does work, assuming we're working on the last level of a packedtensoraccessor), just not on GPU yet.
//       1. exponential distribution (Woodruff-Zhang transform) [TODONE]
//       2. Random Laplace Transforms (Random Laplace Feature Maps for Semigroup Kernels on Histograms, Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels)
//       3. Quasi Random Fourier Transform
//

template<typename C, typename Hasher=KWiseHasherSet<4>>
auto cs_compress(const C &in, size_t newdim, const Hasher &hf) {
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
template<typename C, typename Hasher=KWiseHasherSet<4>, typename RNG=blaze::RNG>
auto wz_compress(const C &in, size_t newdim, const Hasher &hf, double p) {
    //using FT = std::decay_t<decltype(*std::begin(in))>;
    if(newdim > in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    C ret(newdim * ns);
    std::exponential_distribution<double> gen(p);
    for(unsigned j = 0; j < ns; ++j) {
        for(unsigned i = 0; i < in.size(); ++i) {
            const auto v = in[i];
            auto hv = hf(i, j);
            auto dm = div.divmod(hv);
            auto ind = dm.rem * ns + j;
            RNG rng(dm.quot >> 1);
            const double mult = gen(rng) * (dm.quot & 1 ? 1: -1);
            ret.operator[](ind) += v * mult;
        }
        // TODO: decompress.
        // Sample using the same seed, just multiply by inverse
    }
    return ret;
}

template<typename C, typename OutC, typename Hasher=KWiseHasherSet<4>,
         typename=std::enable_if_t<!std::is_arithmetic<OutC>::value>,
         typename RNG=blaze::RNG>
auto &wz_decompress(const C &in, const Hasher &hf, OutC &ret, double p) {
    // TODO: some kind of importance sampling, weighting larger items more.
    PREC_REQ(in.size() % hf.size() == 0, "in dimension must be divisible by hf count");
    size_t olddim = in.size() / hf.size();
    size_t newdim = ret.size();
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    std::exponential_distribution<double> gen(p);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 9> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            auto dm = div.divmod(hv);
            RNG rng(dm.quot >> 1);
            tmp[j] = in[dm.rem * ns + j] / gen(rng) * (dm.quot & 1 ? 1: -1);
        }
        common::sort::insertion_sort(tmp, tmp + hf.size());
        ret[i] = (tmp[ns >> 1] + tmp[(ns - 1) >> 1]) * .5;
    }
    return ret;
}

template<typename C, typename OutC, typename Hasher=KWiseHasherSet<4>,
         typename=std::enable_if_t<!std::is_arithmetic<OutC>::value>>
auto &cs_decompress(const C &in, const Hasher &hf, OutC &ret) {
    PREC_REQ(in.size() % hf.size() == 0, "in dimension must be divisible by hf count");
    size_t olddim = in.size() / hf.size();
    size_t newdim = ret.size();
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 9> mem(hf.size());
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

template<typename C, typename OutC=C, typename Hasher=KWiseHasherSet<4>>
auto cs_decompress(const C &in, const Hasher &hf, size_t newdim) {
    OutC ret(newdim);
    cs_decompress<C, OutC, Hasher>(in, hf, ret);
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
