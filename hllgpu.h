#ifndef HLL_GPU_H__
#define HLL_GPU_H__
#include "hll.h"

#ifdef __CUDACC__
#include "thrust/thrust.h"
#endif

namespace sketch {

using namespace hll;
auto sum_union_hlls(unsigned p, const std::vector<const uint8_t *__restrict__, Alloc> &re) {
    size_t maxnz = 64 - p + 1, nvals = maxnz + 1;
    size_t nsets = re.size(), nschoose2 = ((nsets - 1) * nsets) / 2;
    size_t m = 1ull << p;
#ifndef __CUDACC__
    std::vector<uint32_t> ret(nvals * nsets * m);
#else
    thrust::device_vector<uint32_t> ret(nvals * nsets * m);
#endif
    for(size_t i = 0; i < nsets; ++i) {
        auto p1 = re[i];
#ifndef __CUDACC__
        #pragma omp parallel for
        for(size_t j = i + 1; j < nsets; ++j) {
            auto p2 = re[j];
            auto rp = ret.data() + i * nsets * m;
            std::array<uint32_t, 64> z{0};
            for(size_t subi = 0; subi < m; subi += 8) {
                ++z[std::max(p1[subi + 0], p2[subi + 0])];
                ++z[std::max(p1[subi + 1], p2[subi + 1])];
                ++z[std::max(p1[subi + 2], p2[subi + 2])];
                ++z[std::max(p1[subi + 3], p2[subi + 3])];
                ++z[std::max(p1[subi + 4], p2[subi + 4])];
                ++z[std::max(p1[subi + 5], p2[subi + 5])];
                ++z[std::max(p1[subi + 6], p2[subi + 6])];
                ++z[std::max(p1[subi + 7], p2[subi + 7])];
            }
            std::memcpy(ret.data() + (i * nsets * m) + j * m, z.data(), m);
        }
#else
        for(size_t j = i + 1; j < nsets; ++j) {
            auto p2 = re[j];
            auto rp = ret.data() + i * nsets * m;
            std::array<uint32_t, 64> z{0};
            for(size_t subi = 0; subi < m; subi += 8) {
                ++z[max(p1[subi + 0], p2[subi + 0])];
                ++z[max(p1[subi + 1], p2[subi + 1])];
                ++z[max(p1[subi + 2], p2[subi + 2])];
                ++z[max(p1[subi + 3], p2[subi + 3])];
                ++z[max(p1[subi + 4], p2[subi + 4])];
                ++z[max(p1[subi + 5], p2[subi + 5])];
                ++z[max(p1[subi + 6], p2[subi + 6])];
                ++z[max(p1[subi + 7], p2[subi + 7])];
            }
            if(cudaMemcpy(ret.data() + (i * nsets * m) + j * m, z.data(), cudaMemcpyDeviceToDevice)) throw 1;
        }
#endif
    }
    return std::make_pair(std::move(ret), nvals);
}
#endif

} // sketch

#endif
