#ifndef HLL_GPU_H__
#define HLL_GPU_H__
#include "hll.h"

#ifdef __CUDACC__
//#include "thrust/thrust.h"
#endif

namespace sketch {


__host__ __device__ double inline origest(const uint32_t *p, unsigned l2) {
    auto m = size_t(1) << l2;
    double alpha = m == 16 ? .573 : m == 32 ? .697 : m == 64 ? .709: .7213 / (1. + 1.079 / m);
    double s = p[0];
    for(auto i = 1u; i < 64; ++i)
        s += ldexp(p[i], i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
    return alpha * m * m / s;
}

using namespace hll;
template<typename Alloc>
auto sum_union_hlls(unsigned p, const std::vector<const uint8_t *__restrict__, Alloc> &re) {
    size_t maxnz = 64 - p + 1, nvals = maxnz + 1;
    size_t nsets = re.size(), nschoose2 = ((nsets - 1) * nsets) / 2;
    size_t m = 1ull << p;
    std::vector<uint32_t> ret(nvals * nsets * m);
    for(size_t i = 0; i < nsets; ++i) {
        auto p1 = re[i];
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
    }
    return std::make_pair(std::move(ret), nvals);
}

#ifdef __CUDACC__

__global__ void calc_sizes(const uint8_t *p, unsigned l2, size_t nhlls, uint32_t *sizes) {
    extern __shared__ int shared[];
    uint32_t *sums = (uint32_t *)shared;
    for(int i = 0; i < 64; ++i) sums[i] = 0;
    size_t nelem = size_t(1) << l2;
    auto gid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    auto tid = threadIdx.y * blockDim.x + threadIdx.x;
    auto gsz = blockDim.x * blockDim.y;
    auto hllid = (gid  + gsz - 1) / gsz, hllrem = (gid % gsz);
    int nreg_each = (nelem + gsz - 1) / gsz;
    if(hllid >= nhlls)
        return;
    uint32_t arr[64]{0};
    auto hp = p + nelem * hllid;
    //auto nblocks = (nhlls + 63) / 64;
    __syncthreads();
    for(unsigned i = nreg_each * hllrem;  i < min(size_t((nreg_each * (hllrem + 1))), nelem); ++i) {
        ++arr[hp[i]];
    }
    for(auto i = 0u; i < 64; ++i) {
        atomicAdd(sums + i, hp[i]);
    }
    __syncthreads();
    if(tid == 0)
        sizes[hllid] = origest(sums, l2);
}

__host__ std::vector<uint32_t> all_pairs(const uint8_t *p, unsigned l2, size_t nhlls) {
    size_t nc2 = (nhlls * (nhlls - 1)) / 2;
    uint32_t *sizes;
    size_t nb = sizeof(uint32_t) * nhlls;
    cudaError_t ce;
    if((ce = cudaMalloc((void **)&sizes, nb)))
        throw ce;
    //size_t nblocks = 1;
    std::fprintf(stderr, "About to launch kernel\n");
    calc_sizes<<<64,threads,64 * sizeof(uint32_t)/*,nhlls * sizeof(float)*/>>>(p, l2, nhlls, sizes);
    std::fprintf(stderr, "Finished kernel\n");
    cudaDeviceSynchronize();
    std::vector<uint32_t> ret(nhlls);
    if(cudaMemcpy(ret.data(), sizes, nhlls * sizeof(uint32_t), cudaMemcpyDeviceToHost)) throw 3;
    //thrust::copy(sizes, sizes + ret.size(), ret.begin());
    cudaFree(sizes);
    return ret;
}
#endif

} // sketch

#endif
