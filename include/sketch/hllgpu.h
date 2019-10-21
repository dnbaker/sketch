#ifndef HLL_GPU_H__
#define HLL_GPU_H__
#include "hll.h"
#include "exception.h"
#include <type_traits>

#ifdef __CUDACC__
//#include "thrust/thrust.h"
#endif

namespace sketch {
using hrc = std::chrono::high_resolution_clock;


template<typename T, typename=std::enable_if_t<std::is_arithmetic<T>::value>>
#ifdef __CUDACC__
__host__ __device__
#endif
INLINE uint64_t nchoose2(T x) {
    uint64_t ret = x;
    ret *= (x - 1);
    return ret >>= 1;
}

using exception::CudaError;


template<typename T>
__host__ __device__
INLINE void increment_maxes(T *SK_RESTRICT arr, unsigned x1, unsigned x2) {
#if __CUDA_ARCH__
    x1 = __vmaxu4(x1, x2);
#elif __SSE2__
    auto v = _mm_max_pu8(*(__m64 *)&x1, *(__m64 *)x2);
    std::memcpy(&x1, &v, sizeof(v));
#else
    // Manual
    unsigned x3 = std::max(x1>>24, x2>>24);
    x3 <<= 8;
    x3 |= std::max((x1 >> 16) & 0xFFu, (x2 >> 16) & 0xFFu);
    x3 <<= 8;
    x3 |= std::max((x1 >> 8) & 0xFFu, (x2 >> 8) & 0xFFu);
    x1 = (x3 << 8) | std::max(x1 & 0xFFu, x2 & 0xFFu);
#endif
    ++arr[x1&0xFFu];
    ++arr[(x1>>8)&0xFFu];
    ++arr[(x1>>16)&0xFFu];
    ++arr[x1>>24];
}

template<typename T>
__host__ __device__
double inline origest(const T &p, unsigned l2) {
    auto m = size_t(1) << l2;
    const double alpha = m == 16 ? .573 : m == 32 ? .697 : m == 64 ? .709: .7213 / (1. + 1.079 / m);
    double s = p[0];
    #pragma unroll 8
    for(auto i = 1u; i < 64 - l2 + 1; ++i)
        s += ldexp(p[i], -i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
    return alpha * m * m / s;
}

using namespace hll;
template<typename Alloc>
auto sum_union_hlls(unsigned p, const std::vector<const uint8_t *SK_RESTRICT, Alloc> &re) {
    size_t maxnz = 64 - p + 1, nvals = maxnz + 1;
    size_t nsets = re.size();//, nschoose2 = ((nsets - 1) * nsets) / 2;
    size_t m = 1ull << p;
    std::vector<uint32_t> ret(nvals * nsets * m);
    for(size_t i = 0; i < nsets; ++i) {
        auto p1 = re[i];
        OMP_PRAGMA("omp parallel for")
        for(size_t j = i + 1; j < nsets; ++j) {
            auto p2 = re[j];
            auto rp = ret.data() + i * nsets * m;
            std::array<uint32_t, 64> z{0};
            for(size_t subi = 0; subi < m; subi += 8) {
                increment_maxes(z.data(),
                    *reinterpret_cast<const unsigned *>(&p1[subi]),
                    *reinterpret_cast<const unsigned *>(&p2[subi]));
                increment_maxes(z.data(),
                    *reinterpret_cast<const unsigned *>(&p1[subi+4]),
                    *reinterpret_cast<const unsigned *>(&p2[subi+4]));
            }
            std::memcpy(ret.data() + (i * nsets * m) + j * m, z.data(), m);
        }
    }
    return std::make_pair(std::move(ret), nvals);
}

template<typename T, typename T2, typename T3, typename=typename std::enable_if<std::is_integral<T>::value && std::is_integral<T2>::value && std::is_integral<T3>::value>::type>
__device__ __host__ static inline T ij2ind(T i, T2 j, T3 n) {
    return i < j ? (((i) * (n * 2 - i - 1)) / 2 + j - (i + 1)): (((j) * (n * 2 - j - 1)) / 2 + i - (j + 1));
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
#if 0
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



__global__ void calc_sizes_1024(const uint8_t *p, unsigned l2, size_t nhlls, uint32_t *sizes) {
    if(blockIdx.x >= nhlls) return;
    extern __shared__ int shared[];
    uint8_t *registers = (uint8_t *)shared;
    auto hllid = blockIdx.x;
    int nreg = 1 << l2;
    int nper = (nreg + (nhlls - 1)) / nhlls;
    auto hp = p + (hllid << l2);
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= (nhlls - 1) * nhlls / 2)
        return;
    for(int i = threadIdx.x * nper; i < min((threadIdx.x + 1) * nper, nreg); ++i)
        registers[i] = hp[i];
    __syncthreads();
    uint32_t arr[64]{0};
    hp = p + (threadIdx.x << l2);
    #pragma unroll 8
    for(int i = 0; i < (1L << l2); i += 4) {
        increment_maxes(arr, *(unsigned *)&hp[i], *(unsigned *)&registers[i]);
    }
    sizes[gid] = origest(arr, l2);
}

__global__ void calc_sizes_large(const uint8_t *SK_RESTRICT p, unsigned l2, size_t nhlls, size_t nblocks, size_t mem_per_block, uint32_t *SK_RESTRICT sizes) {
    extern __shared__ int shared[];
    auto sptr = reinterpret_cast<uint8_t *>(&shared[0]);
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;
    auto gid = bid * blockDim.x + tid;
    size_t nc2 = nchoose2(nhlls);
    const size_t nreg = size_t(1) << l2;
    // First, divide the total amount of work that our block of workers will process
    auto range_start = uint64_t(bid) * nc2 / nblocks;
    const auto range_end = size_t(std::max(uint64_t(bid + 1) * nc2 / nblocks, uint64_t(nc2)));
    // These parameters are shared between all threads in a block
    if(range_start >= range_end) return; // Skip overflows
    auto lhid = range_start / nhlls;
    auto rhid = range_start % nhlls;
    const uint8_t *lhs = p + (lhid << l2);
    for(;;) {
        const uint8_t *const rhs = p + (rhid << l2);
        if(++range_start == range_end) break;
        rhid = range_start % nhlls;
        lhid = range_start / nhlls;
        lhs = p + (lhid << l2);
    }
}

__host__ std::vector<uint32_t> all_pairsu(const uint8_t *SK_RESTRICT p, unsigned l2, size_t nhlls, size_t &SK_RESTRICT rets) {
    size_t nc2 = nchoose2(nhlls);
    uint32_t *sizes;
    size_t nb = sizeof(uint32_t) * nc2;
    size_t m = 1ull << l2;
    cudaError_t ce;
#if __CUDACC__
    if((ce = cudaMalloc((void **)&sizes, nb)))
        throw CudaError(ce, "Failed to malloc");
#else
#error("This function can't be compiled by a non-cuda compiler")
#endif
    //size_t nblocks = 1;
    std::fprintf(stderr, "About to launch kernel\n");
    auto t = hrc::now();
#if 0
    if(nc2 <= (1ull << 32)) {
        size_t nblocks = nc2;
        unsigned tpb = std::min(1024, 1<<l2); // Ensure that none of my indexing is too messed up
        calc_sizesnew<<<nblocks, tpb>>>(p, l2, nhlls, nblocks, sizes);
    }
#else
    size_t tpb = nhlls/2 + (nhlls&1);
    if(tpb > 1024) {
        throw std::runtime_error("Current implementation is limited to 1024 by 1024 comparisons. TODO: fix this with a reimplementation");
        static constexpr size_t nblocks = 0x20000ULL;
        static constexpr size_t mem_per_block = (16 << 20) / nblocks;
        // This means work per block before updating will be mem_per_block / nblocks
        
        calc_sizes_large<<<nblocks,1024,mem_per_block>>>(p, l2, nhlls, nblocks, mem_per_block, sizes);
    } else {
        calc_sizes_1024<<<nhlls,(nhlls+1)/2,m>>>(p, l2, nhlls, sizes);
    }
#endif
    if((ce = cudaDeviceSynchronize())) throw CudaError(ce, "Failed to synchronize");
    if((ce = cudaGetLastError())) throw CudaError(ce, "");
    auto t2 = hrc::now();
    rets = (t2 - t).count();
    std::fprintf(stderr, "Time: %zu\n", rets);
    std::fprintf(stderr, "Finished kernel\n");
    std::vector<uint32_t> ret(nc2);
    if((ce = cudaMemcpy(ret.data(), sizes, nb, cudaMemcpyDeviceToHost))) throw CudaError(ce, "Failed to copy device to host");
    //thrust::copy(sizes, sizes + ret.size(), ret.begin());
    cudaFree(sizes);
    return ret;
}
#endif

} // sketch

#endif
