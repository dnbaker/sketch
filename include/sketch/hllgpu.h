#ifndef HLL_GPU_H__
#define HLL_GPU_H__
#include <omp.h>
#include "hll.h"
#include "exception.h"
#include <type_traits>
#if USE_THRUST
#  include <thrust/device_vector.h>
#endif

namespace sketch {
using hrc = std::chrono::high_resolution_clock;

CUDA_ONLY(using exception::CudaError;)

template<typename T, typename=std::enable_if_t<std::is_arithmetic<T>::value>>
CUDA_ONLY(__host__ __device__)
INLINE uint64_t nchoose2(T x) {
    return (uint64_t(x) * uint64_t(x - 1)) / 2;
}


template<typename T>
CUDA_ONLY(__host__ __device__)
INLINE void increment_maxes(T *SK_RESTRICT arr, unsigned x1, unsigned x2) {
#if __CUDA_ARCH__
    x1 = __vmaxu4(x1, x2);
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

#ifdef __CUDACC__
template<typename T, typename T2, typename T3, typename=typename std::enable_if<
             std::is_integral<T>::value && std::is_integral<T2>::value && std::is_integral<T3>::value
         >::type>
__device__ __host__ static inline T ij2ind(T i, T2 j, T3 n) {
#define ARRAY_ACCESS(row, column) (((row) * (n * 2 - row - 1)) / 2 + column - (row + 1))
    const auto i1 = min(i, j), j1 = min(j, i);
    return ARRAY_ACCESS(i1, j1);
#undef ARRAY_ACCESS
}
__host__ __device__ INLINE void set_lut_portion(uint32_t i, uint32_t jstart, uint32_t jend, uint32_t n, uint64_t *lut) {
    SK_UNROLL(8)
    for(;jstart < jend; ++jstart) {
        lut[ij2ind(i, jstart, n)] = (uint64_t(i) << 32) | jstart;
    }
}

#endif // __CUDACC__

template<typename T>
CUDA_ONLY(__host__ __device__)
INLINE double origest(const T &p, unsigned l2) {
    const auto m = 1u << l2;
    const double alpha = m == 16 ? .573 : m == 32 ? .697 : m == 64 ? .709: .7213 / (1. + 1.079 / m);
    double s = p[0];
    SK_UNROLL(8)
    for(auto i = 1u; i < 64 - l2 + 1; ++i) {
#if __CUDA_ARCH__
        s += ldexpf(p[i], -i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
#else
        s += std::ldexp(p[i], -i);
#endif
    }
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

#ifdef __CUDACC__



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
    uint32_t arr[60]{0};
    hp = p + (threadIdx.x << l2);
    SK_UNROLL(8)
    for(int i = 0; i < (1L << l2); i += 4) {
        increment_maxes(arr, *(unsigned *)&hp[i], *(unsigned *)&registers[i]);
    }
    sizes[gid] = origest(arr, l2);
}

#if 0
__host__ INLINE uint3 calculate_indices(uint32_t block_id, uint32_t nhlls, uint16_t nblock_per_cmp) {
    // Think it through: is there any reason why you have to work in rows at all?
    const int cmp_ind = block_id / nblock_per_cmp;
    const auto ij = ind2ij(cmp_id, nhlls);
    // ind2ij is WRONG, needs to be written.
    // Is there a simple function for this?
    //if(cmp_ind :
    return uint3(ij.x, ij.y, cmp_ind);
}
#endif

__global__ void calc_sizes_row(const uint8_t *SK_RESTRICT p, unsigned l2, unsigned nhlls,
                               unsigned nrows, unsigned starting_rownum, unsigned mem_per_block,
                               uint32_t *SK_RESTRICT register_sums,
                               uint32_t *SK_RESTRICT sizes, uint16_t nblock_per_cmp) {
    extern __shared__ int shared[];
    auto sptr = reinterpret_cast<uint32_t *>(&shared[0]);

    const int block_id = blockIdx.x +
                         blockIdx.y * gridDim.x;

    const int cmp_ind = block_id / nblock_per_cmp + starting_rownum;
    // TODO: back-calculate indices for chunk
    // Generalize to not be limited to row sizes but arbitrary chunks
    // Dispatch
    // ???
    // PROFIT
#if 0
    const uint3 = calculate_indices(block_id, nhlls, nblock_per_cmp);
#endif
    auto ncmp = nrows * nhlls;
    if(cmp_ind > ncmp) return;

    auto nblocks = ncmp * nblock_per_cmp;

    const int global_tid = block_id * blockDim.x + threadIdx.x;
    auto tid = threadIdx.x;
    const int nl2s = 64 - l2 + 1;
    for(int i = tid * (nl2s) / blockDim.x; i < min((tid + 1) * nl2s / blockDim.x, nl2s); ++i)
        sptr[i] = 0;
    auto bid = blockIdx.x;
    auto gid = bid * blockDim.x + tid;
    const uint32_t nreg = size_t(1) << l2;
    uint32_t arr[60]{0};
#if 0
    int lhi, rhi;
    get_indices(nhlls, nblock_per_cmp, block_id, &lhi, &rhi);
    int total_workers = blockDim.x * nblock_per_cmp;
    #pragma unroll 8
    for(int i = nreg * tid / total_workers; i < min(nreg, (nreg * (tid + 1) / total_workers)); i += 4) {
        increment_maxes(arr,
                        *reinterpret_cast<unsigned *>(&hp[i]),
                        *reinterpret_cast<unsigned *>(&registers[i]));
    }
    __syncthreads();
    for(int i = tid * (nl2s) / blockDim.x; i < min((tid + 1) * nl2s / blockDim.x, nl2s); ++i)
        atomicAdd(register_sums + (nl2s * , sptr[i]);
#endif
}

__host__ std::vector<uint32_t> all_pairsu(const uint8_t *SK_RESTRICT p, unsigned l2, size_t nhlls, size_t &SK_RESTRICT rets) {
    size_t nc2 = nchoose2(nhlls);
    uint32_t *sizes;
    size_t nb = sizeof(uint32_t) * nc2;
    size_t m = 1ull << l2;
    cudaError_t ce;
    //size_t nblocks = 1;
    std::fprintf(stderr, "About to launch kernel\n");
    auto t = hrc::now();
    size_t tpb = nhlls/2 + (nhlls&1);
    std::vector<uint32_t> ret(nc2);
    PREC_REQ(l2 > 5, "l2 must be 6 or greater");
    if(tpb > 1024) {
        const size_t mem_per_block = sizeof(uint32_t) * (64 - l2 + 1);
        const int nrows = 1; // This can/should be changed eventually
        const int nblock_per_cmp = 4; // Can/should be changed later
        size_t ncmp_per_loop = nhlls * nrows;
        size_t nblocks = nblock_per_cmp * ncmp_per_loop;
        auto ydim = (nblocks + 65535 - 1) / 65535;
        dim3 griddims(nblocks < 65535 ? nblocks: 65535, ydim);
        if((ce = cudaMalloc((void **)&sizes, ncmp_per_loop * sizeof(uint32_t))))
            throw CudaError(ce, "Failed to malloc for row");
#if USE_THRUST
        thrust::device_vector<uint32_t> tv((64 - l2 + 1) * ncmp_per_loop);
        uint32_t *register_sums(raw_pointer_cast(tv.data()));
#else
        uint32_t *register_sums;
        if((ce = cudaMalloc((void **)&register_sums, ncmp_per_loop * mem_per_block)))
            throw CudaError(ce, "Failed to malloc for row");
#endif
        // This means work per block before updating will be mem_per_block / nblocks
        for(int i = 0; i < (nhlls + (nrows - 1)) / nrows; i += nrows) {
            // zero registers
            cudaMemset(register_sums, 0, ncmp_per_loop * mem_per_block);
            calc_sizes_row<<<griddims,256,mem_per_block>>>(p, l2, nhlls, nrows, i, mem_per_block, register_sums, sizes, nblock_per_cmp);
            if((ce = cudaDeviceSynchronize())) throw CudaError(ce, "Failed to synchronize");
            cudaMemcpy(sizes, ret.data(), ncmp_per_loop * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        }
    } else {
        if((ce = cudaMalloc((void **)&sizes, nb)))
            throw CudaError(ce, "Failed to malloc");
        calc_sizes_1024<<<nhlls,(nhlls+1)/2,m>>>(p, l2, nhlls, sizes);
        if((ce = cudaDeviceSynchronize())) throw CudaError(ce, "Failed to synchronize");
        if((ce = cudaMemcpy(ret.data(), sizes, nb, cudaMemcpyDeviceToHost))) throw CudaError(ce, "Failed to copy device to host");
    }
    if((ce = cudaDeviceSynchronize())) throw CudaError(ce, "Failed to synchronize");
    if((ce = cudaGetLastError())) throw CudaError(ce, "");
    auto t2 = hrc::now();
    rets = (t2 - t).count();
    std::fprintf(stderr, "Time: %zu\n", rets);
    std::fprintf(stderr, "Finished kernel\n");
    //thrust::copy(sizes, sizes + ret.size(), ret.begin());
    cudaFree(sizes);
    return ret;
}
#endif

} // sketch

#endif
