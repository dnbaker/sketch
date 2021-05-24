#ifndef HLL_GPU_H__
#define HLL_GPU_H__
#include "sketch/hll.h"
#include "macros.h"
#include "exception.h"
#include <type_traits>
#if USE_THRUST
#  include <thrust/device_vector.h>
#endif
#include <chrono>

namespace sketch {
using hrc = std::chrono::high_resolution_clock;

#ifdef __CUDACC__
using exception::CudaError;
#endif

template<typename T, typename=std::enable_if_t<std::is_arithmetic<T>::value>>
#ifdef __CUDACC__
__host__ __device__
#endif
INLINE uint64_t nchoose2(T x) {
    return (uint64_t(x) * uint64_t(x - 1)) / 2;
}

enum DeviceStorageFmt: int {
    DEFAULT_6_BIT_HLL,
    PACKED_4_BIT_HLL,
    WIDE_8_BIT_HLL,
    BBMH
};


template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
INLINE void increment_maxes(T *arr, unsigned x1, unsigned x2) {
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

template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
INLINE void increment_maxes_packed16(T *arr, const unsigned x1, const unsigned x2) {
#if __CUDA_ARCH__
    static constexpr unsigned mask = 0x0F0F0F0Fu;
    unsigned tmp1 = x1 & mask, tmp2 = x2 & mask;
    tmp1 = __vmaxu4(tmp1, tmp2);
    ++arr[tmp1&0xFu];
    ++arr[(tmp1>>8)&0xFu];
    ++arr[(tmp1>>16)&0xFu];
    ++arr[tmp1>>24];
    tmp1 = __vmaxu4((x1>>4)&mask, (x2>>4)&mask);
    ++arr[tmp1&0xFu];
    ++arr[(tmp1>>8)&0xFu];
    ++arr[(tmp1>>16)&0xFu];
    ++arr[tmp1>>24];
#else
    using std::max;
    // Manual
    ++arr[max(x1>>28, x2>>28)];
    ++arr[max((x1>>24)&0xFu, (x2>>24)&0xFu)];
    ++arr[max((x1>>20)&0xFu, (x2>>20)&0xFu)];
    ++arr[max((x1>>16)&0xFu, (x2>>16)&0xFu)];
    ++arr[max((x1>>12)&0xFu, (x2>>12)&0xFu)];
    ++arr[max((x1>>8)&0xFu, (x2>>8)&0xFu)];
    ++arr[max((x1>>4)&0xFu, (x2>>4)&0xFu)];
    ++arr[max(x1&0xFu, x2&0xFu)];
#endif
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
    SK_UNROLL_8
    for(;jstart < jend; ++jstart) {
        lut[ij2ind(i, jstart, n)] = (uint64_t(i) << 32) | jstart;
    }
}

#endif // __CUDACC__

template<typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
INLINE double origest(const T &p, unsigned l2) {
    const auto m = 1u << l2;
    const double alpha = m == 16 ? .573 : m == 32 ? .697 : m == 64 ? .709: .7213 / (1. + 1.079 / m);
    double s = p[0];
    SK_UNROLL_8
    for(auto i = 1u; i < 64 - l2 + 1; ++i) {
#if __CUDA_ARCH__
        s += ldexpf(p[i], -i); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
#else
        s += std::ldexp(p[i], -i);
#endif
    }
    return alpha * m * m / s;
}

using namespace ::sketch::hll;
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
    SK_UNROLL_8
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
        //calc_sizes_large<<<nblocks,1024,mem_per_block>>>(p, l2, nhlls, nblocks, mem_per_block, sizes);
        throw std::runtime_error("Current implementation is limited to 1024 by 1024 comparisons. TODO: fix this with a reimplementation");
    } else {
        if((ce = cudaMalloc((void **)&sizes, nb)))
            throw CudaError(ce, "Failed to malloc");
        calc_sizes_1024<<<nhlls,(nhlls+1)/2,m>>>(p, l2, nhlls, sizes);
        if((ce = cudaDeviceSynchronize())) throw CudaError(ce, "Failed to synchronize");
        if((ce = cudaMemcpy(ret.data(), sizes, nb, cudaMemcpyHostToDevice))) throw CudaError(ce, "Failed to copy device to host");
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

#ifdef __CUDACC__

template<typename FT>
void __global__ original_hll_kernel(const uint8_t *const dmem,
            FT *drmem, size_t nelem, size_t rowind, size_t round_nrows, int tpb, int nblocks, int p, size_t entrysize, float alpha) {
    // int nblocks = (nelem_ * numrows_ * entrysize_ + tpb - 1) / tpb; // This means one per 256 bytes.
    extern __shared__ shared[];
    //uint32_t ls[64]{0}; // Number of integers: r. (meaning one never accesses a higher number)
    //const auto r = 64 - p + 1;
    if(threadIdx.x < 64) {
        shared[threadIdx.x] = 0;
    }
    __syncthreads();

    const int tid = threadIdx.x;
    int blockid = blockIdx.x;
    int rhsnum = blockid % (entrysize / tpb); // Tells us which in (nr * nelem) counting order we're using.
    int rhsidx = rhsnum % nelem;
    int lhsidx = rhsnum / nelem + rowind;
    int bytes_per_thread = entrysize / tpb;
    auto offset_within = (tid * bytes_per_thread);
    auto lhscmpptr = dmem + (entrysize * lhsidx) /* start of the entry */
                          + offset_within;
    auto rhscmpptr = dmem + (entrysize * rhsidx) + offset_within;
    // This has been recently cudaMemset'd, so we can do local accumulations and follow them up with global updates
    SK_UNROLL_4
    for(unsigned i = 0; i < bytes_per_thread / 4; ++i) {
        auto v = __vmaxu4((const uint32_t *)&lhscmpptr[i], (const uint32_t *) rhscmpptr[i]);
        atomicAdd(&shared[v&0xFF], 1);
        atomicAdd(&shared[(v>>8)&0xFF], 1);
        atomicAdd(&shared[(v>>16)&0xFF], 1);
        atomicAdd(&shared[(v>>24)&0xFF], 1);
    }
    if(tid < 64 - p + 1) {
        if(shared[tid]) {
            auto v = ::ldexpf(shared[tid], -tid)
            atomicAdd(drmem + rhsnum, v);
        }
    }
    cudaDeviceSynchronize();
    auto gid = threadIdx.x + (blockIdx.x * blockDim.x);
    if(gid < nelem * round_nrows) {
        drmem[gid] = alpha * nelem * nelem / drmem[gid];;
    }
}


struct GPUDistanceProcessor {
protected:
    uint8_t *dmem_; // device memory (for sketches)
    void *cmem_; // host memory (for returning values)
    void *drmem_; // device memory (for holding return values)
    size_t nelem_;
    size_t entrysize_;
    size_t numrows_;
    uint32_t use_float_:1, use_gz_:1;
    void *fp_ = nullptr;
    DeviceStorageFmt storage_ = DEFAULT_6_BIT_HLL;
public:
    auto nelem() const {return nelem_;}
    auto entrysize() const {return entrysize_;}
    bool flush_to_file() const {
        return nelem_ == numrows_;
    }
    uint32_t elemsz() const {return use_float_ ? 4: 8;}
    void perform_flush(size_t nrows=0) {
        if(nrows == 0) nrows = numrows_;
        assert(fp_);
        if(use_gz_) {
            if(gzwrite(static_cast<gzFile>(fp_), cmem_, nrows * nelem_ * elemsz()) != nelem_ * nrows * elemsz())
                throw ZlibError("Failed to write to file\n");
        } else {
            if(std::fwrite(cmem_, elemsz(), nelem_ * nrows, static_cast<std::FILE *>(fp_)) != nelem_ * nrows)
                throw std::runtime_error("Failed to write to file\n");
        }
    }
    void open_fp(const char *fp) {
        if(use_gz_) {
            if(fp_) gzclose(static_cast<gzFile>(fp_));
            fp_ = gzopen(fp, "wb");
        } else {
            if(fp_) std::fclose(static_cast<std::FILE *>(fp_));
            fp_ = std::fopen(fp, "w");
        }
    }
    GPUDistanceProcessor(const GPUDistanceProcessor &) = delete;
    GPUDistanceProcessor(GPUDistanceProcessor &&o) {
        std::memset(this, 0, sizeof(*this));
        std::swap_ranges(reinterpret_cast<uint8_t *>(this), reinterpret_cast<uint8_t *>(this) + sizeof(*this),
                         reinterpret_cast<uint8_t *>(&o));
    }
    GPUDistanceProcessor() {
        std::memset(this, 0, sizeof(*this));
    }
    void device_alloc() {
        size_t nb = entrysize_ * nelem_;
        cudaError_t ce;
        if(dmem_ && (ce = cudaFree(dmem_)))
            throw CudaError(ce, "Failed to free");
        if(drmem_ && (ce = cudaFree(drmem_)))
            throw CudaError(ce, "Failed to free");
        if((ce = cudaMalloc((void **)&dmem_, nb)))
            throw CudaError(ce, "Failed to alloc");
        if((ce = cudaMalloc((void **)&drmem_, nelem_ * elemsz() * numrows_)))
            throw CudaError(ce, "Failed to alloc");
        //std::fprintf(stderr, "cudaMalloc'd\nnb: %zu. dmem: %p", nb, (void *)dmem_);
    }
    GPUDistanceProcessor(size_t nelem, size_t entrysize, size_t numrows=0, bool use_float=true, bool use_gz=false): nelem_(nelem), entrysize_(entrysize), use_float_(use_float), use_gz_(use_gz)
    {
        drmem_ = nullptr;
        cmem_ = nullptr;
        dmem_ = nullptr;
        fp_ = nullptr;
        numrows_ = numrows == 0 ? nelem_: numrows; // Defaults to full matrix in memory
        device_alloc();
        if((cmem_ = std::malloc((elemsz()) * nelem_ * numrows_)) == nullptr)
            throw std::bad_alloc();
        std::fprintf(stderr, "just malloc'd cmem (%p). first byte: %d\n", (void *)cmem_, ((uint8_t *)cmem_)[0]);
#if 0
        cudaError_t ce;
        if((ce = cudaMalloc((void **)&dmem_, nelem * entrysize)))
            throw CudaError(ce, "Failed to cudamalloc");
#endif
    }
    virtual void *row_start(size_t rownum) const { // Override this if you want to skip over the re-computed values
        return static_cast<void *>(static_cast<uint8_t *>(cmem_) + (nelem_ * elemsz()));
    }
    virtual void *sketch_data(size_t index) const { // Override this if you want to skip over the re-computed values
        //std::fprintf(stderr, "sketch ptr for index %zu is %p + %zu (%p)\n", index, (void *)dmem_, (void *)(static_cast<uint8_t *>(dmem_) + entrysize_ * index));
        return static_cast<void *>(static_cast<uint8_t *>(dmem_) + entrysize_ * index);
    }
    void process_sketches() {
        switch(storage_) {
            case DEFAULT_6_BIT_HLL:
                process([this](void *drmem, size_t row_index, size_t round_nrows, int tpb, int nblocks){
                    auto dmem = dmem_;
                    auto n = nelem_;
                    auto p = ilog2(entrysize_); // p is log2 of size in bytes, which means that log2 of remainder is at most 64 - p + 1
                    auto r = 64 - p + 1;
                    cudaError_t ce;
                    auto alpha = entrysize_ == 16 ? .673: entrysize_ == 32 ? .697: entrysize_ == 64 ? .709: .7213 / (1. + 1.079 / entrysize_);
                    if((ce =  cudaMemset(drmem_, 0, round_nrows * elemsz()))) throw CudaError(ce, "Failed to cudaMemset.");
                    if(use_float_) {
                        original_hll_kernel<<<nblocks, tpb, r * sizeof(uint32_t)>>>(dmem, static_cast<float *>(drmem_), n, row_index, round_nrows, tpb, nblocks, p, entrysize_, alpha);
                    } else {
                        original_hll_kernel<<<nblocks, tpb, r * sizeof(uint32_t)>>>(dmem, static_cast<double *>(drmem_), n, row_index, round_nrows, tpb, nblocks, p, entrysize_, alpha);
                    }
                    // 6 bit kernel
                });
                break;
            case PACKED_4_BIT_HLL:
                process([dmem=dmem_,n=nelem_,esz=entrysize_](void *drmem, size_t row_index, size_t round_nrows, int tpb, int nblocks){
                     // 4 bit kernel
                });
                break;
            case WIDE_8_BIT_HLL:
                process([dmem=dmem_,n=nelem_,esz=entrysize_](void *drmem, size_t row_index, size_t round_nrows, int tpb, int nblocks){
                     // 8 bit kernel
                });
                break;
            case BBMH:
                process([dmem=dmem_,n=nelem_,esz=entrysize_](void *drmem, size_t row_index, size_t round_nrows, int tpb, int nblocks){
                     // b-bit kernel
                });
                break;
            default: {
                char buf[64];
                std::sprintf(buf, "Failed to process; unknown storage type: %d\n", storage_);
                throw std::runtime_error(buf);
            }
        }
    }
    template<typename F>
    void process(const F &f) { // Default, unpacked 8-bit HLLs
        if(entrysize_ < 1024) throw std::runtime_error("entrysize must be at least 1024. (4 for intrinsics, 256 for threads per block");
        assert(entrysize_ % 256 == 0);
        size_t rind    = 0; // Row index
        constexpr int tpb = 256;
        int nblocks = (nelem_ * numrows_ * entrysize_ + tpb - 1) / tpb;
        // This means one per 256 bytes.
        // TODO:
        std::thread copy_and_flush;
        cudaError_t ce;
        for(size_t tranche = 0, ntranches = (nelem_ - 1 + numrows_) / numrows_; tranche < ntranches; ++tranche) {
            size_t round_nrows = std::min(numrows_, nelem_ - numrows_);
            f(static_cast<uint8_t *>(drmem_) + (nelem_ * elemsz() * rind) /* local destination */,
              rind, round_nrows, tpb, nblocks);
#if 0
            original_hll_compare<<<nblocks, tpb, 64 * sizeof(uint32_t)>>>(
                dmem_,
                /* data = */
                drmem_ + (nelem_ * elemsz() * rind),
                /* local destination */
                rind, // round index
                round_nrows
            );
#elif 0
            packed_hll_compare<<<nblocks, tpb, 64 * sizeof(uint32_t)>>>(
                dmem_,
                /* data = */
                drmem_ + (nelem_ * elemsz() * rind),
                /* local destination */
                rind, // round index
                round_nrows
            );
#endif
            cudaDeviceSynchronize(); // Ensure computations have completed
            if(copy_and_flush.joinable()) copy_and_flush.join(); // Ensure that flushing to disk is completed.
            // At this point, the CPU buffer is available for loading.
            // TODO: double the memory on the device and compute the next portion
            // while transferring
            size_t nb = nelem_ * elemsz() * round_nrows;
            std::fprintf(stderr, "drmem %p. nb: %zu. round nr: %zu\n", (void *)drmem_, nb, round_nrows);
            std::fprintf(stderr, "cmem %p\n", (void *)cmem_);
            if((ce = cudaMemcpy(cmem_/* + (nelem_ * elemsz() * rind)*/, drmem_, nb, cudaMemcpyDeviceToHost)))
                throw CudaError(ce, "Couldn't copy back to computer\n");
            copy_and_flush = std::thread([&](){
                this->perform_flush(round_nrows);
            });
            rind += numrows_;
        }
        if(copy_and_flush.joinable()) copy_and_flush.join();
    }
    ~GPUDistanceProcessor() {
        cudaError_t ce;
        if((ce = cudaFree(dmem_))) {
            CudaError exception(ce, "Error called in GPUDistanceProcessor. (This will actually cause the program to fail");
            std::fprintf(stderr, "not throwing an error to avoid terminating the program. Message: %s\n", exception.what());
        }
        if((ce = cudaFree(drmem_))) {
            CudaError exception(ce, "Error called in GPUDistanceProcessor. (This will actually cause the program to fail");
            std::fprintf(stderr, "not throwing an error to avoid terminating the program. Message: %s\n", exception.what());
        }
        std::free(cmem_);
    }
};



template<typename It>
void copy_hlls(GPUDistanceProcessor &gp, It hllstart, It hllend, bool direct=true) {
    //using HllType = std::add_lvalue_reference_t<std::decay_t<decltype(*hllstart)>>;
    assert(std::distance(hllstart, hllend) == gp.nelem());
    size_t nbper = hllstart->size();
    assert(gp.entrysize() == nbper);
    auto tmp(std::make_unique<uint8_t []>(std::distance(hllstart, hllend) * nbper));
    auto dist = std::distance(hllstart, hllend);
    cudaError_t ce;
    if(!direct) {
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < dist; ++i) {
            It it = hllstart;
            std::advance(it, i); // Usually addition, but different if a weird iterator
            size_t offset = i * nbper;
            std::memcpy(tmp.get() + offset, it->data(), nbper);
        }
    } else {
        for(size_t i = 0; i < dist; ++i) {
            It it = hllstart;
            std::advance(it, i); // Usually addition, but different if a weird iterator
            //std::fprintf(stderr, "Index %zu\nCopying %zu bytes to %p from %p\n", i, nbper, (void *)gp.sketch_data(i), (void *)it->data());
            if((ce = cudaMemcpy(gp.sketch_data(i), it->data(), nbper, cudaMemcpyHostToDevice)))
                throw CudaError(ce, "Failed to copy to device");
        }
    }
}

template<typename It>
GPUDistanceProcessor setup_hlls(const char *fn, It hs, It he, int nr, bool direct=true) {
    GPUDistanceProcessor ret(std::distance(hs, he), hs->size(), nr);
    ret.open_fp(fn);
    copy_hlls(ret, hs, he, direct);
    return ret;
}

#endif /* #ifdef __CUDACC__ */

} // sketch

#endif

