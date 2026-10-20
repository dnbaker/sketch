#ifndef SKETCH_GPU_COMMON_CUH__
#define SKETCH_GPU_COMMON_CUH__
/*
 * Common scaffolding for the sketch GPU layer (CUDA, hipify-cl portable to
 * HIP/AMD and ZLUDA). Nothing in this header (or any other gpu .cuh header)
 * is included by the CPU headers; the layer is opt-in via explicit include.
 *
 * Conventions:
 *  - Scalar math lives in SKGPU_HD constexpr functions with portable
 *    implementations so the SAME code compiles for g++/clang++ host TUs
 *    (used by tests/host_math_check.cpp on GPU-less machines) and for the
 *    nvcc/hipcc device pass. Device-only intrinsics (__clzll, atomics,
 *    __global__) are confined to thin wrappers guarded by SKGPU_CUDA /
 *    SKGPU_DEVICE_PASS.
 *  - Kernels are __global__ declarations here and defined thinly in
 *    gpu/sketch.cu; host launchers are inline in these headers.
 *  - The bit-identity contract: given the same pre-hashed uint64 stream,
 *    GPU register/bloom/minhash state is bit-identical to the CPU
 *    implementations in hll.h / bf.h / ccm.h / bbmh.h / fracminhash.h.
 *    Update operations are order-insensitive (max/min/add/or), so atomic
 *    parallelism does not change results.
 */
#include <cstdint>
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP__)
#  define SKGPU_CUDA 1
#else
#  define SKGPU_CUDA 0
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#  define SKGPU_DEVICE_PASS 1
#else
#  define SKGPU_DEVICE_PASS 0
#endif

#if SKGPU_CUDA
#  define SKGPU_HD __host__ __device__
#  define SKGPU_DEVICE __device__
#  define SKGPU_GLOBAL __global__
#  define SKGPU_RESTRICT __restrict__
#else
/* CPU-only build: everything CUDA-specific below compiles away; the scalar
 * math remains available for host-side reference implementations. */
#  define SKGPU_HD
#  define SKGPU_DEVICE
#  define SKGPU_GLOBAL
#  define SKGPU_RESTRICT
#endif

namespace skgpu {

// Portable 64-bit count-leading-zeros. __clzll(0) is documented as 64 on
// device; __builtin_clzll(0) is UB on the host (we return 64 there). All
// sketch uses below keep a sentinel bit set (mirroring the CPU headers), so
// the x == 0 case never reaches this function in practice.
SKGPU_HD inline constexpr std::uint32_t clz64(std::uint64_t x) noexcept {
#if SKGPU_DEVICE_PASS
    return std::uint32_t(__clzll(x));
#elif defined(__GNUC__) || defined(__clang__)
    return x ? std::uint32_t(__builtin_clzll(x)): std::uint32_t(64);
#else
    std::uint32_t n = 0;
    while(!(x & (std::uint64_t(1) << 63))) {++n; x <<= 1;}
    return n;
#endif
}

// Grid-stride loop bound helper: number of blocks for n items at tpb
// threads, capped so launch dims stay well inside grid limits; the kernels
// grid-stride, so a cap only costs iterations, not correctness.
inline std::uint32_t nblocks(std::size_t n, std::uint32_t tpb, std::uint32_t max_blocks = 1u << 22) {
    const std::uint64_t want = std::uint64_t(n + tpb - 1) / tpb;
    return std::uint32_t(want == 0 ? 1: (want < max_blocks ? want: max_blocks));
}

} // namespace skgpu

#if SKGPU_CUDA
#  include <cuda_runtime.h>
#  include <stdexcept>
#  include <string>
namespace skgpu {
// One-line error check for runtime API calls, dnbaker-style (throw on error).
#define SKGPU_CK(x) do {\
    const cudaError_t skgpu_ce_ = (x);\
    if(skgpu_ce_ != cudaSuccess) \
        throw std::runtime_error(std::string("[skgpu:" __FILE__ ":") + std::to_string(__LINE__) + "] " \
                                 + cudaGetErrorString(skgpu_ce_) + " for " #x);\
} while(0)

// Counter increment with CPU compact-vector wrap semantics: ccmbase_t stores
// through a compact vector, so a stored count is (value & ((1 << nbits) - 1)).
// nbits >= 32 reduces to a plain atomicAdd (wrap mod 2^32, no mask needed).
SKGPU_DEVICE inline void counter_add_wrap(std::uint32_t *p, std::uint32_t nbits) {
    if(nbits >= 32) {
        atomicAdd(p, 1u);
    } else {
        const std::uint32_t mask = (std::uint32_t(1) << nbits) - 1;
        std::uint32_t old = *p;
        for(;;) {
            const std::uint32_t assumed = old;
            old = atomicCAS(p, assumed, (assumed + 1) & mask);
            if(old == assumed) break;
        }
    }
}

inline void *malloc_dev(std::size_t nb) {
    void *ret = nullptr;
    SKGPU_CK(cudaMalloc(&ret, nb ? nb: 1));
    return ret;
}
inline void free_dev(void *p) { if(p) SKGPU_CK(cudaFree(p)); }
template<typename T>
inline T *memcpy_h2d(const T *src, std::size_t n) {
    void *dst = malloc_dev(n * sizeof(T));
    SKGPU_CK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
    return static_cast<T *>(dst);
}
template<typename T>
inline void memcpy_d2h(T *dst, const T *src, std::size_t n) {
    SKGPU_CK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
}
} // namespace skgpu
#endif // SKGPU_CUDA

#endif // SKETCH_GPU_COMMON_CUH__
