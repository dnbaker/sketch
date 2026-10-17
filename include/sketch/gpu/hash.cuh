#ifndef SKETCH_GPU_HASH_CUH__
#define SKETCH_GPU_HASH_CUH__
/*
 * Device path for sketch::hash::SplitMix64Hash (include/sketch/hash.h).
 * The finalizer is a bijection: xorshift-multiply with two odd multiplies
 * (public-domain splitmix64 constants), applied here bit-for-bit:
 *
 *   x ^= x >> 30; x *= 0xbf58476d1ce4e5b9;
 *   x ^= x >> 27; x *= 0x94d049bb133111eb;
 *   x ^= x >> 31;
 *
 * (CEIFused<RShiftXor<30>, CEIMul<0xbf58476d1ce4e5b9>, RShiftXor<27>,
 *  CEIMul<0x94d049bb133111eb>, RShiftXor<31>>; both seeds are odd, so the
 *  CEIMul `| 1` is a no-op.) Multiplication is mod 2^64 on both sides.
 */
#include "sketch/gpu/common.cuh"

namespace skgpu {

SKGPU_HD inline constexpr std::uint64_t splitmix64_hash(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

} // namespace skgpu

#if SKGPU_CUDA
namespace skgpu {
SKGPU_GLOBAL void hash_kernel(const std::uint64_t *SKGPU_RESTRICT in, std::uint64_t *SKGPU_RESTRICT out, std::size_t n);

inline void hash(const std::uint64_t *d_in, std::uint64_t *d_out, std::size_t n, unsigned tpb = 256) {
    hash_kernel<<<nblocks(n, tpb), tpb>>>(d_in, d_out, n);
    SKGPU_CK(cudaGetLastError());
}
} // namespace skgpu
#endif

#endif // SKETCH_GPU_HASH_CUH__
