#ifndef SKETCH_GPU_MINHASH_CUH__
#define SKETCH_GPU_MINHASH_CUH__
/*
 * Independent-sketch batch updates for b-bit minhash (bbmh) and fractional
 * minhash (fmh) — the dashing2 use case: (sketch_index, hash) streams
 * updating thousands of independent sketches in parallel.
 *
 * bbmh bit-identity semantics mirrored from BBitMinHasher<T>::add
 * (include/sketch/bbmh.h, T = uint64_t):
 *   bucket = hv >> (64 - p)          // top p bits select the register
 *   val    = (hv << p) >> p          // remaining 64 - p bits
 *   core[bucket] = min(core[bucket], val)
 * Sketch s lives at core[s << p .. (s+1) << p) (sketch-per-row layout),
 * initialized to default_val<T>() = UINT64_MAX. atomicMin on u64 is
 * order-insensitive, so the GPU rows equal the CPU sketches exactly.
 *
 * fmh bit-identity semantics mirrored from fracminhashbase_t::add
 * (include/sketch/fracminhash.h): keep hashes with
 *   hv <= max_hash && hv != ~uint64_t(0)
 * (the ~0 sentinel is the CPU set's empty marker). The GPU kernel stream-
 * compacts kept hashes into out (order nondeterministic; sort before
 * comparing) and maintains per-sketch kept counts and running minima
 * (fracminhashbase_t::min_hash), which are order-insensitive.
 */
#include "sketch/gpu/common.cuh"

namespace skgpu {

SKGPU_HD inline constexpr std::uint32_t bbmh_bucket(std::uint64_t hv, std::uint32_t p) noexcept {
    return std::uint32_t(hv >> (64 - p));
}
SKGPU_HD inline constexpr std::uint64_t bbmh_val(std::uint64_t hv, std::uint32_t p) noexcept {
    return (hv << p) >> p;
}
SKGPU_HD inline constexpr bool fmh_keep(std::uint64_t hv, std::uint64_t max_hash) noexcept {
    return hv <= max_hash && hv != ~std::uint64_t(0);
}

} // namespace skgpu

#if SKGPU_CUDA
namespace skgpu {
SKGPU_GLOBAL void bbmh_batch_update_kernel(const std::uint32_t *SKGPU_RESTRICT sketch_ids, // n pairs
                                           const std::uint64_t *SKGPU_RESTRICT hashes,     // n pre-hashed values
                                           std::size_t n, std::uint32_t p,
                                           std::uint64_t *SKGPU_RESTRICT core);            // nsketches << p registers
SKGPU_GLOBAL void fmh_batch_filter_kernel(const std::uint32_t *SKGPU_RESTRICT sketch_ids,
                                          const std::uint64_t *SKGPU_RESTRICT hashes,
                                          std::size_t n, std::uint64_t max_hash,
                                          std::uint64_t *SKGPU_RESTRICT kept,       // capacity >= kept count
                                          std::uint32_t *SKGPU_RESTRICT kept_sids,  // capacity >= kept count
                                          std::uint32_t *SKGPU_RESTRICT counts,     // nsketches, zero-initialized
                                          std::uint64_t *SKGPU_RESTRICT mins,       // nsketches, initialized to ~0
                                          std::uint32_t *SKGPU_RESTRICT ticket);    // 1, zero-initialized

// d_core: nsketches << p uint64 registers, initialized to ~0. p in [1, 63].
inline void bbmh_batch_update(const std::uint32_t *d_sids, const std::uint64_t *d_hashes, std::size_t n,
                              std::uint32_t p, std::uint64_t *d_core, unsigned tpb = 256) {
    if(p == 0 || p >= 64) throw std::runtime_error(std::string("[skgpu::bbmh] p must be in [1, 63], got ") + std::to_string(p));
    bbmh_batch_update_kernel<<<nblocks(n, tpb), tpb>>>(d_sids, d_hashes, n, p, d_core);
    SKGPU_CK(cudaGetLastError());
}
// d_mins must be initialized to ~0 (cudaMemset(d_mins, 0xFF, nsketches *
// sizeof(uint64_t)) works); d_counts/d_ticket zero-initialized. d_kept and
// d_kept_sids must have room for every passing hash (n entries always
// suffices); kept_sids[i] is the sketch of kept[i].
inline void fmh_batch_filter(const std::uint32_t *d_sids, const std::uint64_t *d_hashes, std::size_t n,
                             std::uint64_t max_hash, std::uint64_t *d_kept, std::uint32_t *d_kept_sids,
                             std::uint32_t *d_counts, std::uint64_t *d_mins, std::uint32_t *d_ticket,
                             unsigned tpb = 256) {
    fmh_batch_filter_kernel<<<nblocks(n, tpb), tpb>>>(d_sids, d_hashes, n, max_hash, d_kept, d_kept_sids, d_counts, d_mins, d_ticket);
    SKGPU_CK(cudaGetLastError());
}

} // namespace skgpu
#endif // SKGPU_CUDA

#endif // SKETCH_GPU_MINHASH_CUH__
