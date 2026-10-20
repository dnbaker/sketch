#ifndef SKETCH_GPU_COUNT_CUH__
#define SKETCH_GPU_COUNT_CUH__
/*
 * GPU count-min sketch (ccm) counter updates, bit-identical to
 * ccmbase_t<update::Increment, ..., conservative_update=false>
 * (include/sketch/ccm.h) with matching count wrap, on pre-hashed streams.
 *
 * Semantics mirrored from ccmbase_t::add with conservative_update = false:
 *   per element, per subtable i in [0, nhashes):
 *     hv    = hash(x ^ seeds_[i])          // PRE-HASHED on the caller side
 *     index = (hv & mask) + ((1 << l2sz) * i)
 *     data[index] += 1                     // Increment
 * Counter stores wrap mod 2^nbits on both sides (the CPU compact vector
 * masks on store), which the nbits parameter reproduces (nbits = 32 is a
 * plain atomicAdd). NOTE: the CPU non-conservative path passes
 * maxv = 1 << nbits to range_check where a bit width is expected, so for
 * nbits >= 6 the range_check shift count is a UB power of two >= 64 and
 * every update is silently suppressed on x86 (a ccm.h quirk — use nbits <= 5
 * when diffing against that reference, or diff against a hand-rolled
 * counter array, as tests/host_math_check.cpp does).
 *
 * The conservative-update variant (increment only the tied-minimum counters)
 * does not commute across elements, so a bit-identical GPU version needs one
 * thread per element and a collision-free stream (or warp-aggregated
 * min-then-update); that is planned work — see gpu/DESIGN.md. The
 * conservative=false update is the one dashing2-style pipelines use.
 */
#include "sketch/gpu/common.cuh"

namespace skgpu {

SKGPU_HD inline constexpr std::uint64_t ccm_index(std::uint64_t hv, std::uint32_t i, std::uint32_t l2sz) noexcept {
    return (hv & ((std::uint64_t(1) << l2sz) - 1)) + (std::uint64_t(i) << l2sz);
}

} // namespace skgpu

#if SKGPU_CUDA
namespace skgpu {
SKGPU_GLOBAL void ccm_add_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, // n_elements * nhashes pre-hashed values
                                 std::size_t total, std::uint32_t nhashes, std::uint32_t l2sz, std::uint32_t nbits,
                                 std::uint32_t *SKGPU_RESTRICT counters);

// d_counters: (nhashes << l2sz) uint32 counters, zero-initialized. nbits is
// the CPU reference's counter width (counts wrap mod 2^nbits).
inline void ccm_add(const std::uint64_t *d_hashes, std::size_t n_elements, std::uint32_t nhashes,
                    std::uint32_t l2sz, std::uint32_t *d_counters, std::uint32_t nbits = 32, unsigned tpb = 256) {
    if(l2sz > 31) throw std::runtime_error(std::string("[skgpu::ccm] l2sz must be <= 31, got ") + std::to_string(l2sz));
    if(nbits == 0 || nbits > 32) throw std::runtime_error(std::string("[skgpu::ccm] nbits must be in [1, 32], got ") + std::to_string(nbits));
    ccm_add_kernel<<<nblocks(n_elements * nhashes, tpb), tpb>>>(d_hashes, n_elements * nhashes, nhashes, l2sz, nbits, d_counters);
    SKGPU_CK(cudaGetLastError());
}

} // namespace skgpu
#endif // SKGPU_CUDA

#endif // SKETCH_GPU_COUNT_CUH__
