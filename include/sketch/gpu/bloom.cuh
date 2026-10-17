#ifndef SKETCH_GPU_BLOOM_CUH__
#define SKETCH_GPU_BLOOM_CUH__
/*
 * GPU blocked-bloom-filter bit updates (bfbase_t, include/sketch/bf.h) and
 * counting-bloom counter updates, bit-identical on pre-hashed streams.
 *
 * bf bit-identity semantics mirrored from bfbase_t::addh + sub_set1 (bf.h):
 *   p     = l2sz (the filter spans m = 1 << l2sz bits, mask = m - 1)
 *   npw   = 64 / p               (lut::nhashesper64bitword[p], integer division)
 *   per element, per group g of seeds (group size gsize = min(npw, nh - g*npw)):
 *     hv = hf(element ^ seeds_[g])           // PRE-HASHED on the caller side
 *     for i in [0, gsize): set1(hv >> (i * p)) with set1(ind): ind &= mask;
 *                                                      core[ind >> 6] |= 1<<(ind & 63)
 * atomicOr is order-insensitive, so the GPU state equals the CPU state for
 * the same (element, seed-group) hash values. Feed hashes laid out
 * element-major with bf_ngroups(nh, npw) uint64s per element.
 *
 * cbf (counting bloom, classic) mirrors the counter layout of
 * ccmbase_t<update::Increment, ..., conservative_update=false>
 * (include/sketch/ccm.h) with 32-bit counters here: per element, per
 * subtable i in [0, nh):
 *   hv = hf(element ^ seeds_[i])             // PRE-HASHED on the caller side
 *   counters[(hv & mask) + (i << l2sz)] += 1 // mask = (1 << l2sz) - 1
 * Counter stores wrap mod 2^nbits on both sides (the CPU compact vector
 * masks on store); pass the sketch's nbits (default 32 = wrap mod 2^32, a
 * plain atomicAdd) and GPU totals equal CPU totals exactly.
 */
#include "sketch/gpu/common.cuh"

namespace skgpu {

SKGPU_HD inline constexpr std::uint32_t bf_npw(std::uint32_t p) noexcept {
    return 64 / p; // == lut::nhashesper64bitword[p] for p in [1, 63]
}
SKGPU_HD inline constexpr std::uint32_t bf_ngroups(std::uint32_t nh, std::uint32_t npw) noexcept {
    return (nh + npw - 1) / npw;
}
SKGPU_HD inline constexpr std::uint32_t bf_group_size(std::uint32_t g, std::uint32_t nh, std::uint32_t npw) noexcept {
    const std::uint32_t rem = nh - g * npw;
    return rem < npw ? rem: npw;
}
// One set1 on a word array (host-usable reference; kernel uses atomicOr).
inline void bf_set1(std::uint64_t *SKGPU_RESTRICT core, std::uint64_t mask, std::uint64_t ind) noexcept {
    ind &= mask;
    core[ind >> 6] |= std::uint64_t(1) << (ind & 63);
}

} // namespace skgpu

#if SKGPU_CUDA
namespace skgpu {
SKGPU_GLOBAL void bf_update_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, // n_elements * ngroups pre-hashed values
                                   std::size_t total, std::uint32_t ngroups,
                                   std::uint32_t nh, std::uint32_t npw, std::uint32_t shift,
                                   std::uint64_t mask, std::uint64_t *SKGPU_RESTRICT core);
SKGPU_GLOBAL void cbf_add_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, // n_elements * nh pre-hashed values
                                 std::size_t total, std::uint32_t nh, std::uint32_t l2sz, std::uint32_t nbits,
                                 std::uint32_t *SKGPU_RESTRICT counters);

// d_core: (1 << (l2sz - 6)) uint64 words (2^l2sz bits total), zero-initialized.
inline void bf_update(const std::uint64_t *d_hashes, std::size_t n_elements, std::uint32_t l2sz,
                      std::uint32_t nh, std::uint64_t *d_core, unsigned tpb = 256) {
    if(l2sz < 7 || l2sz > 63) throw std::runtime_error(std::string("[skgpu::bf] l2sz must be in [7, 63], got ") + std::to_string(l2sz));
    const std::uint32_t npw = bf_npw(l2sz), ng = bf_ngroups(nh, npw);
    bf_update_kernel<<<nblocks(n_elements * ng, tpb), tpb>>>(d_hashes, n_elements * ng, ng, nh, npw, l2sz, (std::uint64_t(1) << l2sz) - 1, d_core);
    SKGPU_CK(cudaGetLastError());
}
// d_counters: nh << l2sz uint32 counters, zero-initialized. nbits is the
// counter width of the CPU reference sketch (counts wrap mod 2^nbits; 32 for
// a plain mod-2^32 counting bloom).
inline void cbf_add(const std::uint64_t *d_hashes, std::size_t n_elements, std::uint32_t nh,
                    std::uint32_t l2sz, std::uint32_t *d_counters, std::uint32_t nbits = 32, unsigned tpb = 256) {
    if(l2sz > 31) throw std::runtime_error(std::string("[skgpu::cbf] l2sz must be <= 31, got ") + std::to_string(l2sz));
    if(nbits == 0 || nbits > 32) throw std::runtime_error(std::string("[skgpu::cbf] nbits must be in [1, 32], got ") + std::to_string(nbits));
    cbf_add_kernel<<<nblocks(n_elements * nh, tpb), tpb>>>(d_hashes, n_elements * nh, nh, l2sz, nbits, d_counters);
    SKGPU_CK(cudaGetLastError());
}

} // namespace skgpu
#endif // SKGPU_CUDA

#endif // SKETCH_GPU_BLOOM_CUH__
