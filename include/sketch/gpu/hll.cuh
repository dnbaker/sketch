#ifndef SKETCH_GPU_HLL_CUH__
#define SKETCH_GPU_HLL_CUH__
/*
 * GPU HLL register updates + register merges, bit-identical to
 * sketch::hll::hllbase_t (include/sketch/hll.h) on the same pre-hashed
 * uint64 stream.
 *
 * Bit-identity semantics mirrored from hllbase_t::add (hll.h):
 *   q     = 64 - np
 *   index = (q == 64) ? 0 : uint32_t(hashval >> q)      // top np bits
 *   lzt   = clz(((hashval << 1) | 1) << (np - 1)) + 1   // sentinel keeps clz operand nonzero
 *   core[index] = max(core[index], lzt)                 // byte-wise, order-insensitive
 * np must be in [1, 62] (hllbase_t rejects np >= 63; np == 0 is a clz
 * shift-by-2^32-1 UB on CPU too). Register values never exceed q + 1 <= 64,
 * so a byte per register matches the CPU core layout exactly.
 *
 * Merge mirrors operator+= (hll.h): per-register byte-wise max over any
 * number of cores = sketch union.
 *
 * Cardinality estimation stays on CPU for now: copy the merged registers
 * back and reuse sketch::hll::hllbase_t::report (or detail::ertl_ml_estimate
 * over detail::sum_counts) unchanged.
 *
 * Atomic note: CUDA has no 8-bit atomicMax, so the register update uses the
 * classic word-level CAS loop on the containing aligned uint32 (each byte
 * changes finitely many times and only grows, so the loop terminates and the
 * result is exactly the per-byte max regardless of interleaving). For heavy
 * contention, a shared-memory privatized variant (block-local register
 * copies, byte max in shared memory, one global merge at the end) is the
 * planned optimization; see gpu/DESIGN.md.
 */
#include "sketch/gpu/common.cuh"
#include "sketch/gpu/hash.cuh"

namespace skgpu {

SKGPU_HD inline constexpr std::uint32_t hll_index(std::uint64_t hashval, std::uint32_t np) noexcept {
    return (64 - np) == 64 ? std::uint32_t(0): std::uint32_t(hashval >> (64 - np));
}
SKGPU_HD inline constexpr std::uint8_t hll_lzt(std::uint64_t hashval, std::uint32_t np) noexcept {
    return std::uint8_t(clz64(((hashval << 1) | 1) << (np - 1)) + 1);
}

} // namespace skgpu

#if SKGPU_CUDA
namespace skgpu {

// Byte-wise atomic max on a byte array whose base is 4-byte aligned and
// whose length is a multiple of 4 (true for HLL cores: m = 1 << np, np >= 2;
// cudaMalloc bases are 256-byte aligned).
SKGPU_DEVICE inline void hll_byte_atomic_max(std::uint8_t *SKGPU_RESTRICT core, std::uint32_t idx, std::uint8_t val) {
    std::uint32_t *word = reinterpret_cast<std::uint32_t *>(core) + (idx >> 2);
    const std::uint32_t shift = (idx & 3u) * 8u;
    const std::uint32_t bmask = 0xFFu << shift;
    std::uint32_t old = *word;
    while(std::uint8_t((old >> shift) & 0xFFu) < val) {
        const std::uint32_t assumed = old;
        old = atomicCAS(word, assumed, (assumed & ~bmask) | (std::uint32_t(val) << shift));
        if(old == assumed) break;
    }
}

SKGPU_GLOBAL void hll_update_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, std::size_t n, std::uint32_t np, std::uint8_t *SKGPU_RESTRICT core);
SKGPU_GLOBAL void hll_update_hash_kernel(const std::uint64_t *SKGPU_RESTRICT in, std::size_t n, std::uint32_t np, std::uint8_t *SKGPU_RESTRICT core);
SKGPU_GLOBAL void hll_merge_kernel(std::uint8_t *SKGPU_RESTRICT out, const std::uint8_t *SKGPU_RESTRICT src, std::size_t ncores, std::size_t m);

inline void hll_np_check(std::uint32_t np) {
    if(np == 0 || np >= 63) throw std::runtime_error(std::string("[skgpu::hll] np must be in [1, 62], got ") + std::to_string(np));
}

// Pre-hashed stream: hashes[i] is the 64-bit HLL hash value (what CPU
// hllbase_t::add takes). d_core must be zero-initialized (cudaMemset) with
// 1 << np bytes.
inline void hll_update(const std::uint64_t *d_hashes, std::size_t n, std::uint32_t np, std::uint8_t *d_core, unsigned tpb = 256) {
    hll_np_check(np);
    hll_update_kernel<<<nblocks(n, tpb), tpb>>>(d_hashes, n, np, d_core);
    SKGPU_CK(cudaGetLastError());
}
// Variant that hashes on device with skgpu::splitmix64_hash first
// (bit-identical to hllbase_t<sketch::hash::SplitMix64Hash>::addh).
inline void hll_update_hashed(const std::uint64_t *d_in, std::size_t n, std::uint32_t np, std::uint8_t *d_core, unsigned tpb = 256) {
    hll_np_check(np);
    hll_update_hash_kernel<<<nblocks(n, tpb), tpb>>>(d_in, n, np, d_core);
    SKGPU_CK(cudaGetLastError());
}
// Union of ncores register arrays of m bytes each (byte-wise max), mirroring
// repeated hllbase_t::operator+=. out must not overlap src[1..], but out ==
// src row 0 is fine (each output byte depends only on input byte i).
inline void hll_merge(std::uint8_t *d_out, const std::uint8_t *d_src, std::size_t ncores, std::size_t m, unsigned tpb = 256) {
    hll_merge_kernel<<<nblocks(m, tpb), tpb>>>(d_out, d_src, ncores, m);
    SKGPU_CK(cudaGetLastError());
}

} // namespace skgpu
#endif // SKGPU_CUDA

#endif // SKETCH_GPU_HLL_CUH__
