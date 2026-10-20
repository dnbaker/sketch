// Thin __global__ definitions for the sketch GPU layer. All semantics live
// in SKGPU_HD inline helpers in include/sketch/gpu/*.cuh; these bodies are
// grid-stride loops only. Compiled by nvcc (or hipcc via hipify-cl); nothing
// in the CPU build references this translation unit.
#include "sketch/gpu/hash.cuh"
#include "sketch/gpu/hll.cuh"
#include "sketch/gpu/bloom.cuh"
#include "sketch/gpu/count.cuh"
#include "sketch/gpu/minhash.cuh"

namespace skgpu {

SKGPU_GLOBAL void hash_kernel(const std::uint64_t *SKGPU_RESTRICT in, std::uint64_t *SKGPU_RESTRICT out, std::size_t n) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = splitmix64_hash(in[i]);
}

SKGPU_GLOBAL void hll_update_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, std::size_t n, std::uint32_t np, std::uint8_t *SKGPU_RESTRICT core) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint64_t hv = hashes[i];
        hll_byte_atomic_max(core, hll_index(hv, np), hll_lzt(hv, np));
    }
}

SKGPU_GLOBAL void hll_update_hash_kernel(const std::uint64_t *SKGPU_RESTRICT in, std::size_t n, std::uint32_t np, std::uint8_t *SKGPU_RESTRICT core) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint64_t hv = splitmix64_hash(in[i]);
        hll_byte_atomic_max(core, hll_index(hv, np), hll_lzt(hv, np));
    }
}

SKGPU_GLOBAL void hll_merge_kernel(std::uint8_t *SKGPU_RESTRICT out, const std::uint8_t *SKGPU_RESTRICT src, std::size_t ncores, std::size_t m) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += stride) {
        std::uint8_t v = src[i];
        for(std::size_t s = 1; s < ncores; ++s) {
            const std::uint8_t w = src[s * m + i];
            v = w > v ? w: v;
        }
        out[i] = v;
    }
}

SKGPU_GLOBAL void bf_update_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, std::size_t total, std::uint32_t ngroups,
                                   std::uint32_t nh, std::uint32_t npw, std::uint32_t shift,
                                   std::uint64_t mask, std::uint64_t *SKGPU_RESTRICT core) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const std::uint32_t g = std::uint32_t(idx % ngroups);
        const std::uint64_t hv = hashes[idx];
        const std::uint32_t gsize = bf_group_size(g, nh, npw);
        for(std::uint32_t i = 0; i < gsize; ++i) {
            const std::uint64_t ind = (hv >> (i * shift)) & mask;
            atomicOr(reinterpret_cast<unsigned long long *>(&core[ind >> 6]), std::uint64_t(1) << (ind & 63));
        }
    }
}

SKGPU_GLOBAL void cbf_add_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, std::size_t total, std::uint32_t nh, std::uint32_t l2sz,
                                 std::uint32_t nbits, std::uint32_t *SKGPU_RESTRICT counters) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const std::uint32_t i = std::uint32_t(idx % nh);
        counter_add_wrap(&counters[ccm_index(hashes[idx], i, l2sz)], nbits);
    }
}

SKGPU_GLOBAL void ccm_add_kernel(const std::uint64_t *SKGPU_RESTRICT hashes, std::size_t total, std::uint32_t nhashes, std::uint32_t l2sz,
                                 std::uint32_t nbits, std::uint32_t *SKGPU_RESTRICT counters) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const std::uint32_t i = std::uint32_t(idx % nhashes);
        counter_add_wrap(&counters[ccm_index(hashes[idx], i, l2sz)], nbits);
    }
}

SKGPU_GLOBAL void bbmh_batch_update_kernel(const std::uint32_t *SKGPU_RESTRICT sketch_ids, const std::uint64_t *SKGPU_RESTRICT hashes,
                                           std::size_t n, std::uint32_t p, std::uint64_t *SKGPU_RESTRICT core) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        std::uint64_t *row = core + (std::size_t(sketch_ids[i]) << p);
        atomicMin(reinterpret_cast<unsigned long long *>(&row[bbmh_bucket(hashes[i], p)]),
                  (unsigned long long)bbmh_val(hashes[i], p));
    }
}

SKGPU_GLOBAL void fmh_batch_filter_kernel(const std::uint32_t *SKGPU_RESTRICT sketch_ids, const std::uint64_t *SKGPU_RESTRICT hashes,
                                          std::size_t n, std::uint64_t max_hash,
                                          std::uint64_t *SKGPU_RESTRICT kept,
                                          std::uint32_t *SKGPU_RESTRICT kept_sids,
                                          std::uint32_t *SKGPU_RESTRICT counts,
                                          std::uint64_t *SKGPU_RESTRICT mins,
                                          std::uint32_t *SKGPU_RESTRICT ticket) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for(std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        const std::uint64_t h = hashes[i];
        if(!fmh_keep(h, max_hash)) continue;
        const std::uint32_t pos = atomicAdd(ticket, 1u);
        kept[pos] = h;
        kept_sids[pos] = sketch_ids[i];
        atomicAdd(&counts[sketch_ids[i]], 1u);
        atomicMin(reinterpret_cast<unsigned long long *>(&mins[sketch_ids[i]]), (unsigned long long)h);
    }
}

} // namespace skgpu
