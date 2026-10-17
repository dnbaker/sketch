#ifndef SKETCH_GPU_TESTS_REFERENCE_H__
#define SKETCH_GPU_TESTS_REFERENCE_H__
/*
 * CPU-side reference computations for the GPU bit-identity tests.
 *
 * nvcc must never see the CPU sketch headers (frontend ICEs and
 * include-order-dependent constexpr errors observed in the wild on CUDA
 * 12.8.1), so ALL code that includes sketch/*.h lives in the plain-g++
 * translation unit tests/reference.cpp. This interface uses only POD and
 * stdlib types — test_gpu_sketches.cu includes nothing else.
 *
 * Conventions: every function fills caller-provided output arrays and, where
 * the GPU kernel consumes a pre-hashed stream, that stream (so the test
 * uploads exactly what the CPU reference was built from).
 */
#include <cstddef>
#include <cstdint>

namespace skref {

// sketch::hash::SplitMix64Hash
std::uint64_t splitmix64(std::uint64_t x);

// Pre-hashed values per element for skgpu::bf_update (== the CPU bf's seed
// count, i.e. ceil(nh / (64 / l2sz))).
std::uint32_t bf_ngroups(std::uint32_t l2sz, std::uint32_t nh);
// CPU bfbase_t reference (l2sz, nh, seed 137). Fills flat (n * bf_ngroups(l2sz, nh)
// uint64s, element-major: WangHash(els[e] ^ seeds[g])) — exactly the stream
// skgpu::bf_update consumes — and core ((1 << (l2sz - 6)) uint64 words).
void bf(const std::uint64_t *els, std::size_t n, std::uint32_t l2sz, std::uint32_t nh,
        std::uint64_t *flat, std::uint64_t *core);

// CPU hllbase_t reference registers (m = 1 << np bytes) over a pre-hashed stream.
void hll_update(const std::uint64_t *hashes, std::size_t n, std::uint32_t np, std::uint8_t *core);
// Same, but the CPU hashes the stream with SplitMix64Hash first (== what
// skgpu::hll_update_hashed does on device).
void hll_update_hashed(const std::uint64_t *in, std::size_t n, std::uint32_t np, std::uint8_t *core);
// Union of three equal-length streams (repeated hllbase_t::operator+=):
// merged registers in core (1 << np bytes) and the CPU estimate over them.
void hll_merge3(const std::uint64_t *s0, const std::uint64_t *s1, const std::uint64_t *s2,
                std::size_t n, std::uint32_t np, std::uint8_t *core, double *report);
// The documented "estimation stays on CPU" path: load raw registers into an
// hllbase_t and return its estimate. Used to check GPU-merged registers give
// the same cardinality as the CPU-merged sketch.
double hll_report(const std::uint8_t *core, std::uint32_t np);

// ccmbase_t<update::Increment, ..., conservative_update=false> reference.
// Fills flat (n * nh uint64s: WangHash(els[e] ^ seeds[i]), seeds generated as
// in ccmbase_t: mt19937_64(seed + 4)) and counters (nh << l2sz uint32s; counts
// wrap mod 2^nbits like the CPU compact-vector store). Used for both the cbf
// and ccm kernels (identical update semantics). nbits <= 5 keeps the CPU
// reference functional — the non-conservative path passes maxv to range_check
// where a bit width belongs, so nbits >= 6 suppresses every update on x86
// (ccm.h quirk; see gpu/README.md and the comments in tests).
void ccm(const std::uint64_t *els, std::size_t n, std::uint32_t l2sz, std::uint32_t nh,
         std::uint64_t seed, std::uint32_t nbits, std::uint64_t *flat, std::uint32_t *counters);

// BBitMinHasher<uint64_t> reference: nsk independent sketches, sketch s at
// core[s << p, (s+1) << p) (initialized to ~0 by the caller's comparison only;
// the returned rows are the final register values).
void bbmh(const std::uint32_t *sids, const std::uint64_t *hvs, std::size_t n,
          std::uint32_t p, std::uint32_t nsk, std::uint64_t *core);

// fracminhashbase_t reference over the same pre-hashed (sid, hv) stream.
// Fills kept_out/kept_sid_out (caller provides room for n entries), counts_out
// and mins_out (nsk entries each; mins are ~0 for sketches with no kept hash).
// Returns the number of kept hashes.
std::size_t fmh(const std::uint32_t *sids, const std::uint64_t *hvs, std::size_t n,
                std::uint64_t max_hash, std::uint32_t nsk,
                std::uint64_t *kept_out, std::uint32_t *kept_sid_out,
                std::uint32_t *counts_out, std::uint64_t *mins_out);

} // namespace skref

#endif // SKETCH_GPU_TESTS_REFERENCE_H__
