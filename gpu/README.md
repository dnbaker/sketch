# sketch: GPU layer (CUDA / HIP-portable)

Opt-in CUDA kernels for the sketch data structures, single-sourced for
`hipify-cl` so the same files build for HIP/AMD (and run under ZLUDA on
AMD-with-CUDA-translation). **Nothing here is included by the CPU headers**;
CPU-only builds of the repo are untouched. The layer is plain
`__global__`/`__device__` kernels + blockDim/gridDim/threadIdx indexing + the
runtime API (`cudaMalloc`/`cudaMemcpy`/`<<<>>>`) — no cooperative groups, no
`cuda::pipeline`, no warp specialization, no `__shfl_sync`. All semantics live
in `SKGPU_HD` inline helpers so the same code compiles for g++/clang++ host
translation units (see `tests/host_math_check.cpp`).

## Layout

```
include/sketch/gpu/common.cuh    macros, portable clz64, launch/error helpers, counter_add_wrap
include/sketch/gpu/hash.cuh      SplitMix64Hash device path (== hash.h SplitMix64Hash)
include/sketch/gpu/hll.cuh       HLL register updates + merges
include/sketch/gpu/bloom.cuh     bf bit-set, cbf counter-add
include/sketch/gpu/count.cuh     ccm counter-add
include/sketch/gpu/minhash.cuh   bbmh/fmh independent-sketch batch updates
sketch.cu                        thin __global__ definitions (grid-stride loops only)
tests/test_gpu_sketches.cu       device bit-identity tests (nvcc)
tests/host_math_check.cpp        same contract, host-only serial folds (plain g++)
DESIGN.md                        phase-2 plan (cqf/bcqf/cht on GPU)
Makefile                         nvcc targets; `make test`, `make hostcheck`
```

## The bit-identity contract

Given the same **pre-hashed** uint64 stream, GPU sketch state is
bit-identical to the CPU implementation:

| GPU entry | CPU reference (include/sketch/*.h) | relied-upon semantics |
|---|---|---|
| `skgpu::splitmix64_hash` | `hash.h` `SplitMix64Hash` | xorshift-multiply finalizer, mod-2^64 multiplies |
| `skgpu::hll_update[_hashed]` | `hll.h` `hllbase_t::add` | `index = hv >> (64-np)`; `lzt = clz(((hv<<1)|1) << (np-1)) + 1`; byte max |
| `skgpu::hll_merge` | `hll.h` `operator+=` | per-register byte-wise max = union |
| `skgpu::bf_update` | `bf.h` `addh`/`sub_set1` | `npw = 64/p` subhashes, `set1(hv >> i*p) & mask`, incl. partial last group |
| `skgpu::cbf_add`, `skgpu::ccm_add` | `ccm.h` `ccmbase_t` (conservative_update=false) | `index = (hv & mask) + (i << l2sz)`; counts wrap mod 2^nbits |
| `skgpu::bbmh_batch_update` | `bbmh.h` `BBitMinHasher::add` | `bucket = hv >> (64-p)`, `val = (hv<<p)>>p`, register min |
| `skgpu::fmh_batch_filter` | `fracminhash.h` `fracminhashbase_t::add` | keep `hv <= max_hash && hv != ~0`; kept count + running min |

All updates are order-insensitive folds (max/min/add/or), so atomic execution
order does not affect the result. Cardinality estimation stays on CPU for
now: copy merged registers back (e.g. into `hllbase_t::mutable_core`) and
reuse the existing estimators unchanged.

### Notes / caveats

- HLL registers are updated with a word-level CAS loop (CUDA has no 8-bit
  `atomicMax`). Registers only grow and are bounded, so the loop terminates;
  the result is exactly the per-byte max. For heavy contention the planned
  optimization is shared-memory privatization per block (DESIGN.md).
- `cbf_add`/`ccm_add` take an `nbits` parameter because the CPU compact
  vector masks on store (counts wrap mod 2^nbits). `nbits=32` (default) is a
  plain `atomicAdd`. Also note the CPU non-conservative path passes
  `maxv = 1 << nbits` to `range_check` where a bit width is expected, so for
  `nbits >= 6` every update is silently suppressed on x86 (UB shift). The
  device tests use `nbits <= 5` for the CPU reference accordingly.
- The conservative ccm update (increment only tied-minimum counters) does not
  commute across elements and is deferred (DESIGN.md).
- bf kernels consume one 64-bit value per (element, seed *group*), matching
  `bf_ngroups(nh, 64/p)` values per element — the same count the CPU
  computes per element.

## Building

Requires nvcc (CUDA >= 9 for sm_70; any recent toolkit) and zlib (`-lz`) for
the CPU reference headers. On a GPU host:

```bash
cd gpu
make                    # builds tests/test_gpu_sketches (ARCH default: sm_70)
make ARCH="-gencode arch=compute_86,code=sm_86"   # e.g. RTX 3090
make test               # build + run
make hostcheck          # no-CUDA scalar-semantics self-check (plain g++)
```

Expected test runtime: well under 5 minutes on a midrange card (dominated by
the ~1M-element HLL streams). `tests/test_gpu_sketches.cu` prints one line
per check and exits nonzero on any failure.

## Running on vast.ai

Cheap way to get a GPU box for a first compile/test (no API keys here —
rent via the vast.ai web UI):

1. Rent an instance with an RTX 3090 / RTX 4090 / A5000 (any sm >= 70 card)
   using one of the standard CUDA images (e.g. `nvidia/cuda:12.4-devel-ubuntu22.04`).
2. `git clone <this repo> && cd sketch/gpu`
3. `apt-get update && apt-get install -y zlib1g-dev g++ make` (CUDA images
   ship nvcc; pick `ARCH` for the card, e.g. `-gencode arch=compute_86,code=sm_86` for a 3090).
4. `make test` — expect all checks to pass; runtime < 5 min.
5. `make hostcheck` — optional cross-check of the scalar contract.
6. Teardown: destroy the instance. Typical cost for this session: <$1
   (instance time plus a few GB of image pull).

For AMD: `hipify-cl -inplace include/sketch/gpu sketch.cu tests/test_gpu_sketches.cu`,
then compile with `hipcc` instead of `nvcc` (set `NVCC=hipcc`, adjust
`ARCH` to the ROCm target, e.g. `--offload-arch=gfx90a`); ZLUDA can run the
CUDA binaries unchanged on AMD with the CUDA-on-ROCm translation layer.

## Porting status

Phase 1 (this directory): hash, HLL update/merge, bf/cbf/ccm updates,
bbmh/fmh batch updates, correctness tests. Phase 2 (cqf/bcqf/cht) is designed
in `DESIGN.md` but not implemented.
