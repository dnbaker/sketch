# Phase 2 design: cqf / bcqf / cht on GPU

Status: design only — no code yet. This is the concrete implementation plan
for a later session. Everything targets the same portable subset and the same
bit-identity contract as phase 1 (see README.md).

## 1. The hard part: cqf insertion is a run shift, not an atomic RMW

`cqfbase_t` (include/sketch/cqf.h) is home-anchored SoA: `metaraw_` = occ
words (1 bit/home) + runend words (1 bit/physical slot), `hveraw_` = holes
bitvector (`hv[i] = !slot_live(i)`), payload either fused `slot_` registers
or packed `remraw_`/`cntraw_`. An insert at home h with an occupied run finds
the insertion offset via rank/select over occ/runend, **shifts the suffix of
the run one slot right** (payload + runend bits), then writes remainder/count
at the offset. A GPU insert is therefore a serialized, order-dependent
mutation of a shared window — the phase-1 trick (order-insensitive atomic
folds) does not apply. Two consequences drive the whole design:

1. **Deterministic bit-identity vs CPU is only well-defined per serialized
   insert stream.** Like the CPU structure under `add_mt`, concurrent inserts
   must be made order-equivalent: aggregate all inserts of a batch, sort by
   home (plus a tie-break = stream index), and apply them home-group by
   home-group. Within one home, apply in stream order — that matches what a
   single-threaded CPU replay of the same batch produces.
2. **The shift is the throughput bottleneck.** A naive one-thread-per-insert
   shift serializes on the run; the plan below parallelizes shifts across
   homes and batches them within a home.

## 2. Proposed kernel pipeline (insert batch)

Stage A — **hash + partition** (embarrassingly parallel, grid-stride):
`hv = hf(x)`; `home = (hv >> r) & (nslots-1)`; remainder = `hv & ((1<<r)-1)`
(exactly `cqfbase_t::part`). Emit `(home, seq, remainder)` tuples.

Stage B — **per-home aggregation** (device radix sort by home, e.g. a
single-pass LSD radix over 32-bit homes into a ping-pong buffer, or
`cub::DeviceRadixSort` if we accept the CUB dependency — decide then; a
hand-rolled 256-bin multi-pass radix is ~100 lines and keeps the dependency
surface zero). Ties broken by `seq` so stream order is preserved.

Stage C — **run probing, warp-aggregated**: one warp per home group (groups
span a contiguous sorted segment; one warp handles consecutive groups via
`__ballot_sync` segmentation — each thread owns one tuple, the ballot over
"same home as lane 0" delineates the group). The warp elects one lane to
locate the home's run via the directory-assisted rank/select path
(`occ_dir_`/`runend_dir_` — the select returns the k-th set bit; on GPU the
directory must be precomputed on host or by a prior kernel and shipped, since
building it is a serial scan). Lane 0 computes `offset_in_run` = upper_bound
of the remainder within the run's remainders (warp-cooperative compare over
the run's remainders, lanes striding the run). Result: per tuple, its final
absolute slot index — but see Stage D, because the shift invalidates
precomputed absolute indices.

Stage D — **shift application, home-parallel**: process sorted homes in
order; for each home, its run's suffix shift is independent of every other
home *unless runs overlap the same cache lines/words* — physically they can
abut (linear overflow tail). Simple correct v1: one thread block per home
group applies that home's inserts sequentially-in-stream-order (block-wide
memmove of payload words + runend bits, then writes). The block loads the
run's touched words into shared memory, shifts in shared, writes back — this
converts global RMW traffic into one read + one write per word. Concurrent
blocks touching adjacent runs at region boundaries need a **word-ownership
discipline**: partition the physical slot space into disjoint word-aligned
spans in Stage B' (a tiny second pass assigns each home group the union of
word spans its shift touches, merging groups whose spans intersect), so no
two blocks ever write the same word. This merging is exactly the
"displacement/decode metadata" bookkeeping: keep it as a small auxiliary
array `(group_id, span_begin_word, span_end_word)` sorted by span.

Stage E — **metadata maintenance**: occ bit set at first-distinct home
(atomicOr, idempotent), runend bits rewritten by the shifting block as a
byproduct of the payload shift (runend bits move with their slots — the CPU
invariant), `used_`/`ndistinct_` accumulated per block via atomicAdd of
per-block counts, holes vector: `hv[i]` cleared for newly occupied slots.
Directory rebuild (occ_dir_/runend_dir_/holes_dir_ `detail::dir`) after the
batch: the `dir` is word-summed popcount layers — a prefix-sum kernel over
occ words produces it directly; keep the flat-scan fallback on CPU for
`structure_check`.

## 3. Occupancy-bitmap scans → `__ballot_sync`

`find_slot`/occupancy probing on CPU scans occ words with popcount
("is home h occupied", "k-th occupied home"). On GPU these become:
- per-warp occ word scan: lane i holds occ word i; `__ballot_sync(mask,
  occ_word != 0)` gives the nonempty-word bitmap directly (this is the CPU
  "first_nonempty" inner loop vectorized across lanes);
- rank within a word: `__popcll` per lane + warp inclusive scan
  (`__reduce_add_sync` where available, else shfl-based — gated on arch,
  with the serial fallback loop for sm_70);
- select (k-th set bit): warp-parallel check `rank_before_word + popc(word)
  > k`, ballot the hit lane, then `__ffsll` in that lane. No
  `__shfl_sync`-heavy emulation of per-bit loops; the ballot IS the scan.
Count-within-run (`count_many` on CPU) maps to the same primitives over the
run's runend words: popcount between run boundaries, lanes parallel over
words.

## 4. Atomics vs per-warp buffers

- **Bit-identity-critical metadata** (occ, runend, holes, payload words):
  never raw atomicOr/atomicExch from arbitrary threads during Stage D —
  block-owned spans only, exactly as above. `atomicAdd` is reserved for pure
  accumulators (`used_`, histograms, ticket counters), where commutativity
  holds.
- **Per-warp staging buffers**: for count/query workloads (no mutation), a
  warp-cooperative probe needs no atomics at all. For insert batches,
  Stage C's elected lane may stage `(home, offset)` pairs in shared memory;
  when shared capacity is exceeded, spill per-block to a global scratch
  region partitioned by block index (no cross-block sharing).
- The holes/Fenwick `first_empty` on the linear overflow tail: v1 recomputes
  per batch on CPU from the final `hv` vector (one pass, it is only needed
  when the tail is actually in play — load factor > ~0.9); a GPU select1
  kernel is the follow-up.

## 5. bcqf and cht specifics

- `bcqf` (block-encoded counting quotient filter, bcqf_frozen.h/bcqf_trie.h):
  the block metadata (select/summary words per 64-slot block) is exactly the
  bitmap-scan material of §3 — same warp primitives, one level down. Frozen
  (read-only) bcqf is the easy first target: pure query kernels (membership,
  count) with zero mutation concerns; insert-on-GPU defers to the cqf
  pipeline plus block-metadata re-derivation per touched block.
- `cht` (cht.h): the decode path (`cht_*_hash` chains, feistel) is
  device-trivial (integer ops); the table probe is open-addressing with
  triangular displacement — warp-aggregated probing: one warp per query,
  lanes probe consecutive displacements, `__ballot_sync(full)` detects the
  cluster-full condition that triggers the grow; inserts use per-warp
  buffers aggregated by bucket and atomicCAS on (key,value) with versioned
  sentinel for the displacement metadata. Left/right-triangular hash choice
  from hash.h (`is_left_triangular` etc.) matters for reduced-only decode —
  the device must use the same reduced-key decode the CPU table was built
  with (full-width decode needs the stored 64-bit value; document per-table).

## 6. Testing plan (when implemented)

Extend `tests/test_gpu_sketches.cu` with the same contract: build the CPU
cqf/bcqf/cht, replay the identical insert stream through the GPU batch
pipeline (serialized per §1), and require byte-identical SoA state
(`metaraw_`, payload arrays, `hveraw_`) — plus a query-fuzz pass comparing
membership/count answers. The `nbits`-wrap lesson from phase 1 applies: any
count field must wrap exactly like the CPU storage.
