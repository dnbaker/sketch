# Changelog

## 0.20.3 (2026-10-22)

### Added
- cht ordered runs: every probe run kept sorted by home. Maintained by a
  local neighbour check on each insert, restored by `make_ordered()` (sorts
  each run in place, O(n): the occupied-slot set of linear probing does not
  depend on insertion order), and recomputed by `read()` (`scan_ordered()`).
  While ordered, get/count/find_slot stop at the first slot whose
  displacement is below the probe distance, so a miss costs about a hit.
  Measured on a 50M-entry 62-bit-key map (ns per miss, unordered ->
  ordered): load 0.80: ~65 -> ~60; 0.85: ~170 -> ~70; 0.90: ~330 -> ~84.
- Test: chttest test_ordered (hits/misses vs a reference map at loads
  0.5/0.8/0.95, write/read flag, inserts after make_ordered).

### Changed
- `cht_div_sizing` splits by fastmod (`policy::SizeDivPolicy::divmod`):
  home = h mod n, rem = h div n, compose = rem * n + home (replaces
  fastrange and its 128-bit decode division). The serialized layout id is
  now 1 = pow2, 2 = fastmod; fastrange-layout (id 0) maps are rejected on
  read. `add_bulk_sorted` inserts key by key under div sizing (fastmod homes
  are not monotone in the hash).

## 0.20.2

### Added
- `BBitMinHasher::densified()`: const convenience returning a densified copy.
  `densify()` documented: per-sketch, explicit (no auto-densify), call before
  `finalize()`; all-empty sketches are left as-is (rc -1). Python bindings:
  `BBitMinHasher.densify` / `densified` (sketch_bbmh).
- Test: bbmh_densifytest -- sentinel elimination, rc/idempotence, identical
  finalize() codes with/without explicit densify, disjoint/known-Jaccard
  accuracy vs the plain estimator at p=13/b=16, merge ordering, determinism,
  densify() cost at p=10..16.

### Measured (p=13, b=16, R=60 independent pairs)
- Densified-Jaccard rmse: 0.24-0.54% for ~2-11k-item sets; 0.5-1.6% for
  ~250-1.4k-item sets. The plain both-empty-excluding raw-register estimator
  is 2-8x sharper at the small end (it adaptively drops uninformative
  registers); densified rmse tracks sqrt(J(1-J)/m_eff) with m_eff ~ registers
  non-empty in either sketch.
- Merge order: raw per-register min is the exact union sketch, so merge raw
  then densify once. min() of two already-densified *sparse* sketches is
  full-length but biased low for union-vs-third-set Jaccard (0.18 vs 0.235 at
  30% fill); bias vanishes as registers fill, exact equality once no register
  is empty.
- densify(): ~2 ns/register at p=10..16 (includes sketch copy).

## 0.20.0 (2026-10-20)

### Breaking
- `add`/`addh` on HLL, b-bit MinHash, DivBBitMinHasher and HyperMinHash are plain stores
  (one thread per sketch). Use `add_threadsafe`/`addh_threadsafe` to update one sketch from
  several threads (plain-load check, then hardware atomic max/min or CAS). `NOT_THREADSAFE`
  no longer affects these classes. shll and CountingBBitMinHasher delete the threadsafe variants.
- sketch headers no longer include `ska::flat_hash_map` (now `ankerl::unordered_dense` v5.3.2,
  C++17); `sketch/hedley.h` removed in favour of SIMDe's.
- vendored `include/flat_hash_map` removed (no remaining users in sketch, bonsai or dashing2).

### Fixed (results change)
- b-bit MinHash densification borrowed from bins that were themselves densified; it now
  borrows only from originally filled bins. Finalized Jaccard rmse on sparse sketches roughly
  halves (p=12, J=0.5: 0.095 -> 0.049 at 30 items/set).
- `BBitMinHasher::jaccard_index` counted registers empty in both sketches as matches
  (0.5 read as 0.81 for 1000 keys at p=12).
- `CountingBBitMinHasher`: add kept the max and bumped the register on ties, addh skipped
  counting, multiset_comparison double-counted mismatched minima; histogram_intersection is
  now an unbiased weighted-Jaccard estimate.
- b-bit finalize recorded a cardinality that counted empty registers as 2^p (323 for 6000
  items at p=10); it now records the live estimate, with linear counting when sparse.
- cht exact mode merged run-mates with different homes; matches now require the displacement.
- setsketch erase-while-iterating loops skipped elements.

### Added
- `cht_t` / `cht_map_t`: compact, decodable hash table (exact and fingerprint modes, runtime
  key/value widths incl. 128-bit keys, non-power-of-two sizes, prefetching batch lookups,
  serialization).
- Hardware atomic min/max (`atomic_minmax.h`: AArch64 LSE, `__atomic_fetch_max/min`, CAS).
- Tests: xxhgoldentest, popcnttest, countingbbmhtest, densifytest, extended atomic_minmax_test.

### Dependencies
- xxHash 0.8.4 (digests unchanged, checked by xxhgoldentest), libpopcnt 4.2, pybind11 3.1.0,
  SIMDe as a submodule (simde-no-tests 0.8.2); unused sleef submodule removed.
  After pulling: `git submodule update --init --recursive`.
