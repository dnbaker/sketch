# Changelog

## Unreleased

- Removed the vendored `include/flat_hash_map` (no users remain in sketch, bonsai or dashing2;
  the headers moved to `ankerl::unordered_dense` in 0.20.0).

## 0.20.0 (2026-10-20)

### Breaking
- `add`/`addh` on HLL, b-bit MinHash, DivBBitMinHasher and HyperMinHash are plain stores
  (one thread per sketch). Use `add_threadsafe`/`addh_threadsafe` to update one sketch from
  several threads (plain-load check, then hardware atomic max/min or CAS). `NOT_THREADSAFE`
  no longer affects these classes. shll and CountingBBitMinHasher delete the threadsafe variants.
- sketch headers no longer include `ska::flat_hash_map` (now `ankerl::unordered_dense` v5.3.2,
  C++17); `sketch/hedley.h` removed in favour of SIMDe's.

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
