.. Sketch documentation master file

Welcome to the documentation for Sketch!
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Getting Started
============

Sketch has a range of sketch data structures implemented. All of them are available in C++, but only a subset of functionality has been exposed to Python.
This documentation is primarily for the Python interface.

Installation (Python)
---------

Build and install from source (PEP 517)::

    git clone --recursive https://github.com/dnbaker/sketch && cd sketch/python
    python3 -m pip install .

Or build a wheel without installing::

    python3 -m pip wheel --no-deps -w dist .

At this point, import the modules from python::

    import sketch_hll, sketch_bbmh, sketch_util, sketch_bf, sketch_hmh
    import sketch_ss, sketch_lsh
    import sketch_ull, sketch_ell, sketch_fmh, sketch_dmh, sketch_pmh

(The ``sketch_ds`` package re-exports these as ``hll``, ``bbmh``, ...,
``pmh``.)

Note on macOS: if your environment's ``CXX`` points at a broken/missing-
headers clang (e.g. Homebrew LLVM without a sysroot), the build
automatically substitutes ``/usr/local/bin/g++-16`` or ``g++-15`` when
present.

Python modules
---------

Cardinality sketches:

- ``sketch_hll`` -- HyperLogLog with Ertl ML / joint-MLE estimation
  (``addh``/``add``, union, jaccard, gz serialization).
- ``sketch_ull`` -- UltraLogLog (Ertl 2024): 8-bit register-compressed HLL
  successor; MLE and optimal-FGRA estimators, martingale (HIP) estimation,
  cross-precision merge (numpy batch add supported).
- ``sketch_ell`` -- ExaLogLog (Ertl 2025): bit-packed registers
  (default t=2, d=20), bias-reduced ML, cross-precision/d merge.
- ``sketch_fmh`` -- FracMinHash (sourmash-style fractional sampling):
  exact-over-samples containment, Poisson cardinality error.
- ``sketch_dmh`` -- DartMinHash (Christiani 2020): unbiased Jaccard from
  k independent buckets, k*J^ ~ Binomial(k, J).

Set similarity:

- ``sketch_bbmh`` -- b-bit minhash (``BBitMinHasher``/``FinalBBitMinHash``),
  SuperMinHash, one-permutation hashing.

One-permutation densification (b-bit minhash)
---------

``BBitMinHasher`` sketches can be *densified* after accumulation: every
register still holding the empty sentinel copies the value of the first
non-empty register in its own probe sequence (Shrivastava--Li densification),
leaving a full-length sketch with no empty registers for downstream uses that
need dense feature vectors (taxonomic classification, grafting/k-mer naming,
feature hashing). Densification is explicit and per-sketch; nothing runs it
automatically::

    h = sketch_bbmh.BBitMinHasher(13, 16)
    for v in values:
        h.addh(v)
    h.densify()          # in place: -1 all-empty (left as-is), 0 full, 1 filled
    h2 = h.densified()   # or a densified copy
    f = h.finalize()     # workflow: add + densify + finalize

Notes:

- ``finalize()`` densifies an internal copy anyway, so the packed b-bit
  Jaccard codes are identical with or without an explicit ``densify()``;
  the call matters for consumers of the raw registers.
- Merge raw sketches with ``|``/``union`` *before* densifying: the raw
  per-register min is the exact union sketch. The min of two already-
  densified *sparse* sketches is full-length but estimates union Jaccard low
  (borrowed registers can sit below the union's own value; 0.18 vs 0.235 at
  30%-filled p=13 registers); the bias vanishes as registers fill.
- Accuracy (p=13, b=16, R=60 independent pairs): densified-Jaccard rmse
  0.24-0.54% for ~2-11k-item sets and 0.5-1.6% for ~250-1.4k-item sets; the
  raw both-empty-excluding estimator is 2-8x sharper at the small end.
  ``densify()`` costs ~2 ns per register at p<=16 (measured p=10..16).

- ``sketch_hmh`` -- HyperMinHash (Yu & Weber).
- ``sketch_ss`` -- SetSketch / CSetSketch family (Byte/Nibble/Short variants).
- ``sketch_pmh`` -- Permutation MinHash (PMH1/PMH2, Ertl): weighted minhash;
  exact unweighted Jaccard, approximate weighted.

Filters and counting:

- ``sketch_bf`` -- Bloom filters (membership, jaccard, union).
- ``sketch_util`` -- hashing/fast-div utilities and comparison helpers.
- ``sketch_lsh`` -- LSH tables.

Testing
---------

The C++ test suite (48 binaries: per-sketch accuracy/merge/serialization
tests, edge-case regression tests, sampler moment tests) runs via
``bash run_tests.sh`` from the repository root; the python unittest suite
is ``python pytest.py`` inside ``python/`` (requires the built modules).

Guarantees validated by the test suite
---------

- Accuracy vs theory at simulated cardinalities 1e2..1e7 (HLL/ULL/ELL/FMH/
  DMH/PMH) and Jaccard 0.1..0.9.
- Register-exact merge (union) and idempotency for every sketch family.
- Serialization round-trips (gzipped files and, for the newer modules,
  ``to_bytes``/``from_bytes``).
- Edge cases: empty/singleton sketches, precision-parameter bounds,
  incompatible-merge rejection, zero-input hashing, hash inversion.
