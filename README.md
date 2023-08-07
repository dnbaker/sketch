# sketch [![Build Status](https://travis-ci.com/dnbaker/sketch.svg?branch=master)](https://travis-ci.com/dnbaker/sketch) [![Documentation Status](https://readthedocs.org/projects/sketch/badge/?version=latest)](https://sketch.readthedocs.io/en/latest/?badge=latest)
sketch is a generic, header-only library providing implementations of a variety of sketch data structures for scalable and streaming applications.
All have been accelerated with SIMD parallelism where possible, most are composable, and many are threadsafe unless `-DNOT_THREADSAFE` is passed as a compilation flag.


## Python documentation

Documentation for the Python interface is available [here](https://sketch.readthedocs.io/en/latest/).

## Dependencies

We directly include blaze-lib, libpopcnt, [compact_vector](https://github.com/gmarcais/compact_vector), ska::flat\_hash\_map, and xxHash for various utilities.
We also have two submodules:

* pybind11, only used for python bindings.
* SLEEF for vectorized math, incorporated with vec.h. It's optionally used (disabled by defining `-DNO_SLEEF=1/#define NO_SLEEF 1`) and only applicable to using rnla.h through blaze-lib.

You can ignore both for most use cases.

## Contents
1. HyperLogLog Implementation [hll.h]
    1. `hll_t`/`hllbase_t<HashStruct>`
    2. Estimates the cardinality of a set using log(log(cardinality)) bits.
    3. Threadsafe unless `-DNOT_THREADSAFE` is passed.
    4. Currently, `hll` is the only structure for which python bindings are available, but we intend to extend this in the future.
2. HyperBitBit [hbb.h]
    1. Better per-bit accuracy than HyperLogLogs, but, at least currently, limited to 128 bits/16 bytes in sketch size.
3. Bloom Filter [bf.h]
    1. `bf_t`/`bfbase_t<HashStruct>`
    2. Naive bloom filter
    3. Currently *not* threadsafe.
4. Count-Min and Count Sketches
    1. ccm.h (`ccmbase_t<UpdatePolicy=Increment>/ccm_t`  (use `pccm_t` for Approximate Counting or `cs_t` for a count sketch).
    2. The Count sketch is threadsafe if `-DNOT_THREADSAFE` is not passed or if an atomic container is used. Count-Min sketches are currently not threadsafe due to the use of minimal updates.
    3. Count-min sketches can support concept drift if `realccm_t` from mult.h is used.
5. MinHash sketches
    1. mh.h (`RangeMinHash` is the currently verified implementation.) We recommend you build the sketch and then convert to a linear container (e.g., a `std::vector`) using `to_container<ContainerType>()` or `.finalize()` for faster comparisons.
        1. BottomKHasher is an alternate that uses more space to reduce runtime, which finalizes() into the same structure.
    2. CountingRangeMinHash performs the same operations as RangeMinHash, but provides multiplicities, which facilitates `histogram_similarity`, a generalization of Jaccard with multiplicities.
    3. Both CountingRangeMinHash and RangeMinHash can be finalized into containers for fast comparisons with `.finalize()`.
    3. A draft HyperMinHash implementation is available as well, but it has not been thoroughly vetted.
    4. Range MinHash implementationsare *not* threadsafe.
    5. HyperMinHash implementation is threa
6. B-Bit MinHash
    1. bbmh.h
    2. One-permutation (partition) bbit minhash
        1. Threadsafe, bit-packed and fully SIMD-accelerated
        2. Power of two partitions are supported in BBitMinHasher, which is finalized into a FinalBBitMinHash sketch. This is faster than the alternative.
        3. We also support arbitrary divisions using fastmod64 with DivBBitMinHasher and its corresponding final sketch, FinalDivBBitMinHash.
    3. One-permutation counting bbit minhash
        1. Not threadsafe.
7. ModHash sketches
    1. mod.h
    2. Estimates both containment and jaccard index, but takes a (potentially) unbounded space.
    3. This returns a FinalRMinHash sketch, reusing the infrastructure for minhash sketches,
       but which calculates Jaccard index and containment knowing that it was generated via mod, not min.
8. HeavyKeeper
    1. hk.h
    3. Reference: https://www.usenix.org/conference/atc18/presentation/gong
    4. A seemingly unilateral improvement over count-min sketches.
        1. One drawback is the inability to delete items, which makes it unsuitable for sliding windows.
        2. It shares this characteristic with the Count-Min sketch with conservative update and the Count-Min Mean sketch.
9. ntcard
    1. mult.h
    2. Threadsafe
    3. Reference: https://www.ncbi.nlm.nih.gov/pubmed/28453674
    4. Not SIMD-accelerated, but also general, supporting any arbitrary coverage level
10. PCSA
    1. pc.h
    2. The HLL is more performant and better-optimized, but this is included for completeness.
    3. Not threadsafe.
    1. Reference: https://dl.acm.org/doi/10.1016/0022-0000%2885%2990041-8
       Journal of Computer and System Sciences.
       September 1985 https://doi.org/10.1016/0022-0000(85)90041-8
11. SetSketch
    1. See setsketch.h for continuous and discretized versions of the SetSketch.
    2. This also includes parameter-setting code.

### Test case
To build and run the hll test case:

```bash
make test && ./test
```

### Example
To use as a header-only library:

```c++
using namespace sketch;
hll::hll_t hll(14); // Use 2**14 bytes for this structure
// Add hashed values for each element to the structure.
for(uint64_t i(0); i < 10000000ull; ++i) hll.addh(i);
fprintf(stderr, "Elements estimated: %lf. Error bounds: %lf.\n", hll.report(), hll.est_err());
```


The other structures work with a similar interface. See the type constructors for more information or view [10xdash](https://github.com/dnbaker/10xdash) for examples on using the
same interface for a variety of data structures.

Simply `#include sketch/<header_name>`, or, for one include `#include <sketch/sketch.h>`,
which allows you to write `sketch::bf_t` and `sketch::hll_t` without the subnamespaces.

We use inline namespaces for individual types of sketches, e.g., `sketch::minhash` or `sketch::hll` can be used for clarity, or `sketch::hll_t` can be used, omitting the `hll` namespace.

### OSX Installation
Clang on OSX may fail to compile in AVX512 mode. We recommend using homebrew's gcc:

```
homebrew upgrade gcc || homebrew install gcc
```
and either setting the environmental variables for CXX and CC to the most recent g++/gcc or providing them as Makefile arguments.
At the time of writing, this is `g++-10` and `gcc-10`.

### Multithreading
By default, updates to the hyperloglog structure to occur using atomic operations, though threading should be handled by the calling code. Otherwise, the flag `-DNOT_THREADSAFE` should be passed. The cost of this is relatively minor, but in single-threaded situations, this would be preferred.

## Python bindings
Python bindings are available via pybind11. Simply `cd python && python setup.py install`.

The package has been renamed to `sketch_ds` as of v0.19

Utilities include:
    1. Sketching/serialization for sketch data structures
        1. Supported: sketch_ds.bbmh.BBitMinHasher, sketch_ds.bf.bf, sketch_ds.hmh.hmh, sketch_ds.hll.hll
    2. shs\_isz, which computes the intersection size of sorted hash sets.
        1. Supported: {uint,int}{32,64}, float32, float64
    3. fastmod/fastdiv, which uses the fast modulo reduction to do faster division/mod than numpy.
        1. Supportd: {uint,int}{32,64}
    4. matrix generation functions - taking a list of sketches and creating the similarity function matrix.
        1. Supported: sketch_ds.bbmh.BBitMinHasher, sketch_ds.bf.bf, sketch_ds.hmh.hmh, sketch_ds.hll.hll
        2. Types: "jaccard_matrix", "intersection_matrix", "containment_matrix", "union_size_matrix", "symmetric_containment_matrix"
        3. Returns a compressed distance matrix.
    5. ccount\_eq, pcount\_eq compute the number of identical registers between integral registers.
        1. Inspired by cdist and pdist from scipy.spatial.distance
        2. ccount\_eq computes the number of identical registers between all pairs of rows between two matrices A and B.
            1. Size of returned matrix: (A.shape[0], A.shape[1])
        3. pcount\_eq computes the number of identical registers between all pairs of rows in a single matrix A.
            1. Size of returned matrix: (A.shape[0] * (A.shape[0]) - 1) / 2
        4. pcount\_eq output can be transformed from similarities to distances via -np.log(distmat / A.shape[1]).
