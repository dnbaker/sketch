# sketch [![Build Status](https://travis-ci.com/dnbaker/sketch.svg?branch=master)](https://travis-ci.com/dnbaker/sketch)
sketch is a generic, head-only library wit a variety of sketch data structures for scalable and streaming applications.
All have been accelerated with SIMD parallelism where possible, most are composable, and many are threadsafe unless `-DNOT_THREADSAFE` is passed as a compilation flag.

## Contents
1. HyperLogLog Implementation [hll.h]
    1. `hll_t`/`hllbase_t<HashStruct>`
    2. Estimates the cardinality of a set using log(log(cardinality)) bits.
    3. Threadsafe unless `-DNOT_THREADSAFE` is passed.
    4. Currently, `hll` is the only structure for which python bindings are available, but we intend to extend this in the future.
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
    2. CountingRangeMinHash performs the same operations as RangeMinHash, but provides multiplicities, which facilitates `histogram_similarity`, a generalization of Jaccard with multiplicities.
    3. Both CountingRangeMinHash and RangeMinHash can be finalized into containers for fast comparisons with `.finalize()`.
    3. A draft HyperMinHash implementation is available as well, but it has not been thoroughly vetted.
    4. Range MinHash implementations and the HyperMinHash implementation are *not* threadsafe.
6. B-Bit MinHash
    1. bbmh.h
    2. One-permutation (partition) bbit minhash
        1. Threadsafe, bit-packed and fully SIMD-accelerated
        2. Power of two partitions are supported in BBitMinHasher, which is finalized into a FinalBBitMinHash sketch. This is faster than the alternative.
        3. We also support arbitrary divisions using fastmod64 with DivBBitMinHasher and its corresponding final sketch, FinalDivBBitMinHash.
    3. One-permutation counting bbit minhash
        1. In progress
        2. Not threadsafe.
7. ntcard
    1. mult.h
    2. Threadsafe
    3. Reference: https://www.ncbi.nlm.nih.gov/pubmed/28453674
    4. Not SIMD-accelerated, but also general, supporting any arbitrary coverage level

The following sketches are experimental or variations on prior structures
1. HyperLogFilter [hll.h]
    1. `hlf_t`/`hlfbase_t<HashStruct>`, `chlf_t`/`chlfbase_t<HashStruct>`
    2. New data structure which provides the same quantitative accuracy as a HyperLogLog while providing more effective approximate membership query functionality than the HyperLogLog.
    2. `chlf_t` is identical to the `hlf_t` structure, with the exception that the memory is contiguous and each sketch cannot be used individually.
    3. Threadsafe unless `-DNOT_THREADSAFE` is passed.
2. filterhll [filterhll.h]
    1. `fhll_t`/`fhllbase_t<HashStruct>`
    2. Simple hll/bf combination without rigorous guarantees for requiring an element be present in the bloom filter to be inserted into the HyperLogLog.
    3. Currently *not* threadsafe.
3. Naive Approximate Counting Bloom Filter [cbf.h]
    1. `cbf_t`/`cbfbase_t<HashStruct>`
    2. An array of bloom filters where presence in a sketch at a given index replaces the count for the approximate counting algorithm.
    3. Currently *not* threadsafe.
7. Probabilistic Counting Bloom Filter
    1. `pcbf_t`/`pcbfbase_t<HashStruct>`
    2. An array each of bloom filters and hyperloglogs for approximate counting. The hyperloglogs provide estimated cardinalities for inserted elements, which allows us to estimate the error rates of the bloom filters and therefore account for them in count estimation The hyperloglogs provide estimated cardinalities for inserted elements, which allows us to estimate the error rates of the bloom filters and therefore account for them in count estimation.
    3. Currently *not* threadsafe.

Future work
1. Multiplicities
    1. Consistent Weighted Sampling, Improved CWS
    2. BagMinHash for efficient weighted Jaccard
    3. Multiplicative extensions of HLL
2. Sampling algorithms and core-sets

### Test case
To build and run the hll test case:

```bash
make test && ./test
```

To use:

```c++
using namespace sketch;
hll::hll_t hll(20); // Use 2**20 bytes for this structure
// Add hashed values for each element to the structure.
for(uint64_t i(0); i < 10000000ull; ++i) hll.addh(i);
fprintf(stderr, "Elements estimated: %lf. Error bounds: %lf.\n", hll.report(), hll.est_err());
```

The other structures work with a similar interface. See the type constructors for more information or view [10xdash](https://github.com/dnbaker/10xdash) for examples on using the
same interface for a variety of data structures.

Simply `#include sketch/<header_name>`.

### Multithreading
By default, updates to the hyperloglog structure to occur using atomic operations, though threading should be handled by the calling code. Otherwise, the flag `-DNOT_THREADSAFE` should be passed. The cost of this is relatively minor, but in single-threaded situations, this would be preferred.

## Python bindings
Python bindings are available via pybind11 and then imported through hll.py. hll.py calls an object's __hash__ function. To link against python2, change the "python3-" in the Makefile to "python-".
