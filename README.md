# dnb's Sketch Data Structures
This repository contains a set of sketch data structures from scalable, streaming applications.
All have been accelerated with SIMD parallelism, and HyperLogLogs are threadsafe using atomic operations.

## Contents
1. HyperLogLog Implementation [hll.h]
  0. `hll_t`/`hllbase_t<HashStruct>`
  1. Estimates the cardinality of a set using log(log(cardinality)) bits.
2. HyperLogFilter [hll.h]
  0. `hlf_t`/`hlfbase_t<HashStruct>`
  1. New data structure which provides the same quantitative accuracy as a HyperLogLog while providing more effective approximate membership query functionality than the HyperLogLog.
3. HyperLogFilter [hll.h]
  0. `chlf_t`/`chlfbase_t<HashStruct>`
  1. Identical to the `hlf_t` structure, with the exception that the memory is contiguous and each sketch cannot be used individually.
4. Bloom Filter [bf.h]
  0. `bf_t`/`bfbase_t<HashStruct>`
  1. Naive bloom filter
5. Filterhll [filterhll.h]
  0. `fhll_t`/`fhllbase_t<HashStruct>`
  1. Simple hll/bf combination without rigorous guarantees for requiring an element be present in the bloom filter to be inserted into the HyperLogLog.
6. Naive Approximate Counting Bloom Filter [cbf.h]
  0. `cbf_t`/`cbfbase_t<HashStruct>`
  1. An array of bloom filters where presence in a sketch at a given index replaces the count for the approximate counting algorithm.
7. Probabilistic Counting Bloom Filter
  0. `pcbf_t`/`pcbfbase_t<HashStruct>`
  1. An array each of bloom filters and hyperloglogs for approximate counting. The hyperloglogs provide estimated cardinalities for inserted elements, which allows us to estimate the error rates of the bloom filters and therefore account for them in count estimation The hyperloglogs provide estimated cardinalities for inserted elements, which allows us to estimate the error rates of the bloom filters and therefore account for them in count estimation.

### Test case
To build and run the test case:

```bash
make test && ./test
```

To use:

```c++
hll::hll_t hll(20); // Use 2**20 bytes for this structure
// Add hashed values for each element to the structure.
for(uint64_t i(0); i < 10000000ull; ++i) hll.addh(i);
fprintf(stderr, "Elements estimated: %lf. Error bounds: %lf.\n", hll.report(), hll.est_err());
```

The other structures work with a similar interface. See the type constructors for more information.

Simply `#include hll/<header_name>`.

### Multithreading
By default, updates to the hyperloglog structure to occur using atomic operations, though threading should be handled by the calling code. Otherwise, the flag -DNOT_THREADSAFE should be passed. The cost of this is relatively minor, but in single-threaded situations, this could be preferred.

## Python bindings
Python bindings are available via pybind11 and then imported through hll.py. hll.py calls an object's __hash__ function. To link against python2, change the "python3-" in the Makefile to "python-".
