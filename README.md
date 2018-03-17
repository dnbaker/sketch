# hll
C++ HyperLogLog Implementation with SIMD Parallelism

### Header-only or using -lhll
For faster compile times for downstream projects, default use involves compiling hll.cpp and linking against that library or the object file.
Alternatively, you can define the macro HLL_HEADER_ONLY which will move the cpp code into the header for easier portability.

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

Simply `#include hll/hll.h`. Experimental features are included in the `hll::dev` namespace by default, but can be moved into the `hll` namespace with -DENABLE_HLL_DEVELOP

### Multithreading
By default, updates to the data structure to occur using atomic operations, though threading should be handled by the calling code. Otherwise, the flag -DNOT_THREADSAFE should be passed. The cost of this is relatively minor, but in single-threaded situations, this could be preferred.

## Python bindings
Python bindings are available via pybind11 and then imported through hll.py. hll.py calls an object's __hash__ function. To link against python2, change the "python3-" in the Makefile to "python-".
