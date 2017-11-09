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

To link against the library, add `$PATH/$TO/hll` to your LD_LIBRARY_PATH after building and pass -lhll during compilation. Alternatively, you can simply `#include hll/hll.h` and use the hll.o object file directly in your project.


### `__builtin_clzll` undefined on 0
The behavior of `__builtin_clzll` is undefined on both gcc and clang for an operand of 0.
On most modern hardware, this returns the number of bits in the underlying type (64).
If you wish to avoid this edge case out of principle, compile with `-DAVOID_CLZ_UNDEF`. Note that there is precisely one 64-bit integer [0x7ffffbffffdfffff] which hashes to 0, so that even if it doesn't perform as expected, it would have an extremely minimal effect in practice.

### Multithreading
By default, updates to the data structure to occur using atomic operations, though threading should be handled by the calling code. Otherwise, the flag -DNOT_THREADSAFE should be passed. The cost of this is relatively minor, but in single-threaded situations, this could be preferred.

Additionally, if the sum is desired to be parallelized, the parsum function can use any number of threads to speed up calculation of the final quantity. This uses kthread from [klib](https://github.com/AttractiveChaos/klib), which significantly outperforms openmp in experiments.
