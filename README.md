# hll
C++ HyperLogLog Implementation with SIMD Parallelism

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
