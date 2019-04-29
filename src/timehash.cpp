#include "hash.h"
#include "common.h"
using namespace sketch;
using namespace hash;
#if 1
#define ACC accum ^=
#else
#define ACC
#endif
int main() {
    WangHash hash;
    uint64_t seed1 = hash(uint64_t(1337));
    hash::MultiplyAddXoRot<33> gen1(seed1, hash(seed1));
    hash::MultiplyAddXor gen2(seed1);
    hash::XorMultiply gen3(seed1, hash(seed1));
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t accum = 0;
    uint64_t nelem = 100000000;
    for(uint64_t i = 0; i < nelem; ++i) {
ACC hash(uint64_t(i));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "diff: %zu. accum %zu\n", size_t(std::chrono::nanoseconds(end - start).count()), size_t(accum));
#define DO_THING(hasher, namer) \
    start = std::chrono::high_resolution_clock::now();\
    accum = 0;\
    for(uint64_t i = 0; i < nelem; ++i) {\
       ACC hasher(i);\
    }\
    end = std::chrono::high_resolution_clock::now();\
    std::fprintf(stderr, "for %s diff: %zu. accum %zu\n", namer, size_t(std::chrono::nanoseconds(end - start).count()), size_t(accum));
    
    DO_THING(gen1, "multiplyaddxorot33")
    DO_THING(gen2, "mulyaddor")
    DO_THING(gen3, "xormult")
}
