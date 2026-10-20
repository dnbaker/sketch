// BBitMinHasher::finalize must record the live cardinality estimate. Before
// the fix, each empty register counted as 2^p in the finalized estimate, so
// sparse sketches (n below ~ m ln m) were badly underestimated after finalize
// / write (e.g. 323 for 6000 items at p = 10); and the live harmonic estimate
// itself was biased high while registers were empty (1000 items at p = 10
// read 1610) -- fixed by linear counting in that range.
#include "sketch/bbmh.h"
#include <cmath>
#include <cstdio>

static uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main() {
    int fails = 0;
    for(const unsigned p: {10u, 12u})
        for(const uint64_t n: {1000ull, 3000ull, 6000ull, 20000ull, 200000ull}) {
            sketch::BBitMinHasher<uint64_t> m(p);
            for(uint64_t i = 0; i < n; ++i) m.add(mix(i + 12345));
            const double live = m.cardinality_estimate();
            for(const unsigned b: {8u, 16u, 32u}) {
                const double fin = m.finalize(b).cardinality_estimate();
                const bool ok = std::abs(fin - live) <= 1e-9 * live && std::abs(live - double(n)) < 0.1 * double(n);
                if(!ok) {
                    std::fprintf(stderr, "FAIL p=%u n=%llu b=%u: live %.1f finalized %.1f\n", p, (unsigned long long)n, b, live, fin);
                    ++fails;
                }
            }
        }
    if(fails) return 1;
    std::fprintf(stderr, "bbmh finalize cardinality: all passed\n");
    return 0;
}
