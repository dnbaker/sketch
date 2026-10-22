// BBitMinHasher::intersection_size must exclude registers empty in both
// sketches, exactly as jaccard_index does. Before the fix, both-empty
// registers counted as matches, so two disjoint 10k-key sketches at p = 13
// read an intersection of ~1847 instead of 0.
#include "sketch/bbmh.h"
#include <cmath>
#include <cstdio>

static uint64_t mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main() {
    const unsigned p = 13;
    const uint64_t n = 10000, noverlap = 8000;
    sketch::BBitMinHasher<uint64_t> d1(p), d2(p), s1(p), s2(p);
    for(uint64_t i = 0; i < n; ++i) {
        d1.add(mix(i));             // [0, n)
        d2.add(mix(i + n));         // [n, 2n): disjoint from d1
        s1.add(mix(i));             // [0, n)
        s2.add(mix(i + n - noverlap)); // [n - noverlap, 2n - noverlap): noverlap shared
    }
    int fails = 0;
    const double disjoint_is = d1.intersection_size(d2);
    const double disjoint_ji = d1.jaccard_index(d2);
    if(!(disjoint_is < 50.)) {
        std::fprintf(stderr, "FAIL disjoint p=%u n=%llu: intersection_size %.1f (jaccard_index %.4f)\n",
                     p, (unsigned long long)n, disjoint_is, disjoint_ji);
        ++fails;
    }
    const double shared_is = s1.intersection_size(s2);
    if(std::abs(shared_is - double(noverlap)) > 0.15 * double(noverlap)) {
        std::fprintf(stderr, "FAIL shared p=%u n=%llu overlap=%llu: intersection_size %.1f\n",
                     p, (unsigned long long)n, (unsigned long long)noverlap, shared_is);
        ++fails;
    }
    std::fprintf(stderr, "bbmh intersection: disjoint %.1f (ji %.4f), shared %.1f (true %llu)\n",
                 disjoint_is, disjoint_ji, shared_is, (unsigned long long)noverlap);
    if(fails) return 1;
    std::fprintf(stderr, "bbmh intersection: all passed\n");
    return 0;
}
