// atomic_store_min/max: concurrent updates leave every register at the true
// min/max (u8 and u64, the native path where the target has one, CAS otherwise),
// and HyperLogLog / b-bit MinHash sketches filled from 8 threads equal the serial sketches.
#include "sketch/hll.h"
#include "sketch/bbmh.h"
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>
using namespace sketch;

static uint64_t mix(uint64_t x) {x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL; return x ^ (x >> 31);}

int main() {
#if defined(SKETCH_ATOMIC_MINMAX_NATIVE)
    std::fprintf(stderr, "atomic_minmax: native (LSE)\n");
#elif defined(SKETCH_ATOMIC_MINMAX_BUILTIN)
    std::fprintf(stderr, "atomic_minmax: __atomic_fetch_min/max\n");
#else
    std::fprintf(stderr, "atomic_minmax: CAS\n");
#endif
    const unsigned nt = 8, n = 200000, m = 64;
    std::vector<uint64_t> mn(m, uint64_t(-1)), mx(m, 0), wmn(m, uint64_t(-1)), wmx(m, 0);
    std::vector<uint8_t> mn8(m, 255), mx8(m, 0), wmn8(m, 255), wmx8(m, 0);
    for(unsigned t = 0; t < nt; ++t) for(unsigned i = 0; i < n; ++i) {
        const uint64_t h = mix(t * n + i); const unsigned r = h % m;
        wmn[r] = std::min(wmn[r], h >> 1); wmx[r] = std::max(wmx[r], h >> 1);
        wmn8[r] = std::min<uint8_t>(wmn8[r], uint8_t(h >> 56)); wmx8[r] = std::max<uint8_t>(wmx8[r], uint8_t(h >> 56));
    }
    hll_t hs(12), hp(12);
    BBitMinHasher<uint64_t> bs(10), bp(10);
    for(unsigned i = 0; i < nt * n; ++i) hs.add(mix(i + 77)), bs.add(mix(i + 77));
    std::vector<std::thread> th;
    for(unsigned t = 0; t < nt; ++t) th.emplace_back([&, t] {
        for(unsigned i = 0; i < n; ++i) {
            const uint64_t h = mix(t * n + i); const unsigned r = h % m;
            atomic_store_min(&mn[r], h >> 1); atomic_store_max(&mx[r], h >> 1);
            atomic_store_min(&mn8[r], uint8_t(h >> 56)); atomic_store_max(&mx8[r], uint8_t(h >> 56));
            hp.add(mix(t * n + i + 77)); bp.add(mix(t * n + i + 77));
        }
    });
    for(auto &x: th) x.join();
    for(unsigned r = 0; r < m; ++r) {
        if(mn[r] != wmn[r] || mx[r] != wmx[r] || mn8[r] != wmn8[r] || mx8[r] != wmx8[r]) {std::fprintf(stderr, "register %u differs\n", r); return 1;}
    }
    if(hs.core() != hp.core()) {std::fprintf(stderr, "hll differs\n"); return 1;}
    if(!(bs == bp)) {std::fprintf(stderr, "bbmh differs\n"); return 1;}
    std::fprintf(stderr, "atomic_minmax: registers and sketches match serial (hll %.0f, bbmh %.0f, n %u)\n", hp.report(), bp.cardinality_estimate(), nt * n);
    return 0;
}
