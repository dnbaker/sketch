// atomic_store_min/max: concurrent updates leave every register at the true
// min/max (u8 and u64, the native path where the target has one, CAS otherwise),
// and HyperLogLog / b-bit MinHash / HyperMinHash (8-64-bit registers) sketches
// filled from 8 threads with add_threadsafe equal the sketches filled serially with add.
#include "sketch/hll.h"
#include "sketch/bbmh.h"
#include "sketch/hmh.h"
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
    DivBBitMinHasher<uint64_t> ds(1000, 16), dp(1000, 16);
    std::vector<HyperMinHasher<>> ms, mp;
    for(const unsigned rs: {8, 16, 32, 64}) ms.emplace_back(10, rs), mp.emplace_back(10, rs);
    for(unsigned i = 0; i < nt * n; ++i) {
        const uint64_t h = mix(i + 77);
        hs.add(h), bs.add(h), ds.add(h);
        for(auto &x: ms) x.add(h);
    }
    std::vector<std::thread> th;
    for(unsigned t = 0; t < nt; ++t) th.emplace_back([&, t] {
        for(unsigned i = 0; i < n; ++i) {
            const uint64_t h = mix(t * n + i); const unsigned r = h % m;
            atomic_store_min(&mn[r], h >> 1); atomic_store_max(&mx[r], h >> 1);
            atomic_store_min(&mn8[r], uint8_t(h >> 56)); atomic_store_max(&mx8[r], uint8_t(h >> 56));
            const uint64_t v = mix(t * n + i + 77);
            hp.add_threadsafe(v); bp.add_threadsafe(v); dp.add_threadsafe(v);
            for(auto &x: mp) x.add_threadsafe(v);
        }
    });
    for(auto &x: th) x.join();
    for(unsigned r = 0; r < m; ++r) {
        if(mn[r] != wmn[r] || mx[r] != wmx[r] || mn8[r] != wmn8[r] || mx8[r] != wmx8[r]) {std::fprintf(stderr, "register %u differs\n", r); return 1;}
    }
    if(hs.core() != hp.core()) {std::fprintf(stderr, "hll differs\n"); return 1;}
    if(!(bs == bp)) {std::fprintf(stderr, "bbmh differs\n"); return 1;}
    if(!(ds == dp)) {std::fprintf(stderr, "div bbmh differs\n"); return 1;}
    for(size_t i = 0; i < ms.size(); ++i)
        if(!(ms[i] == mp[i])) {std::fprintf(stderr, "hmh (register size %u) differs\n", 8u << i); return 1;}
    std::fprintf(stderr, "atomic_minmax: registers and sketches match serial (hll %.0f, bbmh %.0f, n %u)\n", hp.report(), bp.cardinality_estimate(), nt * n);
    return 0;
}
