// b-bit MinHash densification (detail::densifybin): every empty bin takes the
// value of the first *originally* filled bin in its own probe sequence, so the
// result does not depend on the order bins are processed in, filled bins are
// untouched, and finalized Jaccard estimates of sparse sketches keep their
// accuracy (the in-place version let bins copy borrowed values, which roughly
// doubled the rmse for sets much smaller than the sketch).
#include "sketch/bbmh.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace sketch;

int main() {
    const uint64_t E = minhash::detail::default_val<uint64_t>();
    // 1. order independence against a reverse-order reference
    for(unsigned nfill: {1u, 3u, 40u, 300u}) {
        std::vector<uint64_t> h(1024, E);
        uint64_t s = 0x9E3779B97F4A7C15ull + nfill;
        for(unsigned k = 0; k < nfill; ++k) {s ^= s << 13; s ^= s >> 7; s ^= s << 17; h[s % h.size()] = s >> 8;}
        std::vector<uint64_t> ref(h);
        for(size_t ii = h.size(); ii--;) {
            if(h[ii] != E) continue;
            uint64_t j = ii, a = 0;
            do j = minhash::detail::twounivhash(ii, ++a) % h.size(); while(h[j] == E);
            ref[ii] = h[j];
        }
        std::vector<uint64_t> d(h);
        if(minhash::detail::densifybin(d) != 1) {std::fprintf(stderr, "densifybin rc\n"); return 1;}
        if(d != ref) {std::fprintf(stderr, "densification depends on bin order (nfill %u)\n", nfill); return 1;}
        for(size_t i = 0; i < h.size(); ++i) if(h[i] != E && d[i] != h[i]) {std::fprintf(stderr, "filled bin changed\n"); return 1;}
    }
    // 2. sparse Jaccard accuracy: 30 items per set at p = 12, J = 0.5
    double se = 0; const int R = 200;
    for(int r = 0; r < R; ++r) {
        BBitMinHasher<uint64_t> a(12, 32), b(12, 32);
        const uint64_t off = uint64_t(r) << 44;
        for(uint64_t i = 0; i < 30; ++i) a.addh(off + i);
        for(uint64_t i = 10; i < 40; ++i) b.addh(off + i);
        const double est = a.finalize(32).jaccard_index(b.finalize(32));
        se += (est - 0.5) * (est - 0.5);
    }
    const double rmse = std::sqrt(se / R);
    std::fprintf(stderr, "densifytest: order-independent; sparse Jaccard rmse %.4f (in-place densification: ~0.095)\n", rmse);
    return rmse < 0.07 ? 0: 1;
}
