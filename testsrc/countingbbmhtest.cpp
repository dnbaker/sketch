// CountingBBitMinHasher: each register holds the minimum masked hash that fell
// in it (same registers as BBitMinHasher) and its counter the multiplicity of
// that minimum in the input; addh hashes and counts like add; and
// histogram_intersection estimates the weighted Jaccard sum(min)/sum(max).
#include "sketch/bbmh.h"
#include <cstdio>
#include <cmath>
#include <map>
#include <vector>
using namespace sketch;

int main() {
    const unsigned p = 8;
    CountingBBitMinHasher<uint64_t, uint32_t> c(p, 32), ch(p, 32);
    BBitMinHasher<uint64_t> plain(p, 32);
    std::vector<std::map<uint64_t, uint32_t>> want(1u << p); // register -> (masked hash -> count)
    uint64_t s = 0x2545F4914F6CDD1Dull;
    for(unsigned i = 0; i < 20000; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const uint64_t base = s % 3000;                 // repeats: multiplicities up to ~15
        const uint64_t h = hash::WangHash()(base);
        c.add(h); plain.add(h); ch.addh(base);
        ++want[h >> (64 - p)][(h << p) >> p];
    }
    const auto &cnt = c.counters();
    for(size_t r = 0; r < want.size(); ++r) {
        if(want[r].empty()) {
            if(cnt[r]) {std::fprintf(stderr, "empty register %zu has count %u\n", r, cnt[r]); return 1;}
            continue;
        }
        if(cnt[r] != want[r].begin()->second) {
            std::fprintf(stderr, "register %zu: count %u, want %u\n", r, cnt[r], want[r].begin()->second); return 1;
        }
    }
    if(!(static_cast<const BBitMinHasher<uint64_t> &>(c) == plain)) {std::fprintf(stderr, "registers differ from BBitMinHasher\n"); return 1;}
    if(ch.counters() != cnt || !(static_cast<const BBitMinHasher<uint64_t> &>(ch) == plain)) {std::fprintf(stderr, "addh differs from add(hash)\n"); return 1;}
    // Weighted Jaccard: 40k keys, multiplicities 1..8, half shared (different counts).
    {
        CountingBBitMinHasher<uint64_t, uint32_t> a(14, 32), b(14, 32);
        double smin = 0, smax = 0;
        for(uint64_t k = 0; k < 40000; ++k) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            const unsigned ca = 1 + s % 8, cb = 1 + (s >> 8) % 8, which = (s >> 16) % 4;
            const unsigned na = which != 3 ? ca: 0, nb = which != 2 ? cb: 0; // 0,1: both; 2: a only; 3: b only
            for(unsigned c = 0; c < na; ++c) a.addh(k);
            for(unsigned c = 0; c < nb; ++c) b.addh(k);
            smin += std::min(na, nb); smax += std::max(na, nb);
        }
        const double truth = smin / smax, est = a.histogram_intersection(b);
        std::fprintf(stderr, "weighted Jaccard: truth %.4f, estimate %.4f\n", truth, est);
        if(std::abs(est - truth) > 0.01) {std::fprintf(stderr, "weighted Jaccard estimate off by more than 0.01\n"); return 1;}
    }
    std::fprintf(stderr, "countingbbmhtest: registers, counts and weighted Jaccard correct\n");
    return 0;
}
