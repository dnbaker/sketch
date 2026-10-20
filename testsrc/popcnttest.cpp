// libpopcnt and the SIMD popcount helpers sketch builds on it (popcnt over
// buffers of every length and offset; popcnt256 per-64-bit-lane counts) must
// agree with __builtin_popcountll, so a libpopcnt update cannot change counts.
#include "sketch/integral.h"
#include <cstdio>
#include <vector>

int main() {
    std::vector<uint64_t> w(1 << 14);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for(auto &x: w) {s ^= s << 13; s ^= s >> 7; s ^= s << 17; x = s;}
    for(size_t i = 0; i < w.size(); i += 97) w[i] = 0, w[i + 1 < w.size() ? i + 1 : i] = ~uint64_t(0);
    const unsigned char *b = reinterpret_cast<const unsigned char *>(w.data());
    const size_t nb = w.size() * sizeof(uint64_t);
    for(size_t off = 0; off < 16; ++off) {
        for(size_t len = 0; off + len <= nb; len = len < 600 ? len + 1 : len * 2 + 3) {
            uint64_t want = 0;
            for(size_t i = 0; i < len; ++i) want += __builtin_popcount(b[off + i]);
            const uint64_t got = popcnt(b + off, len);
            if(got != want) {std::fprintf(stderr, "popcnt off %zu len %zu: %llu != %llu\n", off, len, (unsigned long long)got, (unsigned long long)want); return 1;}
        }
    }
#if defined(__AVX2__)
    for(size_t i = 0; i + 4 <= w.size(); i += 4) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&w[i]));
        alignas(32) uint64_t lanes[4];
        _mm256_store_si256(reinterpret_cast<__m256i *>(lanes), popcnt256(v));
        for(int j = 0; j < 4; ++j)
            if(lanes[j] != uint64_t(__builtin_popcountll(w[i + j]))) {std::fprintf(stderr, "popcnt256 lane %d at %zu\n", j, i); return 1;}
    }
#endif
    std::fprintf(stderr, "popcnttest: popcnt%s match __builtin_popcount\n",
#if defined(__AVX2__)
                 " and popcnt256"
#else
                 ""
#endif
                 );
    return 0;
}
