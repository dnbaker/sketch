// Golden values for the xxHash entry points sketch uses (XXH32, XXH64 one-shot
// and streaming, XXH3_64bits with and without seed, XXH3 streaming), so that an
// update of the vendored include/xxHash cannot silently change any hash and
// therefore any stored sketch. Lengths 0..1100 plus a few large ones cover every
// XXH3 code path (0, 1-3, 4-8, 9-16, 17-128, 129-240, >240 with stripes/blocks),
// and inputs start at every offset 0..7 to exercise unaligned loads.
// Build with -DXXH_VECTOR=0 as well to check the scalar path gives the same values.
// Pass any argument to print the digests instead of checking them.
#include "xxHash/xxh3.h"
#include "xxHash/xxhash.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t mix(uint64_t acc, uint64_t h) {
    acc ^= h + 0x9E3779B97F4A7C15ull + (acc << 6) + (acc >> 2);
    return acc * 0xff51afd7ed558ccdull;
}

int main(int argc, char **argv) {
    std::vector<unsigned char> buf((1 << 20) + 8);
    uint64_t s = 0x243F6A8885A308D3ull;
    for(auto &c: buf) {s = s * 6364136223846793005ull + 1442695040888963407ull; c = s >> 56;}
    std::vector<size_t> lens;
    for(size_t i = 0; i <= 1100; ++i) lens.push_back(i);
    for(size_t l: {2047, 2048, 2049, 4096, 10007, 65536, 1 << 20}) lens.push_back(l);
    const uint64_t seeds[] = {0, 1, 0xDEADBEEFull, 0x9E3779B97F4A7C15ull};
    uint64_t d32 = 0, d64 = 0, d64s = 0, d3 = 0, d3seed = 0, d3s = 0;
    for(size_t off = 0; off < 8; ++off) {
        for(size_t L: lens) {
            if(off + L > buf.size()) continue;
            const unsigned char *p = buf.data() + off;
            for(uint64_t seed: seeds) {
                d32 = mix(d32, XXH32(p, L, uint32_t(seed)));
                d64 = mix(d64, XXH64(p, L, seed));
                d3seed = mix(d3seed, XXH3_64bits_withSeed(p, L, seed));
            }
            d3 = mix(d3, XXH3_64bits(p, L));
            // streaming, in uneven chunks
            XXH64_state_t st64; XXH64_reset(&st64, 7);
            XXH3_state_t st3; XXH3_64bits_reset(&st3);
            for(size_t i = 0, c = 1; i < L; i += c, c = c * 3 + 1) {
                const size_t n = std::min(c, L - i);
                XXH64_update(&st64, p + i, n);
                XXH3_64bits_update(&st3, p + i, n);
            }
            const uint64_t h64 = XXH64_digest(&st64), h3 = XXH3_64bits_digest(&st3);
            if(h64 != XXH64(p, L, 7) || h3 != XXH3_64bits(p, L)) {
                std::fprintf(stderr, "streaming != one-shot at off %zu len %zu\n", off, L);
                return 1;
            }
            d64s = mix(d64s, h64);
            d3s = mix(d3s, h3);
        }
    }
    // Digests produced by the xxHash snapshot vendored at sketch f4931c4 (post-0.8.0)
    struct {const char *name; uint64_t got, want;} r[] = {
        {"XXH32", d32, 0x025306d4c3d8548bull},
        {"XXH64", d64, 0x558af3334d960604ull},
        {"XXH64 streaming", d64s, 0xaedaa1e0a5813d0bull},
        {"XXH3_64bits", d3, 0xe0e77f976a6b28ecull},
        {"XXH3_64bits_withSeed", d3seed, 0xa75c46ad94b0b2cfull},
        {"XXH3 streaming", d3s, 0xe0e77f976a6b28ecull},
    };
    int bad = 0;
    for(auto &x: r) {
        if(argc > 1) std::printf("        {\"%s\", %s, 0x%016" PRIx64 "ull},\n", x.name, "...", x.got);
        else if(x.got != x.want) {std::fprintf(stderr, "%s digest changed: %016" PRIx64 " != %016" PRIx64 "\n", x.name, x.got, x.want); bad = 1;}
    }
    if(argc == 1 && !bad) std::fprintf(stderr, "xxhgoldentest: %zu lengths x 8 offsets, all digests match\n", lens.size());
    return bad;
}
