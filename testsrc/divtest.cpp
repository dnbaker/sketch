#include "common.h"
#include "hk.h"
using namespace sketch;
using namespace common;

int main() {
    size_t maxn = 10000000;
    for(size_t i = 10; i < maxn; i *= 10) {
        schism::Schismatic<uint64_t> div_(i);
        for(size_t j = 10000; j < 1200000; j += (std::rand() & 0xFFFU)) {
            auto div = div_.div(j);
            auto mod = div_.mod(j);
            assert(div == j / i);
            assert(mod == j % i);
        }
    }
    for(size_t i = 10; i < maxn; i *= 10) {
        schism::Schismatic<uint32_t> div_(i);
        for(size_t j = 10000; j < 1200000; j += (std::rand() & 0xFFFU)) {
            auto div = div_.div(j);
            auto mod = div_.mod(j);
            assert(div == j / i);
            assert(mod == j % i);
        }
    }
    for(size_t i = 10; i < maxn; i *= 10) {
        schism::Schismatic<uint64_t, /*shortcircuit=*/ true> div_(i);
        for(size_t j = 10000; j < 1200000; j += (std::rand() & 0xFFFU)) {
            auto div = div_.div(j);
            auto mod = div_.mod(j);
            assert(div == j / i);
            assert(mod == j % i);
        }
    }
    policy::SizePow2Policy<uint64_t> pol(1000);
    for(size_t j = 10; j < maxn * 10; j *= 10) {
        policy::SizePow2Policy<uint64_t> p1(j);
        policy::SizeDivPolicy<uint64_t> p2(j);
        std::fprintf(stderr, "input: %zu. power of two: %zu. div: %zu.\n", j, p1.nelem(), p2.nelem());
    }
    schism::Schismatic<uint32_t> sm(133337);
#if 0
    __m256i vals;
    uint32_t *vp = (uint32_t *)&vals;
    std::mt19937 mt;
    std::array<uint32_t, 8> truem;
    for(size_t i = 0; i < 8; ++i) {
        vp[i] = mt();
        truem[i] = vp[i] % 133337;
    }
    auto vmod = sm.mod(vals);
    vp = (uint32_t *)&vmod;
    for(size_t i = 0; i < 8; ++i) {
        std::fprintf(stderr, "True mod: %u. SIMD mod: %u\n", truem[i], vp[i]);
    }
#endif
}
