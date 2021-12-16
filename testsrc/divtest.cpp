#include "common.h"
#include "hk.h"
using namespace sketch;
using namespace common;

template <typename T>
void doNotOptimizeAway(const T& datum) {
  asm volatile("" ::"r"(datum));
}
template <>
void doNotOptimizeAway(const __m256i & datum) {
  const uint32_t ret = *(const uint32_t *)&datum;
  asm volatile("" ::"r"(ret));
}


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
        assert(truem[i] == vp[i]);
    }
    constexpr size_t nitems = 1ull << 16;
    std::vector<uint32_t> u32vals(nitems);
    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitems; ++i) {
        //doNotOptimizeAway(u32vals[i] % 133337);
        doNotOptimizeAway(sm.mod(u32vals[i]));
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < nitems; i += 8) {
        //doNotOptimizeAway(
        sm.mod(_mm256_loadu_si256((const __m256i *)&u32vals[i]));
        //);
    }
    auto t4 = std::chrono::high_resolution_clock::now();
#if 0
    for(size_t i = 0; i < (1ull << 20); i += 16) {
        sm.mod(_mm256_loadu_si256((const __m256i *)&u32vals[i]), _mm256_loadu_si256((const __m256i *)&u32vals[i + 8]));
    }
    auto t6 = std::chrono::high_resolution_clock::now();
#endif
    auto ts1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    auto ts2 = std::chrono::duration<double, std::milli>(t4 - t2).count();
    //auto ts3 = std::chrono::duration<double, std::milli>(t6 - t5).count();
    std::fprintf(stderr, "Time for serial: %g. Time for SIMD: %g\n", ts1, ts2);
}
