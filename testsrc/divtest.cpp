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

__m256i mod256_div(__m256i x, const schism::Schismatic<uint32_t> &d) {
    for(size_t i = 0; i < 8; ++i) {
        ((uint32_t *)&x)[i] %= d.mod(((uint32_t *)&x)[i]);
    }
    return x;
}

__m256i mod256_man(__m256i x, uint32_t d) {
    for(size_t i = 0; i < 8; ++i) {
        ((uint32_t *)&x)[i] %= d;
    }
    return x;
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
    std::mt19937 mt;
    for(const auto sz0: {0, 1}) {
        const uint32_t sz = mt();
        schism::Schismatic<uint32_t> sm(sz);
        __m256i vals;
        uint32_t *vp = (uint32_t *)&vals;
        std::mt19937 mt;
        std::array<uint32_t, 8> truem;
        for(size_t i = 0; i < 8; ++i) {
            vp[i] = mt();
            truem[i] = vp[i] % sz;
        }
        auto vmod = sm.mod(vals);
        assert(std::equal(truem.begin(), truem.end(), (uint32_t *)&vmod));
        const size_t nitems = (static_cast<size_t>(sz) * sz) & 0xFFFFFFFull;
        uint8_t *buf = new uint8_t[nitems * 4 + 63];
        uint8_t *bptr = buf;
        while(reinterpret_cast<uint64_t>(bptr) % 64) ++bptr;
        uint32_t *u32vals = (uint32_t *)bptr;
        std::iota(u32vals, u32vals + nitems, 0xFFFFFFFu);
        auto t1 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitems; doNotOptimizeAway(sm.mod(u32vals[i++])));
        auto t2 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitems / 8; ++i) {
            doNotOptimizeAway(
            sm.mod(_mm256_load_si256((const __m256i *)&u32vals[i * 8]))
            );
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitems; i += 8) {
            //doNotOptimizeAway(u32vals[i] %= sz);
            doNotOptimizeAway(mod256_man(_mm256_load_si256((const __m256i *)&u32vals[i]), sz));
        }
        auto t6 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitems; i += 8) {
            //doNotOptimizeAway(u32vals[i] %= sz);
            doNotOptimizeAway(mod256_div(_mm256_load_si256((const __m256i *)&u32vals[i]), sm));
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        auto ts1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
        auto ts2 = std::chrono::duration<double, std::milli>(t4 - t2).count();
        auto ts3 = std::chrono::duration<double, std::milli>(t6 - t4).count();
        auto ts4 = std::chrono::duration<double, std::milli>(t7 - t6).count();
        //auto ts3 = std::chrono::duration<double, std::milli>(t6 - t5).count();
        std::fprintf(stderr, "[%d] Time for serial: %g. Time for SIMD: %g. Time for serial compiler %g. Time for serial div: %g\n", sz, ts1, ts2, ts3, ts4);
        t7 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitems; doNotOptimizeAway(u32vals[i++] % sz));
        auto t8 = std::chrono::high_resolution_clock::now();
        auto ts5 = std::chrono::duration<double, std::milli>(t8 - t7).count();
        std::fprintf(stderr, "[%d] %gms for \n", sz, ts5);
    }
}
