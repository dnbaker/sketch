#include "sketch/count_eq.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

using namespace sketch;
using namespace sketch::eq;

static size_t nchecks = 0, nfail = 0;

#define CHECK_EQ(a, b, ctx) do { \
    ++nchecks; \
    if((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s: got %llu want %llu (%s)\n", __FILE__, __LINE__, ctx, \
                     (unsigned long long)(a), (unsigned long long)(b), __PRETTY_FUNCTION__); \
        ++nfail; \
    } \
} while(0)

// ---------------- scalar references ----------------
template<typename T>
static size_t ref_count_eq(const T *lhs, const T *rhs, size_t n) {
    size_t ret = 0;
    for(size_t i = 0; i < n; ++i) ret += lhs[i] == rhs[i];
    return ret;
}
template<typename T>
static std::pair<uint64_t, uint64_t> ref_count_gtlt(const T *lhs, const T *rhs, size_t n) {
    uint64_t a = 0, b = 0;
    for(size_t i = 0; i < n; ++i) { a += lhs[i] > rhs[i]; b += rhs[i] > lhs[i]; }
    return {a, b};
}
static size_t ref_count_eq_nibbles(const uint8_t *lhs, const uint8_t *rhs, size_t nelem) {
    size_t ret = 0;
    auto nib = [](const uint8_t *p, size_t i) { return (p[i >> 1] >> ((i & 1) * 4)) & 0xFu; };
    for(size_t i = 0; i < nelem; ++i) ret += nib(lhs, i) == nib(rhs, i);
    return ret;
}
static std::pair<uint64_t, uint64_t> ref_count_gtlt_nibbles(const uint8_t *lhs, const uint8_t *rhs, size_t nelem) {
    uint64_t a = 0, b = 0;
    auto nib = [](const uint8_t *p, size_t i) { return (p[i >> 1] >> ((i & 1) * 4)) & 0xFu; };
    for(size_t i = 0; i < nelem; ++i) { a += nib(lhs, i) > nib(rhs, i); b += nib(rhs, i) > nib(lhs, i); }
    return {a, b};
}

// ---------------- fill helpers ----------------
template<typename T>
static void fill_random(T *p, size_t n, DefaultRNGType &rng) {
    for(size_t i = 0; i < n; ++i) p[i] = T(rng());
}
static void fill_random(float *p, size_t n, DefaultRNGType &rng) {
    for(size_t i = 0; i < n; ++i) {
        uint64_t v = rng();
        if((v & 0x3) == 0) { // sprinkle special values: NaN, inf, -0
            static const uint32_t specials[] = {0x7FC00000u, 0xFFC00000u, 0x7F800000u, 0xFF800000u, 0x80000000u, 0x00000000u};
            uint32_t u = specials[(v >> 8) % 6];
            std::memcpy(&p[i], &u, 4);
        } else {
            p[i] = float(int64_t(v)) * 1e-6f;
        }
    }
}
static void fill_random(double *p, size_t n, DefaultRNGType &rng) {
    for(size_t i = 0; i < n; ++i) {
        uint64_t v = rng();
        if((v & 0x3) == 0) {
            static const uint64_t specials[] = {0x7FF8000000000000ull, 0xFFF8000000000000ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x8000000000000000ull, 0};
            uint64_t u = specials[(v >> 8) % 6];
            std::memcpy(&p[i], &u, 8);
        } else {
            p[i] = double(int64_t(v)) * 1e-6;
        }
    }
}
template<typename T>
static void fill_signed_boundary(T *p, size_t n, DefaultRNGType &rng) {
    // concentrate mass at the signed/unsigned boundary of T's range
    using U = std::make_unsigned_t<T>;
    for(size_t i = 0; i < n; ++i) {
        uint64_t v = rng();
        switch(v % 4) {
            case 0: p[i] = T(U(1) << (sizeof(T) * 8 - 1)); break; // 0x80..00
            case 1: p[i] = T((U(1) << (sizeof(T) * 8 - 1)) - 1); break; // 0x7F..FF
            case 2: p[i] = T(~U(0)); break;
            default: p[i] = T(v); break;
        }
    }
}
static void fill_mixed_bytes(uint8_t *p, size_t n, DefaultRNGType &rng) {
    for(size_t i = 0; i < n; ++i) {
        uint64_t v = rng();
        switch(v % 4) {
            case 0: p[i] = 0x80u; break;
            case 1: p[i] = 0x7Fu; break;
            case 2: p[i] = 0xFFu; break;
            default: p[i] = uint8_t(v); break;
        }
    }
}

// ---------------- drivers ----------------
template<typename T>
static void test_eq_type(const char *name) {
    for(uint64_t seed: {0ull, 1ull, 0xdeadbeefcafeull, 0x9e3779b97f4a7c15ull}) {
        DefaultRNGType rng(seed);
        for(size_t n: {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(96), size_t(127), size_t(128), size_t(129), size_t(255), size_t(257), size_t(1000), size_t(4095), size_t(4096), size_t(4097)}) {
            for(size_t off: {size_t(0), size_t(1), size_t(3)}) {
                std::vector<T> bufa(n + off + 1), bufb(n + off + 1);
                fill_signed_boundary(bufa.data(), n + off + 1, rng);
                fill_signed_boundary(bufb.data(), n + off + 1, rng);
                const T *lhs = bufa.data() + off, *rhs = bufb.data() + off;
                std::string ctx = std::string(name) + " eq n=" + std::to_string(n) + " off=" + std::to_string(off);
                CHECK_EQ(count_eq(lhs, rhs, n), ref_count_eq(lhs, rhs, n), ctx.c_str());
                auto got = count_gtlt(lhs, rhs, n), want = ref_count_gtlt(lhs, rhs, n);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
            }
        }
    }
}

static void test_eq_bytes() {
    for(uint64_t seed: {0ull, 1ull, 0xdeadbeefcafeull, 0x9e3779b97f4a7c15ull}) {
        DefaultRNGType rng(seed);
        for(size_t n: {size_t(0), size_t(1), size_t(2), size_t(3), size_t(7), size_t(8), size_t(15), size_t(16), size_t(31), size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(95), size_t(96), size_t(127), size_t(128), size_t(129), size_t(255), size_t(256), size_t(257), size_t(511), size_t(1000), size_t(4095), size_t(4096), size_t(4097)}) {
            for(size_t off: {size_t(0), size_t(1), size_t(5), size_t(17), size_t(31), size_t(33)}) {
                std::vector<uint8_t> bufa(n + off + 65), bufb(n + off + 65);
                fill_mixed_bytes(bufa.data(), bufa.size(), rng);
                fill_mixed_bytes(bufb.data(), bufb.size(), rng);
                const uint8_t *lhs = bufa.data() + off, *rhs = bufb.data() + off;
                std::string ctx = "bytes eq n=" + std::to_string(n) + " off=" + std::to_string(off);
                CHECK_EQ(count_eq_bytes(lhs, rhs, n), ref_count_eq(lhs, rhs, n), ctx.c_str());
                CHECK_EQ(count_eq(lhs, rhs, n), ref_count_eq(lhs, rhs, n), ctx.c_str());
                auto got = count_gtlt_bytes(lhs, rhs, n), want = ref_count_gtlt(lhs, rhs, n);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
                got = count_gtlt(lhs, rhs, n); want = ref_count_gtlt(lhs, rhs, n);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
                // aligned-entry dispatch (both pointers 64B-aligned within buffer)
                if(n >= 64) {
                    size_t aoff = (64 - (uintptr_t(bufa.data()) & 63)) & 63;
                    size_t boff = (64 - (uintptr_t(bufb.data()) & 63)) & 63;
                    const uint8_t *al = bufa.data() + aoff, *bl = bufb.data() + boff;
                    std::string actx = ctx + " aligned64";
                    CHECK_EQ(count_eq_bytes(al, bl, n), ref_count_eq(al, bl, n), actx.c_str());
                    auto agt = count_gtlt_bytes(al, bl, n), awant = ref_count_gtlt(al, bl, n);
                    CHECK_EQ(agt.first, awant.first, actx.c_str());
                    CHECK_EQ(agt.second, awant.second, actx.c_str());
                }
            }
        }
    }
}

static void test_nibbles() {
    for(uint64_t seed: {0ull, 1ull, 0xdeadbeefcafeull, 0x9e3779b97f4a7c15ull}) {
        DefaultRNGType rng(seed);
        for(size_t nelem: {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(127), size_t(129), size_t(255), size_t(257), size_t(1000), size_t(4095), size_t(4096), size_t(4097)}) {
            for(size_t off: {size_t(0), size_t(1), size_t(3)}) {
                size_t nbytes = (nelem + 1) / 2;
                std::vector<uint8_t> bufa(nbytes + off + 33), bufb(nbytes + off + 33);
                fill_mixed_bytes(bufa.data(), bufa.size(), rng);
                fill_mixed_bytes(bufb.data(), bufb.size(), rng);
                const uint8_t *lhs = bufa.data() + off, *rhs = bufb.data() + off;
                std::string ctx = "nibbles nelem=" + std::to_string(nelem) + " off=" + std::to_string(off);
                CHECK_EQ(count_eq_nibbles(lhs, rhs, nelem), ref_count_eq_nibbles(lhs, rhs, nelem), ctx.c_str());
                auto got = count_gtlt_nibbles(lhs, rhs, nelem), want = ref_count_gtlt_nibbles(lhs, rhs, nelem);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
            }
        }
    }
}

static void test_gtlt_words() {
    for(uint64_t seed: {uint64_t(0), uint64_t(1), 0xdeadbeefcafeull, 0x9e3779b97f4a7c15ull}) {
        DefaultRNGType rng(seed);
        for(size_t n: {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(127), size_t(129), size_t(1000), size_t(4095), size_t(4096), size_t(4097)}) {
            for(size_t off: {size_t(0), size_t(1), size_t(3)}) {
                std::vector<uint32_t> bufa(n + off + 1), bufb(n + off + 1);
                fill_signed_boundary(bufa.data(), bufa.size(), rng);
                fill_signed_boundary(bufb.data(), bufb.size(), rng);
                const uint32_t *lhs = bufa.data() + off, *rhs = bufb.data() + off;
                std::string ctx = "u32 words gtlt n=" + std::to_string(n) + " off=" + std::to_string(off);
                auto got = count_gtlt_words(lhs, rhs, n), want = ref_count_gtlt(lhs, rhs, n);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
                if(n >= 16) {
                    size_t aoff = ((64 - (uintptr_t(bufa.data()) & 63)) & 63) / sizeof(uint32_t);
                    size_t boff = ((64 - (uintptr_t(bufb.data()) & 63)) & 63) / sizeof(uint32_t);
                    if(aoff + n <= bufa.size() && boff + n <= bufb.size()) {
                        const uint32_t *al = bufa.data() + aoff, *bl = bufb.data() + boff;
                        std::string actx = ctx + " aligned64";
                        auto agt = count_gtlt_words(al, bl, n), awant = ref_count_gtlt(al, bl, n);
                        CHECK_EQ(agt.first, awant.first, actx.c_str());
                        CHECK_EQ(agt.second, awant.second, actx.c_str());
                    }
                }
            }
        }
    }
}

template<typename T>
static void test_gtlt_fp(const char *name) {    for(uint64_t seed: {0ull, 1ull, 0xdeadbeefcafeull, 0x9e3779b97f4a7c15ull}) {
        DefaultRNGType rng(seed);
        for(size_t n: {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4), size_t(5), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(63), size_t(64), size_t(65), size_t(1000), size_t(4095), size_t(4096), size_t(4097)}) {
            for(size_t off: {size_t(0), size_t(1), size_t(3)}) {
                std::vector<T> bufa(n + off + 1), bufb(n + off + 1);
                fill_random(bufa.data(), bufa.size(), rng);
                fill_random(bufb.data(), bufb.size(), rng);
                const T *lhs = bufa.data() + off, *rhs = bufb.data() + off;
                std::string ctx = std::string(name) + " gtlt n=" + std::to_string(n) + " off=" + std::to_string(off);
                auto got = count_gtlt(lhs, rhs, n), want = ref_count_gtlt(lhs, rhs, n);
                CHECK_EQ(got.first, want.first, ctx.c_str());
                CHECK_EQ(got.second, want.second, ctx.c_str());
            }
        }
    }
}

int main() {
    std::fprintf(stderr, "simd_eq ISA: AVX2=%d AVX512F=%d AVX512BW=%d\n",
#ifdef __AVX2__
    1,
#else
    0,
#endif
#ifdef __AVX512F__
    1,
#else
    0,
#endif
#ifdef __AVX512BW__
    1
#else
    0
#endif
    );
    test_eq_bytes();
    test_nibbles();
    test_eq_type<uint16_t>("u16");
    test_eq_type<uint32_t>("u32");
    test_eq_type<uint64_t>("u64");
    test_gtlt_words();
    test_gtlt_fp<float>("f32");
    test_gtlt_fp<double>("f64");
    std::fprintf(stderr, "simd_eq: %zu checks, %zu failures\n", nchecks, nfail);
    return nfail ? 1 : 0;
}
