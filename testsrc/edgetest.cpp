// Edge-case regression tests: hashers (zero input, avalanche, inverses,
// width-sensitive shifts) and sketch boundary behavior (empty/singleton
// sketches, precision bounds, incompatible merges, empty serialization).
#include "hash.h"
#include "hll.h"
#include "ultraloglog.h"
#include "exaloglog.h"
#include "fracminhash.h"
#include "dartminhash.h"
#include "hmh.h"
#include <cstdio>
#include <random>

using namespace sketch;
static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

static unsigned hamdist(uint64_t a, uint64_t b) {return unsigned(popcount(a ^ b));}

void test_hashers() {
    // zero input must not crash and must avalanche (WangHash, MurFinHash, wyhash)
    const uint64_t z = 0;
    CHECK(hamdist(hash::WangHash::hash(0), hash::WangHash::hash(1)) >= 10, "WangHash poor avalanche 0 vs 1");
    CHECK(hash::WangHash::hash(0) != 0, "WangHash(0) == 0");
    CHECK(hamdist(hash::finalize(0), hash::finalize(1)) >= 10, "MurFinHash finalize poor avalanche");
    {
        uint64_t s = 0;
        const uint64_t h0 = wy::wyhash64_stateless(&s);
        uint64_t s2 = 1;
        const uint64_t h1 = wy::wyhash64_stateless(&s2);
        CHECK(hamdist(h0, h1) >= 10, "wyhash poor avalanche");
    }
    // WangHash inversion round-trips (comment in hash.h promises it)
    std::mt19937_64 rng(42);
    for(unsigned i = 0; i < 100000; ++i) {
        const uint64_t v = rng();
        CHECK(hash::WangHash().inverse(hash::WangHash::hash(v)) == v, "WangHash inverse mismatch at %llu", (unsigned long long)v);
    }
    // Rot on uint64_t: round-trip through inverse
    for(unsigned i = 0; i < 1000; ++i) {
        const uint64_t v = rng();
        CHECK(hash::RotL33()(hash::RotR33()(v)) == v, "Rot33 round-trip failed");
    }
    // Rot on a sub-64-bit type: shifts must respect T's width (UB before)
    for(unsigned i = 0; i < 1000; ++i) {
        const uint32_t v = uint32_t(rng());
        CHECK((hash::Rot<5, true>()(hash::Rot<5, false>()(v))) == v, "Rot<uint32_t,5> round-trip failed");
        CHECK((hash::Rot<31, true>()(hash::Rot<31, false>()(v))) == v, "Rot<uint32_t,31> round-trip failed");
    }
    std::fprintf(stderr, "hashers edge tests done\n");
}

void test_sketch_bounds() {
    // hll: default-constructed (np=0) must be usable only after resize; report on empty == 0
    {
        hll::hll_t h(4);
        CHECK(h.report() == 0., "empty hll report != 0");
        h.addh(0); // single element, minimal p
        CHECK(std::isfinite(h.report()) && h.report() >= 0., "single-element hll(p=4) estimate bad: %f", h.report());
        bool threw = false;
        try { hll::hll_t bad(63); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "hll accepted np=63");
        threw = false;
        try { hll::hll_t a(10), b(11); a += b; } catch(...) { threw = true; }
        CHECK(threw, "hll merged mismatched sizes without error");
    }
    // empty serialization round-trips
    {
        hll::hll_t h(8), h2;
        h.write("/tmp/edge_hll.hll"); h2.read("/tmp/edge_hll.hll");
        CHECK(h == h2 && h2.report() == 0., "empty hll round-trip failed");
        std::remove("/tmp/edge_hll.hll");
    }
    {
        ull::ultraloglog_t h(3); // min p
        CHECK(h.report() == 0., "empty ull report != 0");
        h.addh(0); h.addh(0);
        CHECK(h.report() > 0., "single-element ull estimate 0");
        ull::ultraloglog_t h2(3);
        h2.write("/tmp/edge_ull.ull");
        ull::ultraloglog_t h3("/tmp/edge_ull.ull");
        CHECK(h2 == h3, "empty ull round-trip failed");
        std::remove("/tmp/edge_ull.ull");
        bool threw = false;
        try { ull::ultraloglog_t bad(2); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "ull accepted p=2");
        ull::ultraloglog_t h4(4);
        h2.add(h4); // larger into smaller is legal (downsize)
        threw = false;
        try { h4.add(h2); } catch(const std::invalid_argument &) { threw = true; } // smaller into larger is not
        CHECK(threw, "ull accepted merging smaller precision");
    }
    {
        ell::exaloglog_t h(0, 0, 2); // t=0, d=0, p=2: plain HLL-like registers
        CHECK(h.report() == 0., "empty ell report != 0");
        h.addh(0);
        CHECK(h.report() > 0., "single-element ell(t=0,d=0) estimate 0");
        ell::exaloglog_t h2(0, 0, 2);
        h2.write("/tmp/edge_ell.ell");
        ell::exaloglog_t h3("/tmp/edge_ell.ell");
        CHECK(h2 == h3, "empty ell round-trip failed");
        std::remove("/tmp/edge_ell.ell");
        bool threw = false;
        try { ell::exaloglog_t bad(24, 35, 2); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "ell accepted d > max_d(t)");
    }
    {
        fmh::fracminhash_t h(uint64_t(1) << 32); // keep 1/2^32
        CHECK(h.report() == 0. && h.empty(), "empty fmh report != 0");
        h.addh(0); h.addh(0); // hash(0) deterministically kept or not; idempotent either way
        fmh::fracminhash_t h2(uint64_t(1) << 32);
        h2.write("/tmp/edge_fmh.fmh");
        fmh::fracminhash_t h3("/tmp/edge_fmh.fmh");
        CHECK(h2 == h3, "empty fmh round-trip failed");
        std::remove("/tmp/edge_fmh.fmh");
    }
    {
        dmh::dartminhash_t h(2, 7); // minimum k
        CHECK(h.report() == 0., "empty dmh report != 0");
        h.addh(0);
        CHECK(h.report() > 0., "single-element dmh(k=2) estimate 0");
        dmh::dartminhash_t h2(2, 7);
        h2.write("/tmp/edge_dmh.dmh");
        dmh::dartminhash_t h3(2, 7);
        h3.read("/tmp/edge_dmh.dmh");
        CHECK(h2 == h3, "empty dmh round-trip failed");
        std::remove("/tmp/edge_dmh.dmh");
        bool threw = false;
        try { dmh::dartminhash_t bad(1); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "dmh accepted k=1");
    }
    {
        HyperMinHash hm(3, 8); // minimum sizes
        CHECK(hm.getcard() == 0., "empty hmh card != 0");
        hm.addh(0);
        CHECK(std::isfinite(hm.getcard()) && hm.getcard() >= 0., "single-element hmh card bad");
    }
    std::fprintf(stderr, "sketch bounds tests done\n");
}

int main() {
    test_hashers();
    test_sketch_bounds();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "all edge-case tests passed\n");
    return 0;
}
