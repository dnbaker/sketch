// CQF gates: reference stress (both mixes), saturation, serialization,
// merge, mid-stream grow, edge cases, map-API conformance, iterators.
#include "cqf.h"
#include <cstdio>
#include <chrono>
#include <random>
#include <unordered_map>
#include <sys/resource.h>
#include <numeric>
#include <algorithm>
#include <unordered_map>

using namespace sketch;
using clk = std::chrono::high_resolution_clock;
static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

static INLINE uint64_t hred(const cqf::cqf64_64 &f, uint64_t v) {
    const auto p = f.parts(f.hash(v));
    return (p.first << f.r()) | p.second;
}

template<typename Q>
static void full_verify(const Q &f, const std::unordered_map<uint64_t, uint64_t> &ref, const char *tag) {
    for(const auto &kv: ref) {
        const uint64_t c = f.count(kv.first);
        if(c != kv.second) {
            CHECK(false, "cqf mismatch (%s): got %llu want %llu (key %llu)", tag,
                  (unsigned long long)c, (unsigned long long)kv.second, (unsigned long long)kv.first);
            return;
        }
    }
}

void test_fill_levels(double delete_frac, uint32_t qbits=10) {
    const char *tag = delete_frac > 0.6 ? "del70": "mix";
    for(const double target: {0.25, 0.5, 0.75, 0.9}) {
        for(const uint32_t seed: {1u, 7u, 99u}) {
            cqf::cqf64_64 f(qbits);
            const uint64_t nslots = f.nslots();
            const uint64_t target_items = uint64_t(nslots * target);
            std::mt19937_64 rng(seed * 31337 + unsigned(target * 100));
            std::unordered_map<uint64_t, uint64_t> ref;
            while(ref.size() < target_items) {
                const uint64_t k = hred(f, rng());
                if(f.count(k) == 0) { f.add(k); ref[k] = 1; }
            }
            CHECK(std::abs(f.load_factor() - target) < 0.03, "fill off %.3f vs %.2f", f.load_factor(), target);
            full_verify(f, ref, "fill");
            f.structure_check();
            for(unsigned op = 0; op < 40000; ++op) {
                const uint64_t k = hred(f, rng());
                const uint64_t toss = rng();
                if((toss % 100) < unsigned(delete_frac * 100) && ref.count(k)) {
                    CHECK(f.dec(k), "dec of present failed (%s)", tag);
                    if(--ref[k] == 0) ref.erase(k);
                } else if((toss % 100) < unsigned(delete_frac * 100) + 15 && ref.size() > 8) {
                    auto it = ref.begin();
                    std::advance(it, rng() % ref.size());
                    CHECK(f.erase(it->first), "erase failed (%s)", tag);
                    ref.erase(it);
                } else if(ref.count(k) || ref.size() < target_items) {
                    CHECK(f.count(k) == (ref.count(k) ? ref.at(k) : 0), "count in mix (%s)", tag);
                    f.add(k);
                    ++ref[k];
                }
                if((op & 2047) == 2047) { full_verify(f, ref, "mix"); f.structure_check(); }
            }
            full_verify(f, ref, "final");
            f.structure_check();
            std::fprintf(stderr, "cqf %s fill %.2f seed %u done (load %.3f)\n", tag, target, seed, f.load_factor());
        }
    }
}

void test_map_api() {
    // exact-mode conformance, modeled on unordered_map behavior
    cqf::cqf64_64 f(6);
    CHECK(f.empty() && f.size() == 0, "fresh map not empty");
    f.add(100); f.add(100); f.add(200);
    CHECK(f.count(100) == 2 && f.count(200) == 1 && f.count(300) == 0, "map counts wrong");
    CHECK(f.size() == 2, "map size %llu != 2", (unsigned long long)f.size());
    CHECK(f.erase(100), "erase 100 failed"); // erases entirely
    CHECK(f.count(100) == 0 && f.size() == 1, "erase semantics wrong");
    CHECK(!f.erase(999), "erase of absent returned true");
    // re-insert after erase
    f.add(100);
    CHECK(f.count(100) == 1, "re-add after erase wrong");
    // dec to zero also frees
    f.add(300); f.add(300);
    f.dec(300); f.dec(300);
    CHECK(f.count(300) == 0 && f.size() == 2, "dec-to-zero semantics wrong");
    // iterator visits all live pairs
    std::unordered_map<uint64_t, uint64_t> seen;
    for(auto kv: f) ++seen[kv.first];
    CHECK(seen.size() == f.size(), "iterator visited %zu != size %llu", seen.size(), (unsigned long long)f.size());
    for(auto kv: f) CHECK(f.count(kv.first) == kv.second, "iterator pair mismatch");
    f.clear();
    CHECK(f.empty() && f.begin() == f.end(), "clear failed");
    // STL surface: insert/emplace/find/operator[] proxy
    f.insert(400);
    f.insert(std::make_pair(uint64_t(410), uint64_t(4)));
    f.emplace(420, 3);
    CHECK(f.count(400) == 1 && f.count(410) == 4 && f.count(420) == 3, "insert/emplace counts wrong");
    CHECK(f.find(400) != f.end() && f.find(999) == f.end(), "find wrong");
    CHECK(f.find(410) != f.end() && (*f.find(410)).second == 4, "find iterator pair wrong");
    f[500] = 5;
    CHECK(f.count(500) == 5, "operator[] set wrong");
    f[500] += 2;
    CHECK(f.count(500) == 7, "operator[] += wrong");
    f[500] = 1;
    CHECK(f.count(500) == 1, "operator[] shrink wrong");
    --f[500];
    CHECK(f.count(500) == 0, "operator[] -- wrong");
    std::fprintf(stderr, "cqf map-API conformance done\n");
}

void test_grow_and_merge() {
    {   // mid-stream grow at several points
        for(const uint64_t grow_at: {50u, 300u, 900u}) {
            cqf::cqf64_64 f(6); // 64 slots
            std::mt19937_64 rng(grow_at);
            std::unordered_map<uint64_t, uint64_t> ref;
            for(uint64_t i = 0; i < 2000; ++i) {
                const uint64_t k = hred(f, rng());
                f.add(k); ++ref[k];
                if(i == grow_at) { f.structure_check(); f.grow(); f.structure_check(); }
            }
            CHECK(f.q() > 6, "grow did not bump q");
            full_verify(f, ref, "grow");
            f.structure_check();
        }
        std::fprintf(stderr, "cqf grow tests done\n");
    }
        {   // merge vs reference multiset union
        for(const uint32_t seed: {1u, 2u}) {
            cqf::cqf64_64 a(8), b(8);
            std::mt19937_64 rng(seed);
            std::unordered_map<uint64_t, uint64_t> ref;
            for(unsigned i = 0; i < 20000; ++i) {
                const uint64_t k = hred(a, rng());
                a.add(k); ++ref[k];
            }
            for(unsigned i = 0; i < 20000; ++i) {
                const uint64_t k = hred(a, rng());
                b.add(k); ++ref[k];
            }
            a += b;
            full_verify(a, ref, "merge");
            a.structure_check();
        }
        // mismatched q rejects
        bool threw = false;
        try { cqf::cqf64_64 x(6), y(7); x += y; } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "merge accepted mismatched q");
        std::fprintf(stderr, "cqf merge tests done\n");
    }
    {   // fused regime (cqf64_16 at q=16: rem+cnt = 48+16 = 64 bits in one
        // register): same grow/merge/serialize discipline as the packed path
        cqf::cqf64_16 f(15); // packed at q=15 (51+16 > 64)
        CHECK(!f.fused(), "q=15 cqf64_16 unexpectedly fused");
        std::mt19937_64 rng(77);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(unsigned i = 0; i < 40000; ++i) {
            const uint64_t v = rng() % 60000;
            const uint64_t k = f.hash(v);
            if((rng() & 3) == 0 && ref.count(k)) { f.dec(k); if(--ref[k] == 0) ref.erase(k); }
            else { f.add(k); ++ref[k]; }
            if(i == 20000) { f.structure_check(); f.grow(); f.structure_check(); }
        }
        CHECK(f.q() == 16 && f.fused(), "cqf64_16 grow did not land fused at q=16");
        full_verify(f, ref, "fused-grow");
        f.structure_check();
        f.write("/tmp/cqf_fused_rt.gz");
        cqf::cqf64_16 f2(cqf::cqf64_16::from_path, std::string("/tmp/cqf_fused_rt.gz"));
        CHECK(f2.q() == 16 && f2.fused(), "fused round-trip lost the regime");
        full_verify(f2, ref, "fused-round-trip");
        f2.structure_check();
        std::remove("/tmp/cqf_fused_rt.gz");
        // cross-regime: the wire format is packed streams either way, so a
        // fused-regime file must load into a packed-preference filter and
        // vice versa, counts intact
        f.write("/tmp/cqf_xregime.gz");
        cqf::cqf64_16 fp(cqf::cqf64_16::from_path, std::string("/tmp/cqf_xregime.gz"), false);
        CHECK(!fp.fused(), "packed-preference read did not honor the policy");
        full_verify(fp, ref, "fused-to-packed");
        fp.structure_check();
        cqf::cqf64_16 packed(16, false); // forced-packed at the same q
        CHECK(!packed.fused(), "fuse=false did not select the packed regime");
        for(const auto &kv: ref) packed.add_n(kv.first, kv.second);
        full_verify(packed, ref, "packed-build");
        packed.structure_check();
        packed.write("/tmp/cqf_xregime2.gz");
        cqf::cqf64_16 pf(cqf::cqf64_16::from_path, std::string("/tmp/cqf_xregime2.gz"));
        CHECK(pf.q() == 16 && pf.fused(), "default-preference read did not fuse");
        full_verify(pf, ref, "packed-to-fused");
        pf.structure_check();
        std::remove("/tmp/cqf_xregime.gz");
        std::remove("/tmp/cqf_xregime2.gz");
        std::fprintf(stderr, "cqf fused-regime tests done\n");
    }
    {   // bcqf (round-16m block engine): map-API conformance against a
        // reference, mid-stream grow, merge, serialize round-trip
        cqf::bcqf48_16 f(8);
        std::mt19937_64 rng(31);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(unsigned i = 0; i < 40000; ++i) {
            const uint64_t hv = f.reduce(f.hash(rng() % 60000));
            if((rng() & 3) == 0 && ref.count(hv)) { f.dec(hv); if(--ref[hv] == 0) ref.erase(hv); }
            else { f.add(hv); ++ref[hv]; }
            if(i == 20000) { f.structure_check(); f.grow(); f.structure_check(); }
        }
        CHECK(f.q() > 8, "bcqf grow did not bump q");
        full_verify(f, ref, "bcqf-grow");
        f.structure_check();
        {   // counted_ref + STL surface (key drawn from the filter's own
            // hash stream and verified absent: 48-bit reductions of
            // unrelated hashes collide with reference keys by birthday)
            uint64_t k = 0;
            do {k = f.reduce(f.hash(rng()));} while(ref.count(k));
            f[k] = 7;
            CHECK(f.count(k) == 7, "bcqf operator[] write wrong");
            f[k] -= 3;
            CHECK(f.count(k) == 4, "bcqf operator[] -= wrong");
            f.erase(k);
            CHECK(f.count(k) == 0, "bcqf erase wrong");
        }
        f.write("/tmp/bcqf_rt.gz");
        cqf::bcqf48_16 f2(cqf::bcqf48_16::from_path, std::string("/tmp/bcqf_rt.gz"));
        CHECK(f2.q() == f.q(), "bcqf round-trip q mismatch");
        full_verify(f2, ref, "bcqf-round-trip");
        f2.structure_check();
        std::remove("/tmp/bcqf_rt.gz");
        {   // merge vs reference multiset union
            cqf::bcqf48_16 a(8), b(8);
            std::unordered_map<uint64_t, uint64_t> mref;
            for(unsigned i = 0; i < 20000; ++i) { const uint64_t hv = a.reduce(a.hash(rng())); a.add(hv); ++mref[hv]; }
            for(unsigned i = 0; i < 20000; ++i) { const uint64_t hv = a.reduce(a.hash(rng())); b.add(hv); ++mref[hv]; }
            a += b;
            full_verify(a, mref, "bcqf-merge");
            a.structure_check();
            bool threw = false;
            try { cqf::bcqf48_16 x(7), y(8); x += y; } catch(const std::invalid_argument &) { threw = true; }
            CHECK(threw, "bcqf merge accepted mismatched q");
        }
        {   // iteration covers every reference key exactly
            uint64_t seen = 0;
            for(auto kv: f) { CHECK(ref.count(kv.first) && ref[kv.first] == kv.second, "bcqf iterator bad pair"); ++seen; }
            CHECK(seen == ref.size(), "bcqf iterator missed keys");
        }
        {   // SoA layout: same conformance surface, serialize round-trip
            cqf::bcqf_t<48,16,sketch::hash::WangHash,cqf::bcqf_soa_layout> s(8);
            std::unordered_map<uint64_t, uint64_t> sref;
            for(unsigned i = 0; i < 25000; ++i) {
                const uint64_t hv = s.reduce(s.hash(rng() % 40000));
                if((rng() & 3) == 0 && sref.count(hv)) { s.dec(hv); if(--sref[hv] == 0) sref.erase(hv); }
                else { s.add(hv); ++sref[hv]; }
            }
            full_verify(s, sref, "bcqf-soa");
            s.structure_check();
            s.write("/tmp/bcqf_soa_rt.gz");
            decltype(s) s2(decltype(s)::from_path, std::string("/tmp/bcqf_soa_rt.gz"));
            full_verify(s2, sref, "bcqf-soa-round-trip");
            s2.structure_check();
            std::remove("/tmp/bcqf_soa_rt.gz");
        }
        {   // exotic-width instantiations (item 3): u16 and u32 slot
            // stores, serialize round-trip through the logical u64 stream
            cqf::bcqf_t<16,8> w(7);
            for(unsigned i = 0; i < 8000; ++i) w.add(w.reduce(w.hash(rng() % 9000)));
            w.structure_check();
            w.write("/tmp/bcqf_w_rt.gz");
            cqf::bcqf_t<16,8> w2(cqf::bcqf_t<16,8>::from_path, std::string("/tmp/bcqf_w_rt.gz"));
            CHECK(w2.q() == w.q() && w2.size() == w.size(), "bcqf width round-trip mismatch");
            w2.structure_check();
            std::remove("/tmp/bcqf_w_rt.gz");
            const unsigned sb16 = cqf::bcqf_t<16,8>::slot_bytes, sb8 = cqf::bcqf_t<8,1>::slot_bytes;
            CHECK(sb16 == 4 && sb8 == 2, "bcqf slot width selection wrong (%u, %u)", sb16, sb8);
        }
        {   // hash-windowed ranges (item 4): enumeration, union,
            // intersection against reference maps
            cqf::bcqf48_16 r1(10), r2(10);
            std::unordered_map<uint64_t, uint64_t> q1, q2;
            for(unsigned i = 0; i < 20000; ++i) {
                const uint64_t h1 = r1.reduce(r1.hash(rng()));
                const uint64_t h2 = (i & 1) ? h1 : r1.reduce(r1.hash(rng()));
                r1.add(h1); ++q1[h1]; r2.add(h2); ++q2[h2];
            }
            const uint64_t W = 1ULL << 40, lo = (3ULL << 45), hi = lo + W;
            uint64_t n1 = 0, want1 = 0;
            r1.range(lo, hi, [&](uint64_t, uint64_t){++n1;});
            for(const auto &kv: q1) if(kv.first >= lo && kv.first < hi) ++want1;
            CHECK(n1 == want1 && n1 > 10, "bcqf range window wrong (%llu vs %llu)", (unsigned long long)n1, (unsigned long long)want1);
            uint64_t un = 0, uw = 0;
            r1.union_range(r2, lo, hi, [&](uint64_t, uint64_t, uint64_t){++un;});
            for(const auto &kv: q1) if(kv.first >= lo && kv.first < hi) ++uw;
            for(const auto &kv: q2) if(kv.first >= lo && kv.first < hi && !q1.count(kv.first)) ++uw;
            CHECK(un == uw, "bcqf union_range wrong (%llu vs %llu)", (unsigned long long)un, (unsigned long long)uw);
            uint64_t in = 0, iw = 0;
            r1.intersection_range(r2, lo, hi, [&](uint64_t, uint64_t, uint64_t){++in;});
            for(const auto &kv: q1) if(kv.first >= lo && kv.first < hi && q2.count(kv.first)) ++iw;
            CHECK(in == iw, "bcqf intersection_range wrong (%llu vs %llu)", (unsigned long long)in, (unsigned long long)iw);
        }
        {   // build-then-compact -- contents must equal the
            // always-compact construction on the reduced stream
            cqf::bcqf48_16 big(12);
            std::unordered_map<uint64_t, uint64_t> cref; // fp24-space reference
            for(unsigned i = 0; i < 60000; ++i) {
                const uint64_t hv = big.reduce(big.hash(rng()));
                big.add(hv);
                const uint64_t fp = hv & 0xFFFFFFULL;
                const uint64_t nc = cref[fp] + 1;
                cref[fp] = nc > 255 ? 255 : nc; // val8 saturation
            }
            auto c = big.to_compact<24, 8>();
            CHECK(c.size() == cref.size(), "to_compact distinct mismatch (%llu vs %zu)", (unsigned long long)c.size(), cref.size());
            unsigned checked = 0;
            for(const auto &kv: cref) {
                if(c.count(kv.first) != kv.second) { CHECK(false, "to_compact count mismatch"); break; }
                if(++checked > 20000) break;
            }
            c.structure_check();
            // memory: compact slots are the u32 width; the item-6 lock
            // array (64 B per block) keeps the total ratio near 0.6x at
            // small q rather than exactly half -- slots alone halve
            CHECK(c.est_memory_usage().second < big.est_memory_usage().second * 0.65,
                  "to_compact did not shrink memory enough (%zu vs %zu)", c.est_memory_usage().second, big.est_memory_usage().second);
        }
        std::fprintf(stderr, "cqf bcqf tests done\n");
    }
    {   // compose across parameters (rehash)
        cqf::cqfbase_t<64, 8> small(6);
        cqf::cqf64_64 big(7);
        std::mt19937_64 rng(9);
        for(unsigned i = 0; i < 5000; ++i) small.addh(rng());
        big.compose(small);
        for(auto kv: small) CHECK(big.count(kv.first) == kv.second, "compose lost key %llu", (unsigned long long)kv.first);
        std::fprintf(stderr, "cqf compose test done\n");
    }
}

void test_serialization_saturation_edges() {
    {   // saturation at VAL_BITS
        cqf::cqf64_8 f(6);
        for(unsigned i = 0; i < 1000; ++i) f.addh(42);
        CHECK(f.count(f.hash(42)) == 255, "saturation at 255 failed (%llu)", (unsigned long long)f.count(f.hash(42)));
        cqf::cqf64_64 f2(6);
        for(unsigned i = 0; i < 100000; ++i) f2.addh(7);
        CHECK(f2.count(f2.hash(7)) == 100000, "64-bit counter clamped early (%llu)", (unsigned long long)f2.count(f2.hash(7)));
        std::fprintf(stderr, "cqf saturation done\n");
    }
    {   // serialization round-trip
        cqf::cqf64_64 f(8);
        std::mt19937_64 rng(5);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(unsigned i = 0; i < 30000; ++i) {
            const uint64_t k = hred(f, rng());
            if((rng() & 3) == 0 && ref.count(k)) { f.dec(k); if(--ref[k] == 0) ref.erase(k); }
            else if(ref.count(k) || ref.size() < 200) { f.add(k); ++ref[k]; }
        }
        full_verify(f, ref, "pre-ser");
        f.write("/tmp/cqf2_rt.gz");
        cqf::cqf64_64 f2(cqf::cqf64_64::from_path, std::string("/tmp/cqf2_rt.gz"));
        CHECK(f2.q() == f.q(), "round-trip q mismatch");
        full_verify(f2, ref, "round-trip");
        f2.structure_check();
        std::remove("/tmp/cqf2_rt.gz");
        std::fprintf(stderr, "cqf serialization done\n");
    }
    {   // edges: empty, singleton, min size, arg checks, envelope, inexact fpr
        cqf::cqf64_64 f(2);
        CHECK(f.count(7) == 0 && f.empty(), "empty filter wrong");
        f.addh(0);
        CHECK(f.count(f.hash(0)) == 1 && f.size() == 1, "singleton wrong");
        bool threw = false;
        try { cqf::cqf64_64 bad(63); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "accepted q=63");
        threw = false;
        try { cqf::cqf64_64 bad(0); } catch(const std::invalid_argument &) { threw = true; }
        CHECK(threw, "accepted q=0");
        {   // envelope: tiny KEY_BITS must throw overflow, not corrupt
            cqf::cqfbase_t<8, 8> tiny(2); // 4 slots; grow chain ends when remainder bits exhaust
            threw = false;
            try { for(uint64_t i = 0; i < 1000; ++i) tiny.addh(i * 97 + 3); }
            catch(const std::overflow_error &) { threw = true; }
            CHECK(threw, "no overflow at envelope");
        }
        {   // inexact mode: FPR helper sane, no-false-negatives
            cqf::cqf32_8 g(8); // KEY_BITS=32 inexact
            CHECK(!g.exact && g.fpr_bound() >= 0., "fpr helper broken");
            std::mt19937_64 rng(11);
            for(unsigned i = 0; i < 5000; ++i) g.addh(rng() % 5000);
            unsigned fp = 0;
            for(unsigned i = 0; i < 5000; ++i) fp += g.contains(g.hash(1000000 + i * 13));
            CHECK(double(fp) / 5000 < 0.05, "inexact FPR too high: %u/5000", fp);
            CHECK(g.fpr_bound() < 0.05, "fpr_bound inconsistent");
        }
        {   // exact 32/8 instantiation works (val 8-bit)
            cqf::cqf32_8 g(8);
            for(unsigned i = 0; i < 1000; ++i) g.addh(i);
            CHECK(g.size() == 1000, "cqf32_8 size wrong");
            for(unsigned i = 0; i < 1000; ++i) CHECK(g.count(g.hash(i)) == 1, "cqf32_8 lost key");
        }
        std::fprintf(stderr, "cqf edges done\n");
    }
}

// in-binary A/B for the rank popcount route: times the active
// popcount_words against an explicit scalar loop over the same occ vector.
static double bench_popcount_ab(const cqf::cqf64_64 &f) {
    const uint64_t *p = f.occ_w();
    const size_t n = (f.nslots() + 63) / 64;
    const uint64_t reps = 2000;
    volatile uint64_t sink = 0;
    auto t0 = clk::now();
    for(uint64_t r = 0; r < reps; ++r) sink += sketch::cqf::detail::popcount_words(p, n);
    const double dt_route = std::chrono::duration<double>(clk::now() - t0).count();
    t0 = clk::now();
    for(uint64_t r = 0; r < reps; ++r) {
        uint64_t s = 0;
        for(size_t i = 0; i < n; ++i) s += sketch::popcount(p[i]);
        sink += s;
    }
    const double dt_scalar = std::chrono::duration<double>(clk::now() - t0).count();
    std::fprintf(stderr, "popcount A/B: routed %.2f Mwords/s, scalar %.2f Mwords/s (%.1fx)\n",
                 reps * n / dt_route / 1e6, reps * n / dt_scalar / 1e6, dt_scalar / dt_route);
    (void)sink;
    return dt_scalar / dt_route;
}

// Per-op benchmarks at multiple fills and two table sizes: insert (fresh
// distinct keys), query hit/miss, counter update, dec -- all at fixed q so
// numbers reflect the design point (rank/select cost vs load), not growth.
void bench_per_op() {
    for(const uint32_t qbits: {16u, 18u}) {
        for(const double fill: {0.25, 0.5, 0.75, 0.9}) {
            cqf::cqf64_64 f(qbits);
            const uint64_t nslots = f.nslots();
            const uint64_t ceiling = nslots - std::max<uint64_t>(1, nslots / 20);
            const uint64_t target = uint64_t(fill * ceiling);
            std::mt19937_64 rng(42);
            std::unordered_map<uint64_t, uint64_t> ref;
            while(ref.size() < target) { const uint64_t k = hred(f, rng()); if(!f.count(k)) { f.add(k); ref[k] = 1; } }
            const uint64_t nq = 200000;
            std::vector<uint64_t> keys;
            keys.reserve(ref.size());
            for(const auto &kv: ref) keys.push_back(kv.first); // existing keys
            auto sample = [&](uint64_t i) {return keys[(i * 7919) % keys.size()];};
            volatile uint64_t sink = 0; // keep query loops alive
            uint64_t st = 0x9e3779b97f4a7c15ULL;
            auto xnext = [&]() {st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st;};
            auto t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) sink += f.count(sample(i));
            const double dt_hit = std::chrono::duration<double>(clk::now() - t0).count();
            t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) sink += f.count(xnext()); // absent keys
            const double dt_miss = std::chrono::duration<double>(clk::now() - t0).count();
            t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) f.add(sample(i)); // counter bump on existing
            const double dt_upd = std::chrono::duration<double>(clk::now() - t0).count();
            t0 = clk::now();
            for(uint64_t i = 0; i < nq; ++i) f.dec(sample(i)); // undo the bumps
            const double dt_dec = std::chrono::duration<double>(clk::now() - t0).count();
            // insert-fresh: refill from current load, capped (per-op cost grows
            // with the cluster tail; the cap keeps the sweep ~minutes)
            const uint64_t n_ins = std::min<uint64_t>(ceiling - target, 50000);
            t0 = clk::now();
            for(uint64_t i = 0; i < n_ins; ++i) f.add(hred(f, xnext()));
            const double dt_ins = std::chrono::duration<double>(clk::now() - t0).count();
            std::fprintf(stderr, "bench q=%u fill=%.2f: query-hit %.2f M/s, query-miss %.2f M/s, update %.2f M/s, dec %.2f M/s, insert %.2f M/s (%llu)\n",
                         qbits, fill, nq / dt_hit / 1e6, nq / dt_miss / 1e6, nq / dt_upd / 1e6,
                         nq / dt_dec / 1e6, n_ins / dt_ins / 1e6, (unsigned long long)n_ins);
            f.structure_check();
        }
    }
}

// ---- unordered_map A/B (PI request): exact-mode CQF at q=16 vs
// std::unordered_map<uint64_t, V, WangHash> (the repo's own hasher, so
// both sides see the same key distribution; CQF addh uses WangHash too).
// In-binary interleaved A/B (both orders, means reported). N = 2^16
// distinct keys, sequential and shuffled insert order. NOTE: at q=16 the
// CQF ceiling is ~0.95*nslots = 62260, so the N=65536 stream triggers
// exactly one mid-stream grow to q=17 -- that is part of the measured
// insert stream and is noted in the table.
static long peak_rss_bytes() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss; // bytes on macOS, KB on Linux
}
template<typename CQF, typename V>
void bench_map_ab(const char *tag) {
    const uint32_t qbits = 16;
    const uint64_t N = uint64_t(1) << 16;
    std::vector<uint64_t> seq(N), shuf(N);
    std::iota(seq.begin(), seq.end(), uint64_t(0));
    shuf = seq;
    std::mt19937_64 rng(42);
    std::shuffle(shuf.begin(), shuf.end(), rng);
    auto miss_key = [](uint64_t k) {return ~k ^ 0xdeadbeefcafeULL;}; // absent keys
    std::fprintf(stderr, "== %s (q=%u, N=%llu) vs unordered_map<uint64_t,%zu-byte vals, WangHash> ==\n",
                 tag, qbits, (unsigned long long)N, sizeof(V));
    for(const bool shuffled: {false, true}) {
        const std::vector<uint64_t> &keys = shuffled ? shuf: seq;
        // interleaved orders: map-then-CQF and CQF-then-map, mean reported
        auto run_map_insert = [&]() {
            std::unordered_map<uint64_t, V, sketch::hash::WangHash> m;
            m.reserve(N);
            auto t0 = clk::now();
            for(uint64_t i = 0; i < N; ++i) ++m[keys[i]];
            return std::chrono::duration<double>(clk::now() - t0).count();
        };
        auto run_cqf_insert = [&]() {
            CQF f(qbits);
            auto t0 = clk::now();
            for(uint64_t i = 0; i < N; ++i) f.addh(keys[i]);
            return std::chrono::duration<double>(clk::now() - t0).count();
        };
        double ab_m, ab_c, ba_m, ba_c;
        {ab_m = run_map_insert(); ab_c = run_cqf_insert();}
        {ba_c = run_cqf_insert(); ba_m = run_map_insert();}
        const double m_i = 0.5 * (ab_m + ba_m);
        const double c_i = 0.5 * (ab_c + ba_c);
        // build once each for query/update phases (interleaved orders)
        std::unordered_map<uint64_t, V, sketch::hash::WangHash> m;
        m.reserve(N);
        CQF f(qbits);
        for(uint64_t i = 0; i < N; ++i) {++m[keys[i]]; f.addh(keys[i]);}
        auto t0 = clk::now();
        volatile uint64_t sink = 0;
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += m.count(keys[i]);
        double m_hit = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += m.count(miss_key(keys[i]));
        double m_miss = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += uint64_t(++m[keys[i]]);
        double m_upd = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += f.count(f.hash(keys[i]));
        double c_hit = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += f.count(f.hash(miss_key(keys[i])));
        double c_miss = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) {f.addh(keys[i]); sink += 1;}
        double c_upd = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        const double mrat = m_i / c_i;
        std::fprintf(stderr, "%8s insert: map %.3fs (%.2f M/s)  cqf %.3fs (%.2f M/s)  map/cqf %.2fx\n",
                     shuffled ? "shuffled": "sequential", m_i, N / m_i / 1e6, c_i, N / c_i / 1e6, mrat);
        std::fprintf(stderr, "%8s query-hit:  map %.0f ns/op  cqf %.0f ns/op (%.2fx)\n", shuffled ? "shuffled": "sequential",
                     m_hit / N * 1e9, c_hit / N * 1e9, m_hit / c_hit);
        std::fprintf(stderr, "%8s query-miss: map %.0f ns/op  cqf %.0f ns/op (%.2fx)\n", shuffled ? "shuffled": "sequential",
                     m_miss / N * 1e9, c_miss / N * 1e9, m_miss / c_miss);
        std::fprintf(stderr, "%8s update:     map %.0f ns/op  cqf %.0f ns/op (%.2fx)\n", shuffled ? "shuffled": "sequential",
                     m_upd / N * 1e9, c_upd / N * 1e9, m_upd / c_upd);
    }
    // memory: computed footprints. (ru_maxrss is a monotonic per-process
    // max -- by this point earlier phases have already peaked it, so a
    // delta here measured nothing; the formula below is the honest number.)
    // libstdc++ _Hash_node with cached hash code: next ptr (8) + hash (8)
    // + key (8) + val, 8-aligned; bucket array ~N pointers at load ~1.
    const size_t node_bytes = (24 + sizeof(V) + 7) / 8 * 8;
    const double map_bytes = double(N) * (node_bytes + 8);
    CQF f(qbits);
    const auto mem = f.est_memory_usage();
    std::fprintf(stderr, "%s memory: map computed %.2f MB (%zu B/node + 8 B/bucket, %.1f B/entry); "
                 "cqf computed %.2f MB (%.1f B/slot x %llu physical slots, r=%u+%zu val bits) -- cqf/map %.2f\n",
                 tag, map_bytes / 1048576., node_bytes, map_bytes / N,
                 mem.second / 1048576., mem.second / double(2 * (uint64_t(1) << qbits)), 2 * (unsigned long long)(uint64_t(1) << qbits),
                 64 - qbits, sizeof(V) == 8 ? 64 : 16, mem.second / map_bytes);
    // fair same-used insert: CQF filled only to 0.5 / 0.9 of ceiling (no grow)
    for(const double fill: {0.5, 0.9}) {
        const uint64_t ceiling = (uint64_t(1) << qbits) - std::max<uint64_t>(1, (uint64_t(1) << qbits) / 20);
        const uint64_t target = uint64_t(fill * ceiling);
        double mc = 0, cc = 0;
        {auto t0 = clk::now(); {std::unordered_map<uint64_t, V, sketch::hash::WangHash> m; for(uint64_t i = 0; i < target; ++i) ++m[shuf[i]];} mc = std::chrono::duration<double>(clk::now() - t0).count();}
        {auto t0 = clk::now(); {CQF g(qbits); for(uint64_t i = 0; i < target; ++i) g.addh(shuf[i]);} cc = std::chrono::duration<double>(clk::now() - t0).count();}
        std::fprintf(stderr, "%s insert fill=%.1f (used %llu, no grow): map %.0f ns/op, cqf %.0f ns/op (%.2fx)\n",
                     tag, fill, (unsigned long long)target, mc / target * 1e9, cc / target * 1e9, mc / cc);
    }
    // fill sensitivity: CQF at used = 0.5 / 0.9 of ceiling (no grow), map at N
    for(const double fill: {0.5, 0.9}) {
        CQF g(qbits);
        const uint64_t ceiling = (uint64_t(1) << qbits) - std::max<uint64_t>(1, (uint64_t(1) << qbits) / 20);
        const uint64_t target = uint64_t(fill * ceiling);
        for(uint64_t i = 0; i < target; ++i) g.addh(shuf[i]);
        volatile uint64_t sink = 0;
        auto t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += g.count(g.hash(shuf[i % target]));
        double hit = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        t0 = clk::now();
        for(int rep = 0; rep < 4; ++rep) for(uint64_t i = 0; i < N; ++i) sink += g.count(g.hash(miss_key(shuf[i % target])));
        double miss = std::chrono::duration<double>(clk::now() - t0).count() / 4;
        std::fprintf(stderr, "%s fill=%.1f (used %llu): query-hit %.0f ns/op, query-miss %.0f ns/op\n",
                     tag, fill, (unsigned long long)target, hit / N * 1e9, miss / N * 1e9);
    }
}
void test_insert_sorted_bulk() {
    // (a) fresh bulk-load of distinct sorted hashes matches a reference map
    {
        cqf::cqf64_16 f(8);
        std::mt19937_64 rng(7);
        std::vector<uint64_t> hvs(20000);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(auto &v: hvs) {v = rng(); ++ref[v];}
        std::sort(hvs.begin(), hvs.end());
        f.insert_sorted(hvs.begin(), hvs.end());
        CHECK(f.size() == ref.size(), "bulk size %llu != %zu", (unsigned long long)f.size(), ref.size());
        full_verify(f, ref, "bulk-fresh");
        f.structure_check();
        // re-bulk the same keys: pure counter bumps, no growth
        f.insert_sorted(hvs.begin(), hvs.end());
        for(const auto &kv: ref) CHECK(f.count(kv.first) == 2, "re-bulk count != 2");
        f.structure_check();
    }
    // (b) pre-filled table, bulk-append of the sorted upper half
    {
        cqf::cqf64_16 f(8);
        std::mt19937_64 rng(9);
        std::vector<uint64_t> hvs(20000);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(auto &v: hvs) {v = rng(); ++ref[v];}
        std::sort(hvs.begin(), hvs.end());
        for(size_t i = 0; i < 5000; ++i) {f.add(hvs[i]);} // singles pre-fill (lower half)
        f.insert_sorted(hvs.begin() + 5000, hvs.end());
        full_verify(f, ref, "bulk-append");
        f.structure_check();
    }
    // (c) mid-batch grow: bulk-load past the ceiling; batch resumes sorted
    {
        cqf::cqf64_16 f(4); // 16 slots, ceiling 14
        std::vector<std::pair<uint64_t, uint64_t>> batch;
        std::mt19937_64 rng(11);
        for(int i = 0; i < 100; ++i) batch.emplace_back(rng(), 1);
        std::sort(batch.begin(), batch.end());
        f.insert_sorted_counts(batch.begin(), batch.end());
        CHECK(f.q() > 4, "mid-batch grow did not bump q");
        std::unordered_map<uint64_t, uint64_t> ref;
        for(const auto &kv: batch) ++ref[kv.first];
        full_verify(f, ref, "bulk-grow");
        f.structure_check();
    }
    // (d) buffered random-order stream through the aggregate-and-sort wrapper
    {
        cqf::cqf64_16 f(8);
        std::mt19937_64 rng(13);
        std::vector<uint64_t> vals(40000);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(auto &v: vals) {v = rng() % 15000; ++ref[f.hash(v) & ~0ULL] += 0 ? 0: 1;}
        ref.clear();
        for(auto &v: vals) ++ref[f.hash(v)];
        f.insert_buffered(vals.begin(), vals.end(), 4096);
        CHECK(f.size() == ref.size(), "buffered size %llu != %zu", (unsigned long long)f.size(), ref.size());
        full_verify(f, ref, "buffered");
        f.structure_check();
    }
    // (e) buffered-mix stress: random fraction of ops through insert_buffered
    for(const uint32_t seed: {1u, 7u, 99u}) {
        cqf::cqf64_16 f(8);
        std::mt19937_64 rng(seed);
        std::unordered_map<uint64_t, uint64_t> ref;
        std::vector<uint64_t> pending;
        auto flush = [&]() {
            if(pending.empty()) return;
            f.insert_buffered(pending.begin(), pending.end(), 512);
            pending.clear();
        };
        for(unsigned op = 0; op < 20000; ++op) {
            const uint64_t v = rng() % 3000;
            const uint64_t hv = f.hash(v);
            if((rng() & 3) == 0 && ref[hv]) {
                flush();
                CHECK(f.dec(hv), "dec of present failed (buffered mix)");
                if(--ref[hv] == 0) ref.erase(hv);
            } else if((rng() & 3) == 0) {
                pending.push_back(v); // buffered add
                ++ref[hv];
            } else {
                flush();
                f.addh(v);
                ++ref[hv];
            }
            if((op & 2047) == 2047) { flush(); full_verify(f, ref, "buffered-mix"); f.structure_check(); }
        }
        flush();
        full_verify(f, ref, "buffered-mix-final");
        f.structure_check();
    }
    std::fprintf(stderr, "cqf insert_sorted/buffered gates done\n");
}

int main(int argc, char **argv) {
    if(argc > 1 && std::string(argv[1]) == "mapab") { // map A/B only (quick mode)
        std::fprintf(stderr, "unordered_map A/B (exact-mode q=16)\n");
        bench_map_ab<cqf::cqf64_16, uint16_t>("cqf64_16");
        bench_map_ab<cqf::cqf64_64, uint64_t>("cqf64_64");
        return 0;
    }
    const bool bench_only = argc > 1 && std::string(argv[1]) == "bench";
    if(!bench_only) {
        test_map_api();
        test_serialization_saturation_edges();
        test_grow_and_merge();
        test_fill_levels(0.35);
        test_fill_levels(0.70);
        if(failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
        std::fprintf(stderr, "all CQF gates passed; running benchmarks\n");
    }
    {
        cqf::cqf64_64 f(16);
        std::mt19937_64 rng(1);
        while(f.load_factor() < 0.5) f.add(hred(f, rng()));
        bench_popcount_ab(f);
    }
    bench_per_op();
    std::fprintf(stderr, "unordered_map A/B (exact-mode q=16)\n");
    bench_map_ab<cqf::cqf64_16, uint16_t>("cqf64_16");
    bench_map_ab<cqf::cqf64_64, uint64_t>("cqf64_64");
    std::fprintf(stderr, "all CQF tests passed\n");
    return 0;
}
