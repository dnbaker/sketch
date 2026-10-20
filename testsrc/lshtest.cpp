// lshtest: correctness gates for the round-14 LSH table variants
// (sketch::lsh::BandLSHIndex in include/sketch/ssi.h).
//
// Gates:
//  1. Band geometry: contiguous vs overlapping stride, derived band counts,
//     multi-table key counts, multiprobe key inflation.
//  2. Deterministic signature-level known-neighbor recovery for every variant
//     (contiguous, overlapping, composite tiers, multi-table, vote AND/OR,
//     multiprobe): exact per-band hit counts on hand-built signatures.
//  3. Set-level smoke: pmh2 signatures over planted-Jaccard sets; high-J
//     neighbor recovered by OR variants, ranked sanely.
//  4. maxcand early-stop bound respected.

#include "common.h"
#include "ssi.h"
#include "bmh.h"
#include "hnsw.hpp"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <numeric>
#include <string>

using namespace sketch;
using namespace sketch::lsh;
using wmh::pmh2_t;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

struct Sig {
    std::vector<uint64_t> v;
    size_t size() const {return v.size();}
    uint64_t operator[](size_t i) const {return v[i];}
};

static Sig make_sig(size_t m, uint64_t seed) {
    Sig s;
    s.v.resize(m);
    uint64_t x = seed * 0x9E3779B97F4A7C15ULL + 1;
    for(size_t i = 0; i < m; ++i) {
        x = hash::WangHash::hash(x);
        s.v[i] = x;
    }
    return s;
}

static uint32_t count_of(const std::vector<std::pair<uint32_t, uint32_t>> &res, uint32_t id) {
    for(const auto &p: res) if(p.first == id) return p.second;
    return 0;
}

// Gate 1: geometry + key counts.
static void test_geometry() {
    const size_t m = 64;
    // contiguous r=4: 16 bands; overlap stride 2: 31 bands; r=8: 8 bands.
    BandLSHIndex<> idx(m, {{4, 0, 0, 0}, {4, 2, 0, 0}, {8, 0, 0, 0}}, /*ntables=*/3);
    CHECK(idx.nbands(0) == 16, "contiguous r=4 over m=64 should give 16 bands, got %zu", idx.nbands(0));
    CHECK(idx.nbands(1) == 31, "stride-2 r=4 over m=64 should give 31 bands, got %zu", idx.nbands(1));
    CHECK(idx.nbands(2) == 8, "r=8 over m=64 should give 8 bands, got %zu", idx.nbands(2));
    CHECK(idx.n_bands() == (16 + 31 + 8) * 3, "n_bands %llu", (unsigned long long)idx.n_bands());
    idx.insert(make_sig(m, 1), 0);
    CHECK(idx.n_keys() == (16 + 31 + 8) * 3, "n_keys %llu", (unsigned long long)idx.n_keys());
    // multiprobe: budget 4 -> 5 keys per band.
    BandLSHIndex<> mp(m, {{4, 0, 0, 4}}, 1);
    mp.insert(make_sig(m, 2), 7);
    CHECK(mp.n_keys() == 16 * 5, "multiprobe keys %llu", (unsigned long long)mp.n_keys());
    CHECK(mp.n_buckets() <= mp.n_keys(), "buckets <= keys");
    // vote threshold normalization, bad-arg throws
    bool threw = false;
    try { BandLSHIndex<> bad(m, {{4, 5, 0, 0}}); } catch(const std::invalid_argument &) { threw = true; }
    CHECK(threw, "stride >= band_size (and nonzero) must throw");
    threw = false;
    try { BandLSHIndex<> bad(m, {{65, 0, 0, 0}}); } catch(const std::invalid_argument &) { threw = true; }
    CHECK(threw, "band_size > siglen must throw");
}

// Gate 2: deterministic known-neighbor recovery per variant.
static void test_known_neighbor() {
    const size_t m = 64;
    Sig a = make_sig(m, 42);
    // b: identical to a except band [4, 8) (r=4) fully replaced.
    Sig b = a;
    for(size_t i = 4; i < 8; ++i) b.v[i] = a.v[i] ^ 0xDEADBEEFCAFEF00DULL;
    // c: differs from a in exactly ONE register, position 6 (band [4,8), offset 2).
    Sig c = a;
    c.v[6] = a.v[6] ^ 0x0123456789ABCDEFULL;
    // d: far away (independent signature).
    Sig d = make_sig(m, 1337);

    auto build = [&](std::vector<BandTier> tiers, size_t L, size_t vote) {
        BandLSHIndex<> idx(m, tiers, L, vote);
        idx.insert(a, 100);
        idx.insert(b, 101);
        idx.insert(c, 102);
        idx.insert(d, 103);
        return idx;
    };
    {   // contiguous OR: b shares 15/16 bands, c shares 15/16, d shares ~0.
        auto idx = build({{4, 0, 0, 0}}, 1, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 15, "b should hit 15 bands, got %u", count_of(res, 101));
        CHECK(count_of(res, 102) == 15, "c should hit 15 bands, got %u", count_of(res, 102));
        CHECK(count_of(res, 103) <= 2, "far id d hit %u bands", count_of(res, 103));
        CHECK(!res.empty() && res[0].second >= 15, "top candidate should have >= 15 hits");
        // maxcand bound
        auto res2 = idx.query(a, 2);
        CHECK(res2.size() <= 2, "maxcand=2 respected, got %zu", res2.size());
    }
    {   // overlapping OR (stride 2, 31 bands): b changed regs 4..7 -> bands starting at 2,4,6
        // (plus start 2... covering [4,8) region) miss; windows covering any of 4..7: starts 2,3? starts
        // j*2 with [2j,2j+4) meeting {4,5,6,7}: j in {1,2,3} -> 3 misses -> 28. c changed only reg 6,
        // covered by windows starting at 4 and 6 -> 2 misses -> 29.
        auto idx = build({{4, 2, 0, 0}}, 1, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 28, "b should hit 28 overlapping bands, got %u", count_of(res, 101));
        CHECK(count_of(res, 102) == 29, "c should hit 29 overlapping bands, got %u", count_of(res, 102));
    }
    {   // multi-table: 2 tables -> doubled hits.
        auto idx = build({{4, 0, 0, 0}}, 2, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 30, "b should hit 30 bands over 2 tables, got %u", count_of(res, 101));
        CHECK(count_of(res, 103) <= 4, "far id d hit %u bands over 2 tables", count_of(res, 103));
    }
    {   // composite tiered {8, 4}: query walks r=8 first; b differs in [4,8) so 1 of 8 r=8 bands misses;
        // c too. r=4 tier adds 15 each. Total 7 + 15 = 22.
        auto idx = build({{4, 0, 0, 0}, {8, 0, 0, 0}}, 1, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 22, "b should hit 7+15=22 tiered bands, got %u", count_of(res, 101));
        CHECK(count_of(res, 102) == 22, "c should hit 7+15=22 tiered bands, got %u", count_of(res, 102));
    }
    {   // AND composition (vote = nbands = 16): b misses a band -> excluded; only a itself (16) qualifies.
        auto idx = build({{4, 0, 0, 0}}, 1, 16);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 0, "AND: b must not pass, got %u", count_of(res, 101));
        CHECK(count_of(res, 100) == 16, "AND: self must pass all 16, got %u", count_of(res, 100));
    }
    {   // vote = 15: b/c pass (15 hits), d does not.
        auto idx = build({{4, 0, 0, 0}}, 1, 15);
        auto res = idx.query(a);
        CHECK(count_of(res, 101) == 15, "vote15: b count %u", count_of(res, 101));
        CHECK(count_of(res, 103) == 0, "vote15: d count %u", count_of(res, 103));
    }
    {   // multiprobe budget 4: c differs at band offset 2 -> probe 3 matches -> all 16 bands.
        auto idx = build({{4, 0, 0, 4}}, 1, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 102) == 16, "multiprobe: c should hit all 16 bands, got %u", count_of(res, 102));
        CHECK(count_of(res, 101) == 15, "multiprobe cannot help b (all 4 band registers differ), got %u", count_of(res, 101));
    }
    {   // multiprobe budget 1: only offset-0 drops; c differs at offset 2 -> no probe help -> 15.
        auto idx = build({{4, 0, 0, 1}}, 1, 1);
        auto res = idx.query(a);
        CHECK(count_of(res, 102) == 15, "mp budget 1: c count %u", count_of(res, 102));
    }
    {   // independent multi-table routing: a = (X, Y), b = (X, Z) matches a in
        // table 0 only, c = (W, Y) matches a in table 1 only. query_mt must find
        // both b and c (one hit each); a broadcast query finds only b.
        const size_t L = 2;
        std::vector<uint64_t> X, Y, Z, W;
        for(size_t i = 0; i < m; ++i) {
            X.push_back(hash::WangHash::hash(0x1000 + i));
            Y.push_back(hash::WangHash::hash(0x2000 + i));
            Z.push_back(hash::WangHash::hash(0x3000 + i));
            W.push_back(hash::WangHash::hash(0x4000 + i));
        }
        BandLSHIndex<> idx(m, {{4, 0, 0, 0}}, L, 1);
        idx.insert_mt(std::vector<Sig>{{X}, {Y}}, 200); // a
        idx.insert_mt(std::vector<Sig>{{X}, {Z}}, 201); // b
        idx.insert_mt(std::vector<Sig>{{W}, {Y}}, 202); // c
        auto mt = idx.query_mt(std::vector<Sig>{{X}, {Y}});
        CHECK(count_of(mt, 201) == 16, "mt: b (table-0 match) count %u", count_of(mt, 201));
        CHECK(count_of(mt, 202) == 16, "mt: c (table-1 match) count %u", count_of(mt, 202));
        CHECK(count_of(mt, 200) == 32, "mt: self count %u", count_of(mt, 200));
        auto bc = idx.query(Sig{X});
        CHECK(count_of(bc, 201) == 16, "broadcast query: b count %u", count_of(bc, 201));
        CHECK(count_of(bc, 202) == 0, "broadcast query must miss c, got %u", count_of(bc, 202));
    }
}

// Gate 3: set-level smoke with pmh2 signatures over planted-Jaccard sets.
static void test_set_level() {
    const size_t m = 128;
    const double J = 0.9;
    const uint64_t C = 4000, D = uint64_t(C * (1. - J) / (2. * J) + .5);
    const uint64_t salt = 0xC0FFEE;
    pmh2_t<> a(m), b(m);
    for(uint64_t v = 0; v < C; ++v) {a.addh(v ^ salt); b.addh(v ^ salt);}
    for(uint64_t v = 0; v < D; ++v) a.addh((v + (uint64_t(1) << 40)) ^ salt);
    for(uint64_t v = 0; v < D; ++v) b.addh((v + (uint64_t(1) << 41)) ^ salt);
    BandLSHIndex<> idx(m, {{4, 0, 0, 0}, {8, 0, 0, 0}}, 1, 1);
    idx.insert(a.ids(), 0);
    idx.insert(b.ids(), 1);
    auto res = idx.query(a.ids());
    bool found = false;
    for(const auto &p: res) if(p.first == 1) {found = true; CHECK(p.second >= 16, "pmh2 neighbor hits %u", p.second);}
    CHECK(found, "pmh2 set-level: planted J=0.9 neighbor must be retrieved");
    if(found) {
        const double est = a.jaccard_index(b);
        CHECK(std::abs(est - J) < 0.1, "pmh2 est J %g vs true %g", est, J);
    }
}

// Gate 4: tiered lazy-stop probing.
static void test_tiered_probe() {
    const size_t m = 32;
    TieredProbeIndex idx(m, {8, 4, 2});
    CHECK(idx.n_keys() == 0, "empty index keys %llu", (unsigned long long)idx.n_keys());
    Sig a = make_sig(m, 7);
    Sig b = a; for(size_t i = 0; i < 8; ++i)  b.v[i] = a.v[i] ^ 0xDEADBEEFCAFEF00DULL;  // misses r=8 band 0
    Sig c = a; for(size_t i = 8; i < 16; ++i) c.v[i] = a.v[i] ^ 0xDEADBEEFCAFEF00DULL; // misses r=8 band 1
    Sig d = make_sig(m, 99);                                                            // far
    idx.insert(a, 0); idx.insert(b, 1); idx.insert(c, 2); idx.insert(d, 3);
    CHECK(idx.n_keys() == (4 + 8 + 16) * 4, "tiered keys %llu", (unsigned long long)idx.n_keys());
    TieredProbeIndex::ProbeStats st;
    {   // tight stop: tier-0 band 0 (regs 0-8) matches a and c (b misses it) -> 2 candidates, 1 band
        auto r = idx.query(a, 1, 1.0, &st);
        CHECK(r.size() == 2, "stop@1 got %zu cands", r.size());
        CHECK(st.bands_probed == 1 && st.tier_reached == 0, "stop@1 bands %zu tier %zu", st.bands_probed, st.tier_reached);
    }
    {   // tier-0 band 1 (regs 8-16) adds b -> stop at 3 candidates, 2 bands
        auto r = idx.query(a, 1, 3.0, &st);
        CHECK(r.size() == 3, "stop@3 got %zu cands", r.size());
        CHECK(st.bands_probed == 2 && st.tier_reached == 0, "stop@3 bands %zu tier %zu", st.bands_probed, st.tier_reached);
    }
    {   // loose stop exhausts all tiers: candidates {a, b, c}, all 28 bands probed
        auto r = idx.query(a, 10, 3.0, &st);
        CHECK(r.size() == 3, "exhaustive got %zu cands (d must not appear)", r.size());
        CHECK(st.bands_probed == 28 && st.tier_reached == 2, "exhaustive bands %zu tier %zu", st.bands_probed, st.tier_reached);
    }
    bool threw = false;
    try { TieredProbeIndex bad(m, {4, 8}); } catch(const std::invalid_argument &) { threw = true; }
    CHECK(threw, "widths must be given most-specific first");
}

// Gate 5: HNSW over pmh2 register signatures with agreement distance.
// Corpus mirrors the v2 workload: zipf sets, anchors with 11 planted
// near-copies (exact J ~ 0.85) plus zipf-background sets, pmh2 m=64
// signatures. (iid-register synthetic corpora are a hard case for graph
// indexes — near-plateau distances — and are not what this index is for.)
static void test_hnsw() {
    const size_t m = 64, SETSIZE = 400, UNIVERSE = size_t(1) << 17;
    const size_t ngroup = 60, per_group = 12, nbg = 1200;
    DefaultRNGType rng(0x5EED);
    auto sample = [&](size_t n, auto &out) { // n distinct ids, sorted
        ankerl::unordered_dense::set<uint32_t> s;
        while(s.size() < n) s.insert(uint32_t(rng() % UNIVERSE));
        out.assign(s.begin(), s.end());
        std::sort(out.begin(), out.end());
    };
    std::vector<std::vector<uint32_t>> sets;
    std::vector<uint32_t> anchor_id;
    const size_t c = size_t(std::lround(2. * SETSIZE * 0.85 / 1.85)); // shared, exact J = 0.85
    for(size_t g = 0; g < ngroup; ++g) {
        std::vector<uint32_t> base;
        sample(SETSIZE, base);
        anchor_id.push_back(sets.size());
        sets.push_back(base);
        for(size_t cp = 0; cp < per_group - 1; ++cp) {
            std::shuffle(base.begin(), base.end(), rng);
            std::vector<uint32_t> b(base.begin(), base.begin() + c);
            ankerl::unordered_dense::set<uint32_t> avoid(b.begin(), b.end());
            std::vector<uint32_t> fresh;
            while(fresh.size() < SETSIZE - c) {
                const uint32_t v = uint32_t(rng() % UNIVERSE);
                if(!avoid.count(v)) {avoid.insert(v); fresh.push_back(v);}
            }
            b.insert(b.end(), fresh.begin(), fresh.end());
            std::sort(b.begin(), b.end());
            sets.push_back(std::move(b));
        }
    }
    for(size_t i = 0; i < nbg; ++i) {
        std::vector<uint32_t> v;
        sample(SETSIZE, v);
        sets.push_back(std::move(v));
    }
    std::vector<std::vector<uint64_t>> sigs(sets.size());
    for(size_t i = 0; i < sets.size(); ++i) {
        pmh2_t<> s(m);
        for(const auto e: sets[i]) s.addh(e);
        sigs[i] = s.ids();
    }
    auto dist = [m](const std::vector<uint64_t> &x, const std::vector<uint64_t> &y) {
        size_t eq = 0;
        for(size_t i = 0; i < m; ++i) eq += x[i] == y[i];
        return 1. - double(eq) / double(m);
    };
    HNSW<std::vector<uint64_t>, decltype(dist)> g1(16, 200, 42, dist), g2(16, 200, 42, dist);
    for(size_t i = 0; i < sigs.size(); ++i) g1.add(sigs[i]);
    for(size_t i = 0; i < sigs.size(); ++i) g2.add(sigs[i]);
    CHECK(g1.size() == sigs.size(), "hnsw size %zu", g1.size());
    // determinism: identical seed + insertion order -> identical 5-NN
    for(const size_t qi: {size_t(0), size_t(17), size_t(600)}) {
        auto r1 = g1.search(sigs[qi], 5, 100);
        auto r2 = g2.search(sigs[qi], 5, 100);
        CHECK(r1.size() == r2.size() && std::equal(r1.begin(), r1.end(), r2.begin()), "hnsw deterministic q%zu", qi);
    }
    // self-recovery: every item appears in its own 10-NN (top-1 exactness has
    // a known tail on hard items; 10-NN membership catches disconnected nodes)
    for(size_t qi = 0; qi < sigs.size(); qi += 27) {
        auto r = g1.search(sigs[qi], 10, 200);
        bool found = false;
        for(const auto &p: r) found |= p.second == qi;
        CHECK(found, "hnsw self 10-nn missing %zu", qi);
    }
    // agreement-recall vs brute force: the group's planted copies fill the brute top-10
    auto brute_top = [&](size_t qi) {
        std::vector<std::pair<double, uint32_t>> v(sigs.size());
        for(size_t i = 0; i < sigs.size(); ++i) v[i] = {dist(sigs[qi], sigs[i]), uint32_t(i)};
        std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first < b.first: a.second < b.second;
        });
        v.resize(10);
        std::vector<uint32_t> ids; for(auto &p: v) ids.push_back(p.second);
        std::sort(ids.begin(), ids.end());
        return ids;
    };
    double overlap = 0; int nq = 0;
    for(size_t g = 0; g < ngroup; g += 3) {
        for(size_t cp = 0; cp < per_group; cp += 4) {
            const size_t qi = g * per_group + cp;
            auto gt = brute_top(qi);
            auto r = g1.search(sigs[qi], 10, 200);
            std::vector<uint32_t> got; for(auto &p: r) got.push_back(p.second);
            std::sort(got.begin(), got.end());
            std::vector<uint32_t> inter;
            std::set_intersection(gt.begin(), gt.end(), got.begin(), got.end(), std::back_inserter(inter));
            overlap += double(inter.size()) / 10.; ++nq;
        }
    }
    CHECK(overlap / nq >= 0.9, "hnsw agreement recall %.3f (n=%d)", overlap / nq, nq);
}

int main() {
    test_geometry();
    test_known_neighbor();
    test_set_level();
    test_tiered_probe();
    test_hnsw();
    if(failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::fprintf(stderr, "All tests passed\n");
    return 0;
}
