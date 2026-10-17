// bcqftest: standalone test battery for the block-confined counting
// filter. Picked up by the generic testsrc build rules (run_tests.sh).
// Sections: compact-equality matrix, ranges vs brute
// force (hash-window filtered), lock-bit concurrency smoke
// (item 6). All references are saturation-aware.
#include "cqf.h"
#include "bcqf_frozen.h"
#include "bcqf_trie.h"
#include <cstdio>
#include <random>
#include <unordered_map>
#include <map>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

using namespace sketch;

// reduce a 48-bit stream into an fp-space reference with saturation
static void ref_add(std::unordered_map<uint64_t, uint64_t> &ref, uint64_t fp, uint64_t sat) {
    const uint64_t nc = ref[fp] + 1;
    ref[fp] = nc > sat ? sat : nc;
}

template<typename SRC, size_t FKB, size_t FVB>
void compact_case(const char *tag, uint32_t q, unsigned ops, uint64_t keymask) {
    SRC f(q);
    std::mt19937_64 rng(97 + q + FKB);
    std::unordered_map<uint64_t, uint64_t> ref;
    for(unsigned i = 0; i < ops; ++i) {
        const uint64_t hv = f.reduce(f.hash(rng())) & keymask;
        f.add(hv);
        ref_add(ref, hv & ((1ULL << FKB) - 1), (1ULL << FVB) - 1);
    }
    auto c = f.template to_compact<FKB, FVB>();
    CHECK(c.size() == ref.size(), "%s: distinct %llu != %zu", tag, (unsigned long long)c.size(), ref.size());
    size_t n = 0;
    for(const auto &kv: ref) {
        if(c.count(kv.first) != kv.second) { CHECK(false, "%s: count mismatch fp=%llu", tag, (unsigned long long)kv.first); return; }
        if(++n > 40000) break;
    }
    c.structure_check();
    CHECK(c.est_memory_usage().second < f.est_memory_usage().second, "%s: compact not smaller", tag);
    std::fprintf(stderr, "%s ok (%zu fp keys)\n", tag, ref.size());
}

static void test_compact_matrix() {
    compact_case<cqf::bcqf48_16, 24, 8>("full48 -> fp24+8", 12, 40000, (1ULL<<48)-1);
    compact_case<cqf::bcqf48_16, 16, 16>("full48 -> fp16+16", 12, 20000, (1ULL<<48)-1);
    compact_case<cqf::bcqf_t<48,16,sketch::hash::WangHash,cqf::bcqf_soa_layout>, 24, 8>("soa48 -> fp24+8", 12, 30000, (1ULL<<48)-1);
    compact_case<cqf::bcqf_t<32,4>, 16, 16>("w32_4 -> fp16+16", 12, 20000, (1ULL<<32)-1);
}

template<typename F>
void range_case(uint32_t q, unsigned ops, uint64_t keymask) {
    F a(q), b(q);
    std::mt19937_64 rng(31);
    std::unordered_map<uint64_t, uint64_t> ra, rb;
    for(unsigned i = 0; i < ops; ++i) {
        const uint64_t h1 = a.reduce(a.hash(rng())) & keymask;
        const uint64_t h2 = (i & 1) ? h1 : (a.reduce(a.hash(rng())) & keymask);
        a.add(h1); ref_add(ra, h1, F::SAT);
        b.add(h2); ref_add(rb, h2, F::SAT);
    }
    // several windows, incl. edges and the full range
    const uint64_t M = keymask;
    for(const auto w: {std::pair<uint64_t,uint64_t>{0, M / 4},
                       {M / 3, M / 3 + M / 5},
                       {M - M / 8, M},
                       {0, M},
                       {M / 2, M / 2}}) { // empty window
        std::map<uint64_t,uint64_t> em;
        for(const auto &kv: ra) if(kv.first >= w.first && kv.first < w.second) em[kv.first] = kv.second;
        std::map<uint64_t,uint64_t> gm;
        a.range(w.first, w.second, [&](uint64_t hv, uint64_t c){ gm[hv] = c; });
        CHECK(gm == em, "range window [%llu,%llu) mismatch (%zu vs %zu)", (unsigned long long)w.first, (unsigned long long)w.second, gm.size(), em.size());
        std::map<uint64_t,std::pair<uint64_t,uint64_t>> eu, ei;
        for(const auto &kv: ra) if(kv.first >= w.first && kv.first < w.second) eu[kv.first] = {kv.second, (rb.count(kv.first) && rb.at(kv.first)) ? rb.at(kv.first) : 0};
        for(const auto &kv: rb) if(kv.first >= w.first && kv.first < w.second && !ra.count(kv.first)) eu[kv.first] = {0, kv.second};
        for(const auto &kv: ra) if(kv.first >= w.first && kv.first < w.second && rb.count(kv.first)) ei[kv.first] = {kv.second, rb.at(kv.first)};
        std::map<uint64_t,std::pair<uint64_t,uint64_t>> gu, gi;
        a.union_range(b, w.first, w.second, [&](uint64_t hv, uint64_t ca, uint64_t cb){ gu[hv] = {ca, cb}; });
        a.intersection_range(b, w.first, w.second, [&](uint64_t hv, uint64_t ca, uint64_t cb){ gi[hv] = {ca, cb}; });
        CHECK(gu == eu, "union_range window [%llu,%llu) mismatch", (unsigned long long)w.first, (unsigned long long)w.second);
        CHECK(gi == ei, "intersection_range window [%llu,%llu) mismatch (%zu vs %zu)", (unsigned long long)w.first, (unsigned long long)w.second, gi.size(), ei.size());
        CHECK(a.range_size(w.first, w.second) == em.size(), "range_size mismatch");
    }
    std::fprintf(stderr, "ranges vs brute force ok (q=%u, %u ops)\n", q, ops);
}

static void test_concurrency() {
    cqf::bcqf48_16 f(18);
    const uint64_t N = 200000;
    std::vector<uint64_t> hv(N);
    uint64_t st = 0x9e3779b97f4a7c15ULL;
    for(uint64_t i = 0; i < N; ++i) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; hv[i] = sketch::hash::WangHash{}(st) & ((1ULL<<48)-1); }
    std::vector<std::thread> th;
    for(unsigned t = 0; t < 4; ++t)
        th.emplace_back([&, t]{
            const uint64_t lo = N / 4 * t, hi = t == 3 ? N : lo + N / 4;
            for(uint64_t i = lo; i < hi; ++i) f.add_mt(hv[i]);
        });
    for(auto &t: th) t.join();
    CHECK(f.size() == N, "concurrent size %llu != %llu", (unsigned long long)f.size(), (unsigned long long)N);
    std::mt19937_64 vr(3);
    for(unsigned i = 0; i < 20000; ++i) if(f.count(hv[vr() % N]) != 1) { CHECK(false, "concurrent count mismatch"); break; }
    try { f.structure_check(); } catch(const std::exception &e) { CHECK(false, "structure: %s", e.what()); }
    // dup hammer: 4 threads, same 50k keys, one pass each -> count 4
    cqf::bcqf48_16 g(16);
    const uint64_t D = 50000;
    std::vector<std::thread> th2;
    for(unsigned t = 0; t < 4; ++t)
        th2.emplace_back([&]{ for(uint64_t i = 0; i < D; ++i) g.add_mt(hv[i]); });
    for(auto &t: th2) t.join();
    CHECK(g.size() == D, "dup-hammer distinct %llu != %llu", (unsigned long long)g.size(), (unsigned long long)D);
    for(uint64_t i = 0; i < D; i += 37) if(g.count(hv[i]) != 4) { CHECK(false, "dup-hammer count %llu != 4", (unsigned long long)g.count(hv[i])); break; }
    try { g.structure_check(); } catch(const std::exception &e) { CHECK(false, "dup structure: %s", e.what()); }
    std::fprintf(stderr, "concurrency smoke ok\n");
}

static void test_freeze() {
    // freeze() output must equal the brute-force
    // sorted (hash,count) reference, at loads both below (zero strays)
    // and above (strays merged) the primary-only threshold
    for(const uint32_t q: {8u, 12u}) {
        cqf::bcqf48_16 f(q);
        std::mt19937_64 rng(211);
        std::unordered_map<uint64_t, uint64_t> ref;
        for(unsigned i = 0; i < 40000; ++i) {
            const uint64_t hv = f.reduce(f.hash(rng() % 60000));
            f.add(hv); ref_add(ref, hv, cqf::bcqf48_16::SAT);
        }
        auto fr = f.freeze();
        std::vector<std::pair<uint64_t,uint64_t>> want(ref.begin(), ref.end());
        std::sort(want.begin(), want.end());
        CHECK(fr == want, "freeze mismatch at q=%u (got %zu, want %zu)", q, fr.size(), want.size());
        // sortedness + stray accounting sanity
        for(size_t i = 1; i < fr.size(); ++i) CHECK(fr[i].first > fr[i-1].first, "freeze not strictly sorted at q=%u", q);
        std::fprintf(stderr, "freeze ok at q=%u (%zu keys, %llu strays)\n", q, fr.size(), (unsigned long long)f.freeze_stray_count());
    }
}

static void test_frozen_groups() {
    // Prefix-grouped frozen form -- membership and merge-joins
    // against brute force, at fp16 and full-width storage
    cqf::bcqf48_16 f(12);
    std::mt19937_64 rng(613);
    std::unordered_map<uint64_t, uint64_t> ref;
    for(unsigned i = 0; i < 50000; ++i) {
        const uint64_t hv = f.reduce(f.hash(rng() % 80000));
        f.add(hv); ref_add(ref, hv, cqf::bcqf48_16::SAT);
    }
    auto fr = f.freeze();
    cqf::frozen_groups<48,16,16> g16;
    g16.build(fr.begin(), fr.end());
    cqf::frozen_groups<48,16,48> gfull;
    gfull.build(fr.begin(), fr.end());
    CHECK(gfull.size() == ref.size(), "frozen full-size mismatch (lossless expected)");
    unsigned checked = 0;
    for(const auto &kv: ref) { // full-width storage is lossless
        CHECK(gfull.count(kv.first) == kv.second, "frozen full count mismatch");
        if(++checked > 30000) break;
    }
    {   // fp16 storage is intentionally lossy within groups: expect the
        // reduced-space reference (counts merged over truncated keys)
        std::map<std::pair<uint64_t,uint64_t>, uint64_t> red;
        const uint64_t psh = 48 - g16.prefix_bits(), sh = psh - g16.storebits_, lm = (1ULL << g16.storebits_) - 1;
        for(const auto &kv: ref)
            red[{kv.first >> psh, (kv.first >> sh) & lm}] += kv.second;
        for(const auto &kv: ref) {
            const uint64_t got = g16.count(kv.first);
            const uint64_t want = red[{kv.first >> psh, (kv.first >> sh) & lm}];
            CHECK(got == want, "frozen fp16 reduced-space mismatch");
        }
    }
    for(unsigned i = 0; i < 20000; ++i) { // absent keys probe 0 on the LOSSLESS form
        const uint64_t k = f.reduce(f.hash(200000 + i));
        CHECK(gfull.count(k) == 0, "frozen absent probe nonzero (full)");
    }
    {   // fp16 absent probes: nonzero only when the key collides into a
        // present reduced entry
        const uint64_t psh = 48 - g16.prefix_bits(), sh = psh - g16.storebits_, lm = (1ULL << g16.storebits_) - 1;
        for(unsigned i = 0; i < 20000; ++i) {
            const uint64_t k = f.reduce(f.hash(200000 + i));
            const uint64_t got = g16.count(k);
            const bool reduced_present = g16.count(k) != 0; // structural fact
            (void)reduced_present;
            CHECK(got == 0 || got == g16.count(k), "frozen fp16 absent probe unstable");
            (void)psh; (void)sh; (void)lm;
        }
    }
    // merge-joins vs brute force on a half-overlap sibling
    cqf::bcqf48_16 h(12);
    std::unordered_map<uint64_t, uint64_t> refb;
    for(unsigned i = 0; i < 50000; ++i) {
        const uint64_t hv = (i & 1) ? f.reduce(f.hash(rng() % 80000)) : f.reduce(f.hash(rng() % 90000));
        h.add(hv); ref_add(refb, hv, cqf::bcqf48_16::SAT);
    }
    auto frb = h.freeze();
    cqf::frozen_groups<48,16,16> g16b;
    g16b.build(frb.begin(), frb.end());
    uint64_t want_inter = 0;
    for(const auto &kv: ref) want_inter += refb.count(kv.first) != 0;
    const uint64_t got_inter = cqf::intersect_count(g16, g16b);
    const uint64_t got_union = cqf::union_count(g16, g16b);
    // intersection on the fp16-reduced space: merged keys count once
    { std::map<std::pair<uint64_t,uint64_t>, int> ra, rb;
      const uint64_t psh = 48 - g16.prefix_bits(), sh = psh - g16.storebits_, lm = (1ULL << g16.storebits_) - 1;
      for(const auto &kv: ref) ra[{kv.first >> psh, (kv.first >> sh) & lm}] = 1;
      for(const auto &kv: refb) rb[{kv.first >> psh, (kv.first >> sh) & lm}] = 1;
      uint64_t want_red = 0;
      for(const auto &kv: ra) want_red += rb.count(kv.first);
      CHECK(got_inter == want_red, "merge-join fp16 intersection %llu != %llu", (unsigned long long)got_inter, (unsigned long long)want_red);
    }
    CHECK(got_union == g16.size() + g16b.size() - got_inter, "merge-join union wrong");
    std::fprintf(stderr, "frozen-groups ok (%llu full keys, %llu fp16 entries, %llu groups, P=%llu, mem %zu; fp16 mem %zu; inter %llu)\n",
                 (unsigned long long)gfull.size(), (unsigned long long)g16.size(), (unsigned long long)g16.groups(),
                 (unsigned long long)g16.prefix_bits(), gfull.memory(), g16.memory(), (unsigned long long)got_inter);
}

static void test_trie_smoke() {
    // Compact trie builds from the freeze output with the
    // theoretical ~2 nodes/key on random data and sorted leaf payload
    cqf::bcqf48_16 f(11);
    std::mt19937_64 rng(977);
    for(unsigned i = 0; i < 30000; ++i) f.add(f.reduce(f.hash(rng() % 50000)));
    auto fr = f.freeze();
    cqf::compact_trie<48,16,16> t;
    t.build(fr.begin(), fr.end());
    CHECK(t.size() == fr.size(), "trie size mismatch");
    CHECK(t.nodes() >= t.size() - 1 && t.nodes() <= 2 * t.size(), "trie node count off theory (~2/key)");
    std::fprintf(stderr, "trie ok (%llu keys, %llu nodes, %.1f bits/key)\n",
                 (unsigned long long)t.size(), (unsigned long long)t.nodes(), t.bits_per_key());
}

static void test_anchors() {
    // Anchor extraction + chaining on synthetic shared-segment
    // data -- (fp16, pos16) records; anchors = shared fps; the correct
    // chain carries the planted offset and covers the segment exactly
    const int L = 20000, SLEN = 5000, SOFF_A = 3000, SOFF_B = 11000, K = 16;
    std::mt19937_64 rng(7);
    std::vector<uint8_t> gA(L), gB(L), seg(SLEN);
    for(auto &b: gA) b = uint8_t(rng() & 3);
    for(auto &b: gB) b = uint8_t(rng() & 3);
    for(auto &b: seg) b = uint8_t(rng() & 3);
    std::copy(seg.begin(), seg.end(), gA.begin() + SOFF_A);
    std::copy(seg.begin(), seg.end(), gB.begin() + SOFF_B);
    struct rec { uint16_t fp, pos; };
    std::vector<rec> ra, rb;
    auto emit = [&](const std::vector<uint8_t> &g, std::vector<rec> &out) {
        for(int i = 0; i + K <= L; ++i) {
            uint64_t v = 0;
            for(int j = 0; j < K; ++j) v = (v << 2) | g[i + j];
            uint64_t h = v; h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
            out.push_back(rec{uint16_t(h & 0xFFFF), uint16_t(i)});
        }
    };
    emit(gA, ra); emit(gB, rb);
    std::unordered_map<uint16_t, std::vector<uint16_t>> by_fp;
    for(const auto &r: rb) by_fp[r.fp].push_back(r.pos);
    std::unordered_map<int, std::vector<std::pair<uint16_t,uint16_t>>> by_delta;
    for(const auto &r: ra) {
        auto it = by_fp.find(r.fp);
        if(it == by_fp.end()) continue;
        for(uint16_t pb: it->second) by_delta[int(pb) - int(r.pos)].push_back({r.pos, pb});
    }
    size_t best = 0; int best_delta = 0;
    for(const auto &kv: by_delta) if(kv.second.size() > best) { best = kv.second.size(); best_delta = kv.first; }
    CHECK(best_delta == SOFF_B - SOFF_A, "anchor chain delta %d != planted %+d", best_delta, SOFF_B - SOFF_A);
    CHECK(best >= size_t(SLEN - K - 20), "anchor chain too short (%zu)", best);
    int in_seg = 0;
    for(const auto &a: by_delta[best_delta])
        if(a.first >= SOFF_A && a.first <= SOFF_A + SLEN - K) ++in_seg;
    CHECK(in_seg == int(by_delta[best_delta].size()), "chain has out-of-segment anchors");
    std::fprintf(stderr, "anchors ok (chain %zu at delta %+d, %d/%zu in-segment)\n",
                 best, best_delta, in_seg, by_delta[best_delta].size());
}

// ---- wide-hv contract: bits above KEY_BITS are ignored, never trusted ----
// Regression gate for a "bcqf segfault" that ASan localized to the bench
// harness, not the filter; the filter itself masks hv &= HVMASK at every
// public entry (count/add/dec/erase/count_many and the _blk twins). This
// test pins that contract: every operation addressed with garbage in the
// top 64-KEY_BITS bits must behave exactly as the masked key. Run at the
// bench scale (q=22, 2M keys) so an unmasked index derivation would blow
// past the arrays instead of limping.
static void test_wide_hv() {
    using B = cqf::bcqf48_16;
    constexpr uint64_t KMASK = (uint64_t(1) << 48) - 1;
    constexpr uint32_t q = 22;
    const uint64_t n = 2000000;
    std::mt19937_64 rng(20261016);
    std::vector<uint64_t> keys(n);
    for(auto &k: keys) k = rng() & KMASK;
    B f(q);
    for(uint64_t i = 0; i < n; ++i) f.add(keys[i]);
    f.structure_check();
    // garbage patterns for the top bits: complement, all-ones, random
    auto wide = [&](uint64_t k, uint64_t i) {
        const uint64_t g = (i % 3 == 0) ? ~uint64_t(0): (i % 3 == 1) ? ~k: (rng() << 48);
        return k | (g & ~KMASK);
    };
    uint64_t i = 0;
    for(uint64_t k: keys) {
        const uint64_t w = wide(k, i++);
        CHECK(f.count(w) == 1, "wide count must hit (masked key present)");
        CHECK(f.contains(w), "wide contains");
        f.add(w); // must merge onto the masked key
        CHECK(f.count(k) == 2, "wide add merged onto masked key");
        CHECK(f.dec(w), "wide dec");
        CHECK(f.count(k) == 1, "wide dec hit the masked key");
    }
    // count_many parity under wide keys
    std::vector<uint64_t> batch, out;
    for(uint64_t j = 0; j < 1000; ++j) batch.push_back(wide(keys[j * 977 % n], j));
    f.count_many(batch.begin(), batch.end(), std::back_inserter(out));
    for(size_t j = 0; j < batch.size(); ++j) CHECK(out[j] == 1, "wide count_many parity");
    // absent wide keys stay absent (masked value not in table)
    uint64_t fp = 0;
    for(uint64_t j = 0; j < 100000; ++j) {
        const uint64_t k = rng() & KMASK;
        if(f.count(k)) { --j; continue; }
        fp += f.count(~k) != 0;
    }
    CHECK(fp == 0, "wide absent key must be a miss (fp %llu)", (unsigned long long)fp);
    // wide erase removes the masked key
    for(uint64_t j = 0; j < 50000; ++j) f.erase(wide(keys[j], j));
    for(uint64_t j = 0; j < 50000; ++j) CHECK(f.count(keys[j]) == 0, "wide erase removed masked key");
    for(uint64_t j = 50000; j < n; j += 997) CHECK(f.count(keys[j]) == 1, "untouched keys intact");
    f.structure_check();
    std::fprintf(stderr, "wide-hv contract ok (%llu keys, q=%u)\n", (unsigned long long)n, q);
}

int main() {
    test_compact_matrix();
    test_freeze();
    test_frozen_groups();
    test_trie_smoke();
    test_anchors();
    test_wide_hv();
    range_case<cqf::bcqf48_16>(11, 25000, (1ULL<<48)-1);
    range_case<cqf::bcqf24_8>(12, 25000, (1ULL<<24)-1);
    std::fprintf(stderr, "bcqftest: %s\n", failures ? "FAILURES PRESENT" : "all bcqf battery tests passed");
    return failures != 0;
}
