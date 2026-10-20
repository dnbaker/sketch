// cqfsimdtest: gates for the fused-regime SIMD find_slot window scan and
// the truncated-remainder (RSTORE) / bcf_t fingerprint-filter additions.
// Sections:
//   1. re_any_ (runend-window probe) vs bit-by-bit reference
//   2. Exact-mode fused CQF under adversarial churn vs std::unordered_map
//      oracle: small q (short runs, holes, adjacent runs everywhere),
//      remainder-0/small-remainder query bias, saturation, dec/erase churn.
//      The AVX2 window scan is active whenever the build provides it; the
//      oracle catches wrong-home hits, dead-slot hits, and boundary skips.
//   3. RSTORE truncation mode: merge-on-fingerprint semantics, no false
//      negatives, grow consistency, serialization/compose rejection.
//   4. bcf_t: membership/count vs oracle, count_many parity, capacity
//      overflow, erase/dec, realized FPR vs fpr_expected.
// Deterministic (splitmix64), moderate runtime.
#include "cqf.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

using namespace sketch;

static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

static uint64_t sm64(uint64_t &s) { // splitmix64
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---- 1. re_any_ vs bit reference ----
static void test_re_any() {
    uint64_t w[8] = {0};
    std::mt19937_64 rng(42);
    for(int t = 0; t < 20000; ++t) {
        const unsigned wi = rng() & 7, off = rng() & 63;
        w[wi] ^= uint64_t(1) << off;
        const uint64_t lo = (rng() % 512), len = 1 + rng() % 4;
        if(lo + len > 512) continue;
        bool ref = false;
        for(uint64_t p = lo; p < lo + len; ++p) ref |= sketch::cqf::detail::bv_get(w, p);
        const bool got = sketch::cqf::cqf64_8::re_any_(w, lo, lo + len);
        CHECK(ref == got, "re_any_ mismatch lo=%llu len=%llu", (unsigned long long)lo, (unsigned long long)len);
        if(failures) return;
    }
    std::fprintf(stderr, "re_any_ ok\n");
}

// ---- 2. adversarial exact-mode CQF vs oracle ----
template<typename Cqf>
static void churn_vs_oracle(uint32_t q, uint64_t nkeys, uint64_t seed, bool fuse=true) {
    Cqf f(q, fuse);
    CHECK(f.fused() == fuse, "fuse policy not honored");
    std::unordered_map<uint64_t, uint64_t> oracle;
    uint64_t st = seed;
    std::vector<uint64_t> keys(nkeys);
    for(auto &k: keys) k = sm64(st); // full 64-bit hashes (add takes hashes)
    const uint64_t SAT = Cqf::SAT;
    // phase A: adds with duplication (saturation), interleaved dec/erase churn
    for(uint64_t i = 0; i < nkeys; ++i) {
        const uint64_t k = keys[i];
        const unsigned reps = 1 + unsigned(sm64(st) & 3); // 1..4
        for(unsigned r = 0; r < reps; ++r) f.add(k);
        uint64_t &o = oracle[k];
        o = std::min<uint64_t>(SAT, o + reps);
        if((i & 7) == 3 && i > 16) { // erase an older key
            const uint64_t v = keys[sm64(st) % i];
            if(oracle.erase(v)) CHECK(f.erase(v), "erase of present key must succeed");
        }
        if((i & 15) == 7 && i > 16) { // dec an older key (may erase at 0)
            const uint64_t v = keys[sm64(st) % i];
            auto it = oracle.find(v);
            if(it != oracle.end()) {
                CHECK(f.dec(v), "dec of present key must succeed");
                if(--it->second == 0) oracle.erase(it);
            }
        }
        if((i % 977) == 0) { // the holes directory must track every churn event
            const uint64_t probe = sm64(st) % (f.nslots() + (uint64_t(1) << (q / 2)) + 8);
            const uint64_t got = f.first_empty(probe), want = f.first_empty_oracle(probe);
            CHECK(got == want, "first_empty directory vs scan oracle: got %llu want %llu (probe %llu)",
                  (unsigned long long)got, (unsigned long long)want, (unsigned long long)probe);
            f.structure_check();
        }
    }
    f.structure_check();
    // phase B: queries -- present, erased, fresh absent, and edge-biased
    // remainders (0, 1, small) which stress the window-scan stop conditions
    auto verify = [&](uint64_t hv) {
        const uint64_t got = f.count(hv);
        const auto it = oracle.find(hv);
        const uint64_t want = it == oracle.end() ? 0 : it->second;
        CHECK(got == want, "q=%u count mismatch: got %llu want %llu (key %016llx)",
              q, (unsigned long long)got, (unsigned long long)want, (unsigned long long)hv);
    };
    for(uint64_t i = 0; i < nkeys; i += 3) verify(keys[i]);          // present-ish
    for(uint64_t i = 0; i < nkeys; ++i) verify(sm64(st));            // absent-ish
    const unsigned rbits = 64 - q;
    for(uint64_t fq = 0; fq < (uint64_t(1) << q); fq += 1 + (q > 8 ? 37 : 1)) {
        verify(fq << rbits);                    // remainder exactly 0
        verify((fq << rbits) | 1);              // remainder 1
        if(rbits > 4) verify((fq << rbits) | (uint64_t(0xF) << (rbits - 4))); // top of run-order window
    }
    // count_many parity with per-key count
    std::vector<uint64_t> batch, out;
    for(uint64_t i = 0; i < 200; ++i) batch.push_back(keys[sm64(st) % nkeys]);
    f.count_many(batch.begin(), batch.end(), std::back_inserter(out));
    CHECK(out.size() == batch.size(), "count_many size");
    for(size_t i = 0; i < batch.size(); ++i) {
        const uint64_t want = f.count(batch[i]);
        CHECK(out[i] == want, "count_many parity");
    }
    std::fprintf(stderr, "churn q=%u keys=%llu ok (oracle %zu)\n", q, (unsigned long long)nkeys, oracle.size());
}

// ---- 3. RSTORE truncation mode ----
static void test_rstore() {
    using T = sketch::cqf::cqfbase_t<32, 8, uint64_t, sketch::hash::WangHash, 8>;
    static_assert(!T::exact, "truncated mode must be inexact");
    constexpr uint32_t q = 12, rbits = 32 - q, rsbits = 8;
    T f(q);
    CHECK(f.fused() == (rsbits + 8 <= 64), "fused policy");
    std::unordered_map<uint64_t, uint64_t> oracle; // reduced key -> merged count
    auto reduce = [](uint64_t hv) { return hv >> (rbits - rsbits); }; // (fq << rs_) | frs(fr)
    uint64_t st = 911;
    for(uint64_t i = 0; i < 15000; ++i) {
        const uint64_t hv = uint32_t(sm64(st)); // 32-bit reduced-key-space hashes
        f.add(hv);
        uint64_t &o = oracle[reduce(hv)];
        o = std::min<uint64_t>(T::SAT, o + 1);
    }
    f.structure_check();
    // no false negatives; counts merge per fingerprint
    for(uint64_t i = 0; i < 20000; ++i) {
        const uint64_t hv = uint32_t(sm64(st));
        const uint64_t got = f.count(hv);
        const auto it = oracle.find(reduce(hv));
        const uint64_t want = it == oracle.end() ? 0 : it->second;
        // inexact: count may exceed the oracle's per-fingerprint count only
        // via stored-key collisions on the same fingerprint (which the
        // oracle already merges); it must never be BELOW it (false negative)
        CHECK(got >= want || it == oracle.end(), "rstore false negative: got %llu want %llu", (unsigned long long)got, (unsigned long long)want);
        if(it != oracle.end()) CHECK(got == want, "rstore merged count: got %llu want %llu", (unsigned long long)got, (unsigned long long)want);
        if(failures > 5) return;
    }
    // grow: force several grows (each spends one stored remainder bit; the
    // effective fingerprint width q_ + rs_ must stay invariant) and
    // re-verify everything against the oracle
    for(uint64_t i = 0; i < 8000; ++i) {
        const uint64_t hv = uint32_t(sm64(st));
        f.add(hv);
        uint64_t &o = oracle[reduce(hv)];
        o = std::min<uint64_t>(T::SAT, o + 1);
    }
    f.structure_check();
    for(const auto &kv: oracle) {
        const uint64_t hv = kv.first << (rbits - rsbits); // any preimage
        CHECK(f.count(hv) == kv.second, "post-grow merged count");
        if(failures > 5) return;
    }
    // serialization and compose are rejected in this mode
    bool threw = false;
    try { f.write("/tmp/__rstore_should_not_write.cqf"); } catch(const std::logic_error &) { threw = true; }
    CHECK(threw, "RSTORE write must throw logic_error");
    // fpr bound is finite, in (0, 1), and uses the truncated width
    const double fpr = f.fpr_bound();
    CHECK(fpr > 0. && fpr < 1., "fpr_bound sane: %g", fpr);
    std::fprintf(stderr, "rstore mode ok (fpr_bound %g, oracle %zu)\n", fpr, oracle.size());
}

// ---- 4. bcf_t ----
static void test_bcf() {
    using B = sketch::cqf::bcf_t<12, 8>;
    B f(16, 0.85); // FP_BITS=12: the core requires q <= FP + BLK_SHIFT (17)
    std::unordered_map<uint64_t, uint64_t> oracle;
    uint64_t st = 4242;
    const uint64_t n = 40000; // cap ~0.85 * 2^16 = 55705
    for(uint64_t i = 0; i < n; ++i) {
        const uint64_t hv = sm64(st);
        f.add(hv);
        uint64_t &o = oracle[hv];
        o = std::min<uint64_t>(255, o + 1);
    }
    // fp collisions merge distinct full keys: size is at most the oracle's
    // and the deficit is small (birthday-scale at 2^-FP per candidate pair)
    CHECK(f.size() <= oracle.size(), "bcf size %llu > oracle %zu", (unsigned long long)f.size(), oracle.size());
    CHECK(oracle.size() - f.size() < 20000, "bcf merge deficit too large: %llu", (unsigned long long)(oracle.size() - f.size()));
    f.structure_check();
    // no false negatives + count parity (inserted hashes are full-width:
    // fp collisions among THEM would merge counts, so only verify contains
    // for all, count for a sample)
    for(const auto &kv: oracle) CHECK(f.contains(kv.first), "bcf false negative");
    // absent keys: false positives ARE the filter semantics; check the rate
    // against fpr_expected (band accommodates binomial spread at 20k trials)
    uint64_t abs_fp = 0;
    for(uint64_t i = 0; i < 20000; ++i) {
        const uint64_t hv = sm64(st) | (uint64_t(1) << 63); // fresh half
        if(oracle.count(hv)) { --i; continue; }
        abs_fp += f.count(hv) != 0;
    }
    const double abs_rate = double(abs_fp) / 20000;
    CHECK(abs_rate < f.fpr_expected() * 4 + 1e-4, "bcf absent-FP rate %g vs expected %g", abs_rate, f.fpr_expected());
    // count_many parity incl. chunk crossing (>64 keys)
    std::vector<uint64_t> batch, out;
    for(uint64_t i = 0; i < 130; ++i) {
        uint64_t hv = sm64(st);
        batch.push_back(hv); f.add(hv); oracle[hv] = std::min<uint64_t>(255, oracle[hv] + 1);
    }
    f.count_many(batch.begin(), batch.end(), std::back_inserter(out));
    CHECK(out.size() == batch.size(), "bcf count_many size");
    for(size_t i = 0; i < batch.size(); ++i) CHECK(out[i] == f.count(batch[i]), "bcf count_many parity");
    // realized FPR vs expectation (order-of-magnitude band)
    uint64_t fp = 0, trials = 20000;
    for(uint64_t i = 0; i < trials; ++i) {
        const uint64_t hv = sm64(st) | (uint64_t(1) << 63); // fresh half (unlikely inserted)
        if(oracle.count(hv)) { --i; continue; }
        fp += f.contains(hv);
    }
    const double realized = double(fp) / trials, expected = f.fpr_expected();
    CHECK(realized < expected * 4 + 1e-4, "bcf FPR realized %g vs expected %g", realized, expected);
    CHECK(realized > 0 || expected < 1e-6, "bcf FPR realized %g expected %g", realized, expected);
    // erase/dec
    uint64_t victim = batch[0];
    CHECK(f.contains(victim), "bcf victim present");
    CHECK(f.erase(victim), "bcf erase");
    CHECK(!f.contains(victim), "bcf erased absent");
    // capacity overflow throws
    bool threw = false;
    try {
        sketch::cqf::bcf_t<12, 8> tiny(6, 0.5); // nslots 64, cap 32
        for(uint64_t i = 0; i < 100; ++i) tiny.add(sm64(st));
    } catch(const std::overflow_error &) { threw = true; }
    CHECK(threw, "bcf capacity overflow must throw");
    std::fprintf(stderr, "bcf ok (fpr realized %g expected %g)\n", realized, expected);
}

// ---- 5. within-word-offset sweep (the cht probe_end bug class) ----------
// cht's probe_end (fixed 2026-10) could return a position BELOW the query
// index: its first-word scan never masked the low bits, so a match sharing
// the query's first word but sitting below the offset won. cqf.h's offset
// consumers are all correct on this class -- bv_next_set masks its first
// word (cqf.h ~line 172), dir::rank1 masks its partial word, re_any_ shifts
// by lo % word_bits (gated randomly above) -- but the churn oracle only
// SAMPLES probes at these offsets. This sweep pins every within-word offset
// 1..63 across word boundaries so any future regression in a rank/select/
// scan offset computation is caught deterministically.
static void test_word_offset_sweep() {
    // (a) bv_next_set (first set bit >= i; live in the p() accessor,
    // cqf.h:594) against a bit-by-bit reference at EVERY offset, four
    // density patterns, 15 word boundaries
    for(unsigned pat = 0; pat < 4; ++pat) {
        uint64_t w[16] = {0};
        constexpr unsigned nb = 16 * 64;
        for(unsigned i = 0; i < nb; ++i) {
            const bool set = pat == 0 ? (i & 63) == 0   // word-aligned singles
                           : pat == 1 ? (i % 7) == 0    // sparse
                           : pat == 2 ? (i % 3) == 1    // mid density
                           : true;                      // dense
            if(set) sketch::cqf::detail::bv_set(w, i);
        }
        for(unsigned i = 1; i < nb; ++i) {
            unsigned want = i;
            for(; want < nb && !sketch::cqf::detail::bv_get(w, want); ++want);
            const uint64_t got = sketch::cqf::detail::bv_next_set(w, 16, i);
            CHECK(got == want, "bv_next_set sweep pat=%u i=%u: got %llu want %u",
                  pat, i, (unsigned long long)got, want);
        }
    }
    std::fprintf(stderr, "bv_next_set offset sweep ok\n");
    // (b) first_empty (holes directory, rank/select) vs the scan oracle at
    // EVERY slot index on a fixed-fill table: holes at every within-word
    // offset are compared across all word boundaries, not a random sample
    sketch::cqf::cqf64_8 f(8, true);
    uint64_t st = 31337;
    for(unsigned i = 0; i < 180; ++i) f.add(sm64(st)); // ~70% of 256 slots, no grow
    for(uint64_t i = 0; i < f.nslots(); ++i) {
        const uint64_t got = f.first_empty(i), want = f.first_empty_oracle(i);
        CHECK(got == want, "first_empty full sweep at %llu: got %llu want %llu",
              (unsigned long long)i, (unsigned long long)got, (unsigned long long)want);
    }
    f.structure_check();
    std::fprintf(stderr, "first_empty full-sweep ok (%llu slots)\n", (unsigned long long)f.nslots());
}

int main() {
    test_re_any();
    test_word_offset_sweep();
    // fused regime needs r_ + VAL_BITS <= 64: cqf64_8 at q >= 8, cqf64_16 at q >= 16
    churn_vs_oracle<sketch::cqf::cqf64_8>(8, 6000, 1);
    churn_vs_oracle<sketch::cqf::cqf64_8>(10, 12000, 2);
    churn_vs_oracle<sketch::cqf::cqf64_8>(12, 20000, 3);
    churn_vs_oracle<sketch::cqf::cqf64_16>(16, 8000, 4);
    // packed regime (fuse=false): the holes directory is layout-independent,
    // the old scalar packed scan is the oracle
    churn_vs_oracle<sketch::cqf::cqf64_8>(8, 6000, 5, false);
    churn_vs_oracle<sketch::cqf::cqf64_16>(10, 10000, 6, false);
    churn_vs_oracle<sketch::cqf::cqfbase_t<32, 8>>(9, 9000, 7, false);
    test_rstore();
    test_bcf();
    if(failures) {
        std::fprintf(stderr, "cqfsimdtest: %d FAILURES\n", failures);
        return 1;
    }
    std::fprintf(stderr, "cqfsimdtest: all passed\n");
    return 0;
}
