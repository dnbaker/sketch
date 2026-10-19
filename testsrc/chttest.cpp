// cht gates: reversible-hash round-trips (identity / splitmix64 / Feistel-
// 128), exact-mode membership+counts vs cqfbase_t on identical streams,
// decode/dump set-equality (the jellyfish dumper property), truncated-mode
// FPR ballpark, bulk-vs-single equivalence, edge cases.
#include "cht.h"
#include "cqf.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace sketch;
static int failures = 0;
#define CHECK(cond, ...) do { \
    if(!(cond)) { \
        std::fprintf(stderr, "[FAIL:%d] ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); \
        ++failures; \
    } \
} while(0)

using u128 = unsigned __int128;

// ---- 1. hash o invert == identity ---------------------------------------
void test_hash_roundtrips() {
    {   // identity on 48-bit reduced keys: exact on the masked domain
        cht_identity_hash<48> h;
        std::mt19937_64 rng(1);
        for (uint64_t i = 0; i < (1u << 22); ++i) {
            const uint64_t v = rng();
            CHECK(h.invert(h.hash(v)) == (v & ((1ull << 48) - 1)), "identity roundtrip lost bits");
        }
        for (uint64_t e: {0ull, 1ull, (1ull << 48) - 1, ~0ull}) {
            CHECK(h.invert(h.hash(e)) == (e & ((1ull << 48) - 1)), "identity edge %llx", (unsigned long long)e);
            CHECK(h.hash(h.invert(e & ((1ull << 48) - 1))) == (e & ((1ull << 48) - 1)), "identity edge fwd %llx", (unsigned long long)e);
        }
        CHECK(h.key_bits == 48, "identity key_bits");
        std::fprintf(stderr, "cht identity-hash roundtrip done\n");
    }
    {   // splitmix64: documented multiplicative inverses, then a wide sample
        typedef cht_splitmix64_hash H;
        CHECK(H::M1 * H::IM1 == 1 && H::M2 * H::IM2 == 1, "splitmix64 modular inverses wrong");
        H h;
        std::mt19937_64 rng(2);
        for (uint64_t i = 0; i < (1u << 22); ++i) {
            const uint64_t v = rng();
            if (h.invert(h.hash(v)) != v) { CHECK(false, "splitmix64 roundtrip failed at %llu", (unsigned long long)v); break; }
        }
        for (uint64_t e: {0ull, 1ull, ~0ull, 1ull << 63, (1ull << 63) - 1, 0xffffffffull}) {
            CHECK(h.invert(h.hash(e)) == e, "splitmix64 edge %llx", (unsigned long long)e);
            CHECK(h.hash(h.invert(e)) == e, "splitmix64 edge fwd %llx", (unsigned long long)e);
        }
        std::fprintf(stderr, "cht splitmix64 roundtrip done\n");
    }
    {   // Feistel-128 on unsigned __int128
        cht_feistel128_hash h(1337);
        std::mt19937_64 rng(3);
        auto mk = [](uint64_t hi, uint64_t lo) {return (u128(hi) << 64) | lo;};
        for (uint64_t i = 0; i < (1u << 20); ++i) {
            const u128 v = mk(rng(), rng());
            if (h.invert(h.hash(v)) != v) { CHECK(false, "feistel128 roundtrip failed"); break; }
            if (h.hash(h.invert(v)) != v) { CHECK(false, "feistel128 inverse-fwd failed"); break; }
        }
        for (u128 e: {mk(0,0), mk(0,1), mk(~0ull,~0ull), mk(~0ull,0), mk(0,~0ull), mk(1,0)}) {
            CHECK(h.invert(h.hash(e)) == e, "feistel128 edge");
            CHECK(h.hash(h.invert(e)) == e, "feistel128 edge fwd");
        }
        // distinct inputs give distinct hashes (bijection spot-check)
        std::vector<u128> hv(1u << 12);
        for (auto &v: hv) v = h.hash(mk(rng(), rng()));
        std::sort(hv.begin(), hv.end());
        CHECK(std::unique(hv.begin(), hv.end()) == hv.end(), "feistel128 not injective on 4k sample");
        std::fprintf(stderr, "cht feistel128 roundtrip done\n");
    }
}

// ---- 2. exact mode vs cqfbase_t on identical streams ---------------------
void test_exact_vs_cqf() {
    for (const uint32_t q: {8u, 12u}) {
        cht_t<uint64_t, cht_identity_hash<48>, 16> f(q);
        cqfbase_t<48,16> g(q);
        std::mt19937_64 rng(100 + q);
        std::unordered_map<uint64_t, uint64_t> ref;
        const uint64_t target = uint64_t(0.7 * (1ull << q)); // cht has no overflow tail
        for (unsigned i = 0; i < 4000; ++i) {
            // half from a small pool (duplicate counts), half wide (distinct volume)
            const uint64_t k = (i & 1) ? (rng() % 64) : (rng() & ((1ull << 48) - 1));
            if (ref.size() >= target && !ref.count(k)) continue;
            f.add(k); g.add(k); ++ref[k];
        }
        CHECK(f.size() == ref.size(), "cht size %llu != ref %zu (q=%u)", (unsigned long long)f.size(), ref.size(), q);
        for (const auto &kv: ref) {
            CHECK(f.count(kv.first) == kv.second, "cht count %llu != %llu (q=%u)",
                  (unsigned long long)f.count(kv.first), (unsigned long long)kv.second, q);
            CHECK(g.count(kv.first) == kv.second, "cqfbase count mismatch (q=%u)", q);
        }
        unsigned fpc = 0, fpg = 0;
        for (unsigned i = 0; i < 20000; ++i) {
            const uint64_t k = rng() & ((1ull << 48) - 1);
            if (ref.count(k)) continue;
            fpc += f.count(k) != 0; fpg += g.count(k) != 0;
        }
        // exact reduced-key mode: absent-key hits only from fingerprint
        // collisions, ~ n*probe/2^48 -- both sides should see (nearly) none
        CHECK(fpc <= 1, "cht exact mode false positives %u (q=%u)", fpc, q);
        CHECK(fpg <= 1, "cqf exact mode false positives %u (q=%u)", fpg, q);
        std::fprintf(stderr, "cht vs cqfbase exact q=%u done (%zu distinct, fp %u/%u)\n",
                     q, ref.size(), fpc, fpg);
    }
}

// ---- 3. decode/dump (jellyfish dumper property) --------------------------
void test_decode_dump() {
    {   // identity hash: dump recovers the REDUCED keys, counts intact
        const uint32_t q = 16;
        cht_t<uint64_t, cht_identity_hash<48>, 16> f(q);
        std::mt19937_64 rng(7);
        std::unordered_map<uint64_t, uint64_t> ref;
        for (unsigned i = 0; i < 50000; ++i) {
            const uint64_t k = rng() & ((1ull << 48) - 1);
            f.add(k); ++ref[k];
        }
        std::vector<uint64_t> keys;
        f.dump_keys(std::back_inserter(keys));
        CHECK(keys.size() == f.size() && keys.size() == ref.size(), "identity dump size %zu != table %llu",
              keys.size(), (unsigned long long)f.size());
        std::sort(keys.begin(), keys.end());
        std::vector<uint64_t> want;
        want.reserve(ref.size());
        for (const auto &kv: ref) want.push_back(kv.first);
        std::sort(want.begin(), want.end());
        CHECK(keys == want, "identity dump set != inserted reduced-key set");
        // iterator pairs: decoded key + mapped() count agree with the ref
        uint64_t seen = 0;
        for (auto it = f.begin(); it != f.end(); ++it) {
            CHECK(ref.at(*it) == it.mapped(), "identity dump pair count wrong");
            ++seen;
        }
        CHECK(seen == f.size(), "identity iterator visited %llu != size %llu",
              (unsigned long long)seen, (unsigned long long)f.size());
        // home-slot order: iterator slots strictly increase and are occupied
        uint64_t prev = 0;
        bool first = true;
        for (auto it = f.begin(); it != f.end(); ++it) {
            if (!first) CHECK(it.slot() > prev, "iterator not in home-slot order");
            first = false; prev = it.slot();
        }
        std::fprintf(stderr, "cht identity decode/dump done (%llu keys)\n", (unsigned long long)f.size());
    }
    {   // splitmix64: dump recovers the RAW 64-bit keys (hash inversion,
        // not the hash values themselves)
        const uint32_t q = 16;
        cht_t<uint64_t, cht_splitmix64_hash, 16> f(q);
        std::mt19937_64 rng(8);
        std::unordered_map<uint64_t, uint64_t> ref;
        for (unsigned i = 0; i < 50000; ++i) {
            const uint64_t k = rng(); // full 64-bit keys, no pre-hashing
            f.add(k); ++ref[k];
        }
        std::vector<uint64_t> keys;
        f.dump_keys(std::back_inserter(keys));
        std::sort(keys.begin(), keys.end());
        std::vector<uint64_t> want;
        want.reserve(ref.size());
        for (const auto &kv: ref) want.push_back(kv.first);
        std::sort(want.begin(), want.end());
        CHECK(keys == want, "splitmix dump set != inserted raw-key set");
        // the dump must be the INVERTED keys, not stored hash values:
        // {hash(k)} differs from the dump (mixing moves values whp), and
        // count() on a raw key works with no pre-hashing by the caller
        std::vector<uint64_t> hvs;
        cht_splitmix64_hash h;
        for (const auto k: want) hvs.push_back(h.hash(k));
        std::sort(hvs.begin(), hvs.end());
        CHECK(keys != hvs, "dump equals the hash-value set -- inversion missing?");
        CHECK(f.count(want.front()) == 1 && f.count(want.back()) == 1, "splitmix count of raw keys wrong");
        std::fprintf(stderr, "cht splitmix64 decode/dump done (%llu keys)\n", (unsigned long long)f.size());
    }
    {   // truncated mode: membership-only -- dump yields deterministic
        // fingerprint REPRESENTATIVES, never inserted keys
        const uint32_t q = 16;
        cht_t<uint64_t, cht_identity_hash<48>, 16, 12> f(q);
        std::mt19937_64 rng(9);
        std::unordered_map<uint64_t, uint64_t> ref;
        for (unsigned i = 0; i < 50000; ++i) {
            const uint64_t k = rng() & ((1ull << 48) - 1);
            f.add(k); ++ref[k];
        }
        std::vector<std::pair<uint64_t, uint64_t>> pairs;
        f.dump(std::back_inserter(pairs));
        CHECK(pairs.size() == f.size(), "truncated dump size %zu != %llu", pairs.size(), (unsigned long long)f.size());
        uint64_t cntsum = 0;
        for (const auto &kv: pairs) {
            CHECK(f.contains(kv.first), "truncated representative not contained");
            cntsum += kv.second;
        }
        CHECK(cntsum == 50000, "truncated counts lost mass: %llu", (unsigned long long)cntsum);
        // representatives are canonical: the fingerprint re-embedded at the
        // TOP of the remainder field means the low (r - rs) bits are zero
        const unsigned r = f.r();
        const unsigned drop = r > 12 ? r - 12 : 0; // rs_ = min(12, r)
        const uint64_t lowmask = drop ? ((1ull << drop) - 1) : 0;
        for (const auto &kv: pairs)
            CHECK((kv.first & lowmask) == 0, "representative low bits nonzero");
        std::fprintf(stderr, "cht truncated representative dump done (%llu fps, mass %llu)\n",
                     (unsigned long long)f.size(), (unsigned long long)cntsum);
    }
}

// ---- 4. truncated FPR in the clustering-aware ballpark -------------------
void test_truncated_fpr() {
    const uint32_t q = 16, rs = 12;
    const uint64_t nslots = 1ull << q;
    const uint64_t n = uint64_t(0.9 * nslots);
    cht_t<uint64_t, cht_identity_hash<48>, 16, rs> f(q);
    std::mt19937_64 rng(11);
    std::unordered_map<uint64_t, uint64_t> ref;
    while (ref.size() < n) {
        const uint64_t k = rng() & ((1ull << 48) - 1);
        if (ref.count(k) == 0) { f.add(k); ref[k] = 1; }
    }
    const double alpha = f.load_factor();
    // clustering-aware expectation: the probe compares rs_-wide remainders
    // across every occupied slot in the unsuccessful-probe span, whose
    // expected length is L(a) = (1 + 1/(1-a)^2)/2
    const double L = (1. + 1. / (1. - alpha) / (1. - alpha)) / 2.;
    const double est = 1. - std::exp(-L / double(uint64_t(1) << rs));
    uint64_t fp = 0, trials = 0, fn = 0;
    for (uint64_t i = 0; i < 300000; ++i) {
        const uint64_t k = rng() & ((1ull << 48) - 1);
        if (ref.count(k)) continue;
        fp += f.count(k) != 0; ++trials;
    }
    for (const auto &kv: ref) fn += f.count(kv.first) == 0;
    const double got = double(fp) / double(trials);
    std::fprintf(stderr, "cht truncated fpr: measured %.5f over %llu trials; span-aware est %.5f (L=%.1f); fpr_bound %.5f; absent-inserts %llu\n",
                 got, (unsigned long long)trials, est, L, f.fpr_bound(), (unsigned long long)fn);
    CHECK(fn == 0, "truncated mode has false negatives (%llu)", (unsigned long long)fn);
    CHECK(std::abs(got - f.fpr_bound()) < 0.01, "fpr_bound %.5f far from measured %.5f", f.fpr_bound(), got);
    CHECK(got < est * 2.5, "truncated FPR %.5f above 2.5x the span estimate %.5f", got, est);
    CHECK(got > est / 5., "truncated FPR %.5f implausibly below est/5 %.5f (bug?)", got, est / 5.);
    CHECK(got < 0.05, "truncated FPR %.5f not membership-filter class", got);
}

// ---- 5. bulk build == single inserts -------------------------------------
void test_bulk_equiv() {
    const uint32_t q = 12; // 4096 slots
    std::mt19937_64 rng(13);
    std::vector<uint64_t> stream;
    std::unordered_map<uint64_t, uint64_t> ref;
    for (unsigned i = 0; i < 3000; ++i) {
        const uint64_t k = rng() % 1500; // heavy duplicates
        stream.push_back(k); ++ref[k];
    }
    cht_t<uint64_t, cht_identity_hash<48>, 16> f_single(q);
    for (const uint64_t k: stream) f_single.add(k);
    std::vector<uint64_t> sorted(stream);
    std::sort(sorted.begin(), sorted.end()); // identity exact: hash order == key order
    cht_t<uint64_t, cht_identity_hash<48>, 16> f_bulk(q);
    f_bulk.add_bulk_sorted(sorted.begin(), sorted.end());
    CHECK(f_single.size() == f_bulk.size(), "bulk size %llu != single %llu",
          (unsigned long long)f_bulk.size(), (unsigned long long)f_single.size());
    CHECK(f_single.bytes_used() == f_bulk.bytes_used(), "bulk bytes %zu != single %zu",
          f_bulk.bytes_used(), f_single.bytes_used());
    for (const auto &kv: ref) {
        CHECK(f_single.count(kv.first) == kv.second, "single count wrong");
        CHECK(f_bulk.count(kv.first) == kv.second, "bulk count wrong");
    }
    for (unsigned i = 0; i < 20000; ++i) {
        const uint64_t k = rng() % 1500 + 2000; // absent band
        CHECK(f_single.count(k) == 0 && f_bulk.count(k) == 0, "absent key hit");
    }
    // bulk equivalence also in truncated mode (fp-merging included)
    cht_t<uint64_t, cht_identity_hash<48>, 16, 8> t_single(q);
    for (const uint64_t k: stream) t_single.add(k);
    // (home, fp) order = masked-key bits [r - rs, 48) shifted down
    const unsigned fsh = unsigned(48 - q - 8);
    std::vector<uint64_t> tsort(stream);
    std::sort(tsort.begin(), tsort.end(), [fsh](uint64_t a, uint64_t b){return (a >> fsh) < (b >> fsh);});
    cht_t<uint64_t, cht_identity_hash<48>, 16, 8> t_bulk(q);
    t_bulk.add_bulk_sorted(tsort.begin(), tsort.end());
    CHECK(t_single.size() == t_bulk.size(), "truncated bulk size mismatch");
    for (const uint64_t k: stream) CHECK(t_single.count(k) == t_bulk.count(k), "truncated bulk count mismatch");
    {   // wraparound bulk: the cyclic frontier crosses the table end and
        // wraps into the hole below the first home; placement must equal
        // single inserts and decode must recover homes through the wrap.
        // q=6 (64 slots): homes 2..63 all occupied (62 slots) plus an
        // extra distinct key -> the 63rd placement wraps to slot 0.
        const uint32_t q = 6;
        const uint64_t r = 48 - q;
        std::mt19937_64 rng(21);
        std::vector<uint64_t> stream;
        std::unordered_map<uint64_t, uint64_t> ref;
        for (uint64_t h = 2; h < 64; ++h) { // homes 2..63, one distinct key each
            const uint64_t k = (h << r) | (rng() & ((1ull << r) - 1));
            for (unsigned j = 0; j < 1 + (h % 3); ++j) // 1..3 copies (count > 1)
                stream.push_back(k);
            ref[k] = 1 + (h % 3);
        }
        const uint64_t extra = (5ull << r) | (1ull << (r - 1)); // distinct rem in home 5
        stream.push_back(extra); ++ref[extra];
        // 63 distinct fingerprints in 64 slots, first home 2: the build's
        // frontier crosses 63 and wraps into the hole below slot 2
        cht_t<uint64_t, cht_identity_hash<48>, 16> f_single(q);
        for (const uint64_t k: stream) f_single.add(k);
        std::vector<uint64_t> sorted(stream);
        std::sort(sorted.begin(), sorted.end());
        cht_t<uint64_t, cht_identity_hash<48>, 16> f_bulk(q);
        f_bulk.add_bulk_sorted(sorted.begin(), sorted.end());
        CHECK(f_single.size() == f_bulk.size(), "wrap bulk size mismatch");
        for (const auto &kv: ref) {
            CHECK(f_single.count(kv.first) == kv.second, "wrap single count wrong");
            CHECK(f_bulk.count(kv.first) == kv.second, "wrap bulk count wrong");
        }
        for (unsigned i = 0; i < 20000; ++i) { // absent keys (homes 0..1 never inserted)
            const uint64_t k = (uint64_t(i % 2) << r) | (rng() & ((1ull << r) - 1));
            CHECK(f_single.count(k) == 0 && f_bulk.count(k) == 0, "wrap absent hit");
        }
        // decode: the wrapped slot(s) below first_home must still decode
        std::vector<uint64_t> dumped;
        f_bulk.dump_keys(std::back_inserter(dumped));
        std::sort(dumped.begin(), dumped.end());
        std::vector<uint64_t> want;
        for (const auto &kv: ref) want.push_back(kv.first);
        std::sort(want.begin(), want.end());
        CHECK(dumped == want, "wrap bulk dump set mismatch");
        // a truly wrapped slot exists: some key sits at a slot BELOW its home
        uint64_t wrapped = 0;
        for (auto it = f_bulk.begin(); it != f_bulk.end(); ++it)
            if (it.slot() < f_bulk.home(f_bulk.hash(*it))) ++wrapped;
        CHECK(wrapped != 0, "test setup failed to produce a wrapped slot");
        std::fprintf(stderr, "cht wraparound bulk done (%zu keys, %zu wrapped slots)\n", ref.size(), (size_t)wrapped);
    }
    std::fprintf(stderr, "cht bulk-vs-single done (%zu stream, %zu distinct)\n", stream.size(), ref.size());
}

// ---- 6. edges: overflow, bad q, 128-bit config, saturation, masking ------
void test_edges() {
    {   // full table throws std::overflow_error, not corruption
        cht_t<uint64_t, cht_identity_hash<48>, 16> f(2); // 4 slots
        for (uint64_t h = 0; h < 4; ++h) f.add(h << 46); // one per home
        bool threw = false;
        try { f.add(3); } catch (const std::overflow_error &) { threw = true; } // home 0, table full
        CHECK(threw, "no overflow_error at a full 4-slot table");
        CHECK(f.size() == 4 && f.count(0) == 1, "full table state wrong");
        std::fprintf(stderr, "cht overflow edge done\n");
    }
    {   // argument validation
        bool threw = false;
        try { cht_t<uint64_t, cht_identity_hash<48>, 16> f(0); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "accepted q=0");
        threw = false;
        try { cht_t<uint64_t, cht_identity_hash<48>, 16> f(41); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "accepted q=41");
        threw = false;
        try { cht_t<uint64_t, cht_identity_hash<10>, 16> f(10); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "accepted q >= key_bits (r_ = 0)");
    }
    {   // 128-bit keys: exact mode is unreachable (u64 fingerprint cell),
        // truncated mode is the supported configuration
        bool threw = false;
        try { cht_t<u128, cht_feistel128_hash, 16> f(22); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "exact 128-bit table did not reject the u64 cell");
        cht_t<u128, cht_feistel128_hash, 16, 12> f(15); // 32768 slots, fill ~0.61
        std::mt19937_64 rng(17);
        auto mk = [](uint64_t hi, uint64_t lo) {return (u128(hi) << 64) | lo;};
        std::vector<u128> keys;
        for (unsigned i = 0; i < 20000; ++i) {
            const u128 k = mk(rng(), rng());
            f.add(k); keys.push_back(k);
        }
        std::sort(keys.begin(), keys.end());
        CHECK(std::unique(keys.begin(), keys.end()) == keys.end(), "128-bit sample collided (bad luck or bad hash)");
        uint64_t hit = 0;
        for (const u128 k: keys) hit += f.count(k) != 0;
        CHECK(hit == keys.size(), "128-bit membership lost keys: %llu/%zu", (unsigned long long)hit, keys.size());
        // representatives: deterministic, contained, counter mass intact
        uint64_t mass = 0;
        for (auto it = f.begin(); it != f.end(); ++it) { mass += it.mapped(); CHECK(f.contains(*it), "128-bit rep not contained"); }
        CHECK(mass == 20000, "128-bit mass %llu", (unsigned long long)mass);
        std::fprintf(stderr, "cht 128-bit truncated config done\n");
    }
    {   // counter saturation at VAL_BITS
        cht_t<uint64_t, cht_splitmix64_hash, 8> f(8);
        for (unsigned i = 0; i < 1000; ++i) f.add(42);
        CHECK(f.count(42) == 255, "saturation: got %llu", (unsigned long long)f.count(42));
        cht_t<uint64_t, cht_splitmix64_hash, 64> g(8);
        for (unsigned i = 0; i < 100000; ++i) g.add(7);
        CHECK(g.count(7) == 100000, "64-bit counter clamped: %llu", (unsigned long long)g.count(7));
        std::fprintf(stderr, "cht saturation done\n");
    }
    {   // identity masking on garbage top bits (the pre-hashed contract):
        // add(~0) lands at the reduced key 2^48-1 and decodes there
        cht_t<uint64_t, cht_identity_hash<48>, 16> f(8);
        f.add(~0ull);
        CHECK(f.count(~0ull) == 1 && f.count((1ull << 48) - 1) == 1 && f.count((1ull << 48) - 2) == 0,
              "identity masking contract broken");
        auto it = f.begin();
        CHECK(*it == (1ull << 48) - 1, "decode of masked key wrong");
        CHECK(f.begin() != f.end() && ++f.begin() == f.end(), "singleton iterator bounds wrong");
        std::fprintf(stderr, "cht masking contract done\n");
    }
    {   // empty table behavior + exact-mode flag semantics
        cht_t<uint64_t, cht_identity_hash<48>, 16> f(8);
        CHECK(f.empty() && f.size() == 0 && f.count(123) == 0 && f.begin() == f.end(), "empty table wrong");
        const bool id48_exact = cht_t<uint64_t, cht_identity_hash<48>, 16>::exact;
        CHECK(!id48_exact, "key48 identity must be fingerprint-class");
        const bool id64_exact = cht_t<uint64_t, cht_identity_hash<64>, 16>::exact;
        const bool mix_exact = cht_t<uint64_t, cht_splitmix64_hash, 16>::exact;
        const bool mix_trunc_exact = cht_t<uint64_t, cht_splitmix64_hash, 16, 8>::exact;
        CHECK(id64_exact, "key64 identity should be exact-class");
        CHECK(mix_exact, "splitmix64 must be exact-class");
        CHECK(!mix_trunc_exact, "truncated splitmix64 must not claim exact");
        std::fprintf(stderr, "cht empty/flag edges done\n");
    }
}

// ---- 7. bulk build at high fill (defect-A regression) --------------------
// The cyclically-contiguous frontier must absorb the table-end wrap at fill
// 0.9+ (tail runs land in the hole below the first home) instead of throwing
// overflow_error, and stay placement-identical to single inserts of the
// same sorted stream. Exercises the displacement spill (wrapped runs have
// long probes). Exact and truncated modes, fills 0.9 and 0.95.
void test_bulk_highfill() {
    for (const double fill: {0.90, 0.95}) {
        const uint32_t q = 20; // 1M slots
        const uint64_t ns = 1ull << q;
        std::vector<uint64_t> keys;
        {
            std::mt19937_64 rng(23);
            std::unordered_map<uint64_t, uint64_t> seen;
            while (keys.size() < uint64_t(fill * ns)) {
                const uint64_t k = rng() & ((1ull << 48) - 1);
                if (seen.insert({k, 1}).second) keys.push_back(k);
            }
            std::sort(keys.begin(), keys.end()); // identity exact: hash order == key order
        }
        const uint64_t n = keys.size();
        cht_t<uint64_t, cht_identity_hash<48>, 16> f_single(q);
        for (const uint64_t k: keys) f_single.add(k);
        cht_t<uint64_t, cht_identity_hash<48>, 16> f_bulk(q);
        f_bulk.add_bulk_sorted(keys.begin(), keys.end()); // must NOT throw
        CHECK(f_single.size() == n && f_bulk.size() == n, "highfill bulk size %llu/%llu != %llu (fill %.2f)",
              (unsigned long long)f_single.size(), (unsigned long long)f_bulk.size(), (unsigned long long)n, fill);
        CHECK(f_single.bytes_used() == f_bulk.bytes_used(), "highfill bulk bytes differ (spill?)");
        std::mt19937_64 rng(24);
        uint64_t checked = 0;
        for (uint64_t t = 0; t < 40000; ++t, ++checked) {
            const uint64_t k = keys[rng() % n];
            CHECK(f_single.count(k) == 1 && f_bulk.count(k) == 1, "highfill count wrong (fill %.2f)", fill);
        }
        for (uint64_t t = 0; t < 20000; ++t, ++checked) {
            const uint64_t k = rng() & ((1ull << 48) - 1);
            if (std::binary_search(keys.begin(), keys.end(), k)) continue;
            CHECK(f_single.count(k) == 0 && f_bulk.count(k) == 0, "highfill absent key hit (fill %.2f)", fill);
        }
        // decode set-equality on the bulk-built table (wrap + spill + fallback
        // together). The dump comes out in HOME-SLOT order, which is not
        // ascending key order once displacement inverts two homes or the
        // fallback scatters keys into earlier holes -- compare as sets.
        std::vector<uint64_t> dumped;
        f_bulk.dump_keys(std::back_inserter(dumped));
        std::sort(dumped.begin(), dumped.end());
        CHECK(dumped == keys, "highfill bulk dump != input set (fill %.2f)", fill);
        // truncated mode at the same fill: membership-equivalent, no FN.
        // Sizes may LEGITIMATELY differ: single inserts merge equal
        // fingerprints across DIFFERENT homes when one's probe span reaches
        // the other's slot; the bulk path keeps one slot per (home, fp)
        // pair, which is the purer fingerprint-filter semantics (no
        // cross-home count coupling). Membership answers are identical.
        {
            const unsigned fsh = unsigned(48 - q - 12);
            std::vector<uint64_t> tkeys(keys);
            std::sort(tkeys.begin(), tkeys.end(), [fsh](uint64_t a, uint64_t b){return (a >> fsh) < (b >> fsh);});
            cht_t<uint64_t, cht_identity_hash<48>, 16, 12> t_single(q);
            for (const uint64_t k: tkeys) t_single.add(k);
            cht_t<uint64_t, cht_identity_hash<48>, 16, 12> t_bulk(q);
            t_bulk.add_bulk_sorted(tkeys.begin(), tkeys.end());
            CHECK(t_bulk.size() >= t_single.size(), "highfill truncated bulk shrank vs single (fill %.2f)", fill);
            CHECK(t_bulk.count(keys.front()) != 0 && t_bulk.count(keys.back()) != 0, "highfill truncated bulk lost endpoints");
            uint64_t fn = 0;
            for (const uint64_t k: keys) fn += t_bulk.count(k) == 0;
            CHECK(fn == 0, "highfill truncated bulk false negatives: %llu (fill %.2f)", (unsigned long long)fn, fill);
        }
        std::fprintf(stderr, "cht bulk highfill %.2f done (%llu keys, single %zu B, bulk %zu B)\n",
                     fill, (unsigned long long)n, f_single.bytes_used(), f_bulk.bytes_used());
    }
}

// ---- 8. bytes_used is exactly the SoA formula ----------------------------
void test_bytes_accounting() {
    const uint32_t q = 22;
    const uint64_t ns = 1ull << q;
    cht_t<uint64_t, cht_identity_hash<48>, 16> f(q);
    const size_t occ = ns / 64 * 8;                       // 1-bit bitmap
    const size_t rem = ((ns * 26 + 63) / 64 + 1) * 8;     // r = 48-22 = 26 bits + 1 spare word
    const size_t cnt = ((ns * 16 + 63) / 64 + 1) * 8;     // 16-bit counters + 1 spare word
    const size_t dsp = ns;                                // u8 displacement array (DECODE_BITS=8)
    CHECK(f.bytes_used() == occ + rem + cnt + dsp, "bytes_used %zu != occ+rem+cnt+disp %zu",
          f.bytes_used(), occ + rem + cnt + dsp);
    cht_t<uint64_t, cht_identity_hash<48>, 16, 0, false> s(q); // sentinel mode: no bitmap
    CHECK(s.bytes_used() == rem + cnt + dsp, "sentinel bytes_used %zu != rem+cnt+disp %zu", s.bytes_used(), rem + cnt + dsp);
    const bool no_decode_exact = cht_t<uint64_t, cht_identity_hash<48>, 16, true, 0>::exact;
    CHECK(!no_decode_exact, "DECODE_BITS=0 changed exactness semantics");
    std::fprintf(stderr, "cht bytes accounting done (exact q=22: %zu B = %.1f bits/slot with disp array)\n",
                 f.bytes_used(), double(f.bytes_used()) * 8 / double(ns));
}

// ---- 9. SIMD batch paths: lane-identity and table equivalence --------------
// hashes_many/add_many_simd/count_many_simd route u64 keys through
// RevHash::hash4 4 lanes at a time; after the round-2 delegation these lanes
// are bit-identical to the scalar chain (hash()), and add_many_simd inserts
// through the identical probe as add(). These gates pin both facts at the
// table level: batch-built tables must answer exactly like scalar-built ones.
void test_batch_simd() {
    {   // hashes_many == hf_.hash elementwise (SIMD lanes + scalar tail;
        // 100003 is not a multiple of 4, so the tail path runs). Both
        // hash4-carrying instantiations are checked.
        std::mt19937_64 rng(31);
        const unsigned n = 100003;
        std::vector<uint64_t> keys(n), hv(n);
        for (auto &k: keys) k = rng();
        {   cht_t<uint64_t, cht_splitmix64_hash, 16> f(8);
            cht_splitmix64_hash h;
            f.hashes_many(keys.begin(), keys.end(), hv.begin());
            for (unsigned i = 0; i < n; ++i)
                CHECK(hv[i] == h.hash(keys[i]), "hashes_many != hash() at %u", i);
        }
        {   cht_t<uint64_t, cht_identity_hash<48>, 16> f(8);
            cht_identity_hash<48> h;
            f.hashes_many(keys.begin(), keys.end(), hv.begin());
            for (unsigned i = 0; i < n; ++i)
                CHECK(hv[i] == h.hash(keys[i]), "identity hashes_many != hash() at %u", i);
        }
        std::fprintf(stderr, "cht hashes_many lane-identity done (%u keys)\n", n);
    }
    {   // exact mode: SIMD-built table == scalar-built table on identical
        // streams, across every batch-count variant, including dump sets
        const uint32_t q = 15; // 32768 slots; ~23k distinct below
        std::mt19937_64 rng(32);
        std::vector<uint64_t> stream;
        std::unordered_map<uint64_t, uint64_t> ref;
        for (unsigned i = 0; i < 60000; ++i) {
            const uint64_t k = (i % 3) ? (rng() % 3000) : rng(); // dups + wide
            stream.push_back(k); ++ref[k];
        }
        cht_t<uint64_t, cht_splitmix64_hash, 16> f_single(q);
        for (const uint64_t k: stream) f_single.add(k);
        std::vector<uint64_t> shuffled(stream);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        cht_t<uint64_t, cht_splitmix64_hash, 16> f_simd(q);
        f_simd.add_many_simd(shuffled.begin(), shuffled.end());
        CHECK(f_single.size() == f_simd.size(), "add_many_simd size %llu != single %llu",
              (unsigned long long)f_simd.size(), (unsigned long long)f_single.size());
        CHECK(f_single.bytes_used() == f_simd.bytes_used(), "add_many_simd bytes differ");
        std::vector<uint64_t> probe = stream;
        for (unsigned i = 0; i < 20000; ++i) probe.push_back(rng() | (1ull << 60));
        std::vector<uint64_t> c_ref(probe.size()), c_simd(probe.size()), c_pf(probe.size());
        f_simd.count_many(probe.begin(), probe.end(), c_ref.begin());
        f_simd.count_many_simd(probe.begin(), probe.end(), c_simd.begin());
        f_simd.count_many_prefetch(probe.begin(), probe.end(), c_pf.begin());
        CHECK(c_ref == c_simd, "count_many_simd != count_many");
        CHECK(c_ref == c_pf, "count_many_prefetch != count_many");
        {   // sorted variant: input ordered by hf_.hash ascending (the
            // add_bulk_sorted order); must return identical counts
            cht_splitmix64_hash h;
            std::vector<uint64_t> sprobe(probe);
            std::sort(sprobe.begin(), sprobe.end(), [&h](uint64_t a, uint64_t b){return h.hash(a) < h.hash(b);});
            std::vector<uint64_t> c_sorted(sprobe.size());
            f_simd.count_many_sorted(sprobe.begin(), sprobe.end(), c_sorted.begin());
            for (size_t i = 0; i < sprobe.size(); ++i)
                CHECK(c_sorted[i] == f_simd.count(sprobe[i]), "count_many_sorted wrong at %zu", i);
        }
        for (size_t i = 0; i < probe.size(); ++i)
            CHECK(c_ref[i] == f_single.count(probe[i]), "simd-built table count != single-built at %zu", i);
        // dump set-equality between the two builds (both are the raw-key set)
        std::vector<uint64_t> d_single, d_simd;
        f_single.dump_keys(std::back_inserter(d_single));
        f_simd.dump_keys(std::back_inserter(d_simd));
        std::sort(d_single.begin(), d_single.end());
        std::sort(d_simd.begin(), d_simd.end());
        CHECK(d_single == d_simd, "simd-built dump set != scalar-built dump set");
        CHECK(d_single.size() == ref.size(), "dump size %zu != ref %zu", d_single.size(), ref.size());
        for (const uint64_t k: d_single) CHECK(ref.count(k) == 1, "dumped key not in reference set");
        std::fprintf(stderr, "cht exact-mode SIMD batch equivalence done (%zu stream, %zu distinct)\n",
                     stream.size(), ref.size());
    }
    {   // truncated mode: batch paths fall out of the same (home, fp)
        // placement; membership must match single inserts exactly, with no
        // false negatives (per-mode counts may couple across homes on
        // insertion order -- only membership is pinned here)
        const uint32_t q = 12; // 4096 slots, ~2500 distinct below
        std::mt19937_64 rng(33);
        std::vector<uint64_t> stream;
        for (unsigned i = 0; i < 2500; ++i) stream.push_back(rng());
        cht_t<uint64_t, cht_splitmix64_hash, 16, 8> t_single(q);
        for (const uint64_t k: stream) t_single.add(k);
        std::vector<uint64_t> shuffled(stream);
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        cht_t<uint64_t, cht_splitmix64_hash, 16, 8> t_simd(q);
        t_simd.add_many_simd(shuffled.begin(), shuffled.end());
        std::vector<uint64_t> c_ref(stream.size()), c_simd(stream.size());
        t_simd.count_many(stream.begin(), stream.end(), c_ref.begin());
        t_simd.count_many_simd(stream.begin(), stream.end(), c_simd.begin());
        CHECK(c_ref == c_simd, "truncated count_many_simd != count_many");
        uint64_t fn = 0;
        for (size_t i = 0; i < stream.size(); ++i) {
            fn += t_simd.count(stream[i]) == 0;
            CHECK((t_simd.count(stream[i]) != 0) == (t_single.count(stream[i]) != 0),
                  "truncated membership mismatch (single vs simd) at %zu", i);
        }
        CHECK(fn == 0, "truncated SIMD build has false negatives: %llu", (unsigned long long)fn);
        std::fprintf(stderr, "cht truncated-mode SIMD batch membership done\n");
    }
    {   // 128-bit keys: no hash4 exists, so the batch APIs must fall back to
        // the scalar chain and still match single inserts exactly
        const uint32_t q = 10; // 1024 slots, 600 distinct below
        std::mt19937_64 rng(34);
        auto mk = [](uint64_t hi, uint64_t lo) {return (u128(hi) << 64) | lo;};
        std::vector<u128> stream;
        for (unsigned i = 0; i < 600; ++i) stream.push_back(mk(rng(), rng()));
        cht_t<u128, cht_feistel128_hash, 16, 12> f_single(q);
        for (const u128 k: stream) f_single.add(k);
        cht_t<u128, cht_feistel128_hash, 16, 12> f_simd(q);
        f_simd.add_many_simd(stream.begin(), stream.end());
        CHECK(f_single.size() == f_simd.size(), "128-bit batch size mismatch");
        uint64_t bad = 0;
        for (const u128 k: stream) bad += (f_simd.count(k) != 0) != (f_single.count(k) != 0);
        CHECK(bad == 0, "128-bit batch membership mismatches: %llu", (unsigned long long)bad);
        std::fprintf(stderr, "cht 128-bit batch fallback done\n");
    }
}

// assign(): the VAL_BITS field as a general key -> value map (exact mode),
// checked against std::unordered_map: inserts, overwrites, values at the
// SAT boundary, iteration yields (key, value), add() still counts on top,
// structure_check holds, and out-of-range values throw.
static void test_assign_map() {
    const uint32_t q = 16;
    cht_t<uint64_t, cht_splitmix64_hash, 32> f(q);
    std::mt19937_64 rng(31);
    std::unordered_map<uint64_t, uint64_t> ref;
    const uint64_t SAT = decltype(f)::SAT;
    for (unsigned i = 0; i < 50000; ++i) {
        const uint64_t k = rng() % 40000;           // repeats -> overwrites
        const uint64_t v = (i % 97 == 0) ? SAT: 1 + rng() % SAT;
        f.assign(k, v); ref[k] = v;
    }
    CHECK(f.size() == ref.size(), "assign: size %llu != %llu", (unsigned long long)f.size(), (unsigned long long)ref.size());
    uint64_t bad = 0;
    for (const auto &kv: ref) bad += f.get(kv.first) != kv.second;
    CHECK(bad == 0, "assign: %llu values differ from the reference map", (unsigned long long)bad);
    CHECK(f.get(uint64_t(1) << 40) == 0, "assign: absent key reported present");
    size_t n = 0;
    for (auto it = f.begin(); it != f.end(); ++it, ++n) {
        const auto r = ref.find(*it);
        CHECK(r != ref.end() && r->second == it.mapped(), "assign: iteration yields a wrong (key, value)");
    }
    CHECK(n == ref.size(), "assign: iteration visited %zu of %zu", n, ref.size());
    CHECK(f.structure_check(), "assign: structure_check failed");
    bool threw0 = false, threw_big = false;
    try { f.assign(7, 0); } catch (const std::invalid_argument &) { threw0 = true; }
    try { f.assign(7, SAT + 1); } catch (const std::invalid_argument &) { threw_big = true; }
    CHECK(threw0 && threw_big, "assign: out-of-range values must throw");
    cht_t<uint64_t, cht_splitmix64_hash, 8> g(8);
    g.assign(5, 3); g.add(5);                        // counter semantics compose
    CHECK(g.get(5) == 4, "assign then add: got %llu, want 4", (unsigned long long)g.get(5));
    // NARROW keys (32-bit, q = 16: 16-bit remainders): run-mates with equal
    // remainders and different homes are common at this width, so the map
    // must match by home -- every key distinct, every value exact, and
    // absent keys never hit.
    {
        cht_t<uint64_t, cht_identity_hash<32>, 32> n(16);
        std::mt19937_64 r2(41);
        std::unordered_map<uint64_t, uint64_t> nref;
        while (nref.size() < 55000) {
            const uint64_t k = uint32_t(r2()), v = 1 + uint32_t(r2()) % 0xfffffffeu;
            n.assign(k, v); nref[k] = v;
        }
        CHECK(n.size() == nref.size(), "narrow assign merged keys: %llu != %zu", (unsigned long long)n.size(), nref.size());
        uint64_t nbad = 0, nfp = 0, runmates = 0;
        for (const auto &kv: nref) { nbad += n.get(kv.first) != kv.second; runmates += n.count(kv.first) != kv.second; }
        for (unsigned t = 0; t < 200000; ++t) {
            const uint64_t k = uint32_t(r2());
            if (!nref.count(k)) nfp += n.get(k) != 0;
        }
        CHECK(nbad == 0 && nfp == 0, "narrow get: %llu wrong values, %llu false positives", (unsigned long long)nbad, (unsigned long long)nfp);
        CHECK(runmates == 0, "narrow: exact-mode count() disagrees with get() on %llu keys", (unsigned long long)runmates);
        std::fprintf(stderr, "cht narrow-key map exact (%zu keys at load %.2f; count() differs on %llu)\n",
                     nref.size(), n.load_factor(), (unsigned long long)runmates);
    }
    // slot handles: find_slot agrees with get, decode(slot) recovers the key,
    // set_value_at writes through, absent keys map to nslots()
    {
        uint64_t sbad = 0;
        for (const auto &kv: ref) {
            const uint64_t i = f.find_slot(kv.first);
            sbad += i >= f.nslots() || !f.occupied_at(i) || f.decode(i) != kv.first || f.value_at(i) != kv.second;
        }
        CHECK(sbad == 0, "find_slot/decode/value_at disagree on %llu keys", (unsigned long long)sbad);
        CHECK(f.find_slot(uint64_t(1) << 40) == f.nslots(), "find_slot of an absent key must be nslots()");
        const uint64_t k0 = ref.begin()->first, i0 = f.find_slot(k0);
        f.set_value_at(i0, 12345);
        CHECK(f.get(k0) == 12345, "set_value_at did not write through");
        bool threw = false;
        try { f.set_value_at(i0, 0); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "set_value_at(…, 0) must throw");
    }
    std::fprintf(stderr, "cht assign/map done (%zu keys)\n", ref.size());
}

// cht_map_t (single-stream exact map) against std::unordered_map: wide and
// narrow keys, overwrites, near-full fill (exercises the >= 255 displacement
// spill), iteration, slot handles, batch prefetch lookups, structure_check.
template<typename M>
static void check_map_against(M &m, std::unordered_map<uint64_t, uint64_t> &ref, std::mt19937_64 &rng,
                              uint64_t keymask, const char *tag) {
    uint64_t bad = 0, fp = 0;
    for (const auto &kv: ref) {
        bad += m.get(kv.first) != kv.second;
        const uint64_t i = m.find_slot(kv.first);
        bad += i >= m.nslots() || m.decode(i) != kv.first || m.value_at(i) != kv.second;
    }
    for (unsigned t = 0; t < 100000; ++t) {
        const uint64_t k = rng() & keymask;
        if (!ref.count(k)) fp += m.get(k) != 0 || m.find_slot(k) != m.nslots();
    }
    CHECK(bad == 0 && fp == 0, "%s: %llu wrong, %llu false positives", tag, (unsigned long long)bad, (unsigned long long)fp);
    CHECK(m.size() == ref.size(), "%s: size %llu != %zu", tag, (unsigned long long)m.size(), ref.size());
    size_t n = 0;
    for (auto it = m.begin(); it != m.end(); ++it, ++n) {
        const auto r = ref.find(*it);
        CHECK(r != ref.end() && r->second == it.mapped(), "%s: iteration (key, value) wrong", tag);
    }
    CHECK(n == ref.size(), "%s: iteration visited %zu of %zu", tag, n, ref.size());
    std::vector<uint64_t> ks, got, want;
    for (unsigned t = 0; t < 20000; ++t) ks.push_back((t & 1) ? (rng() & keymask): std::next(ref.begin(), rng() % std::min<size_t>(ref.size(), 5000))->first);
    got.resize(ks.size());
    m.get_many_prefetch(ks.begin(), ks.end(), got.begin(), 16);
    for (const auto k: ks) want.push_back(m.get(k));
    CHECK(got == want, "%s: get_many_prefetch != get", tag);
    CHECK(m.structure_check(), "%s: structure_check failed", tag);
}
static void test_cht_map() {
    {   // wide keys, 32-bit values, overwrites, fill ~0.85
        cht_map_t<uint64_t, cht_splitmix64_hash, 32> m(16);
        std::mt19937_64 rng(51);
        std::unordered_map<uint64_t, uint64_t> ref;
        for (unsigned i = 0; i < 70000; ++i) {
            const uint64_t k = rng() % 56000, v = 1 + rng() % m.SAT;
            m.assign(k, v); ref[k] = v;
        }
        check_map_against(m, ref, rng, ~uint64_t(0), "cht_map wide");
        std::fprintf(stderr, "cht_map wide done (%zu keys, load %.2f, %u bits/slot)\n", ref.size(), m.load_factor(), m.cell_bits());
    }
    {   // narrow keys (32-bit identity, q = 16: 16-bit remainders), fill 0.99 -> spills
        cht_map_t<uint64_t, cht_identity_hash<32>, 26> m(16);
        std::mt19937_64 rng(52);
        std::unordered_map<uint64_t, uint64_t> ref;
        while (ref.size() < 64880) {
            const uint64_t k = uint32_t(rng()), v = 1 + rng() % m.SAT;
            m.assign(k, v); ref[k] = v;
        }
        check_map_against(m, ref, rng, 0xffffffffull, "cht_map narrow");
        CHECK(m.num_spilled() > 0, "cht_map narrow: fill 0.99 should exercise the displacement spill");
        std::fprintf(stderr, "cht_map narrow near-full done (%zu keys, load %.3f, %zu B, %zu spilled)\n", ref.size(), m.load_factor(), m.bytes_used(), m.num_spilled());
    }
    {   // write/read round trip
        cht_map_t<uint64_t, cht_splitmix64_hash, 32> m(12);
        std::mt19937_64 rng(53);
        for (unsigned i = 0; i < 3000; ++i) m.assign(rng(), 1 + rng() % m.SAT);
        std::FILE *fp = std::tmpfile();
        m.write(fp); std::rewind(fp);
        auto m2 = decltype(m)::read(fp);
        std::fclose(fp);
        bool same = m2.size() == m.size() && m2.q() == m.q();
        for (auto it = m.begin(); it != m.end(); ++it) same = same && m2.get(*it) == it.mapped();
        CHECK(same && m2.structure_check(), "cht_map: write/read round trip differs");
        bool threw = false;
        std::FILE *fp2 = std::tmpfile();
        m.write(fp2); std::rewind(fp2);
        try { (void)cht_map_t<uint64_t, cht_splitmix64_hash, 26>::read(fp2); } catch (const std::runtime_error &) { threw = true; }
        std::fclose(fp2);
        CHECK(threw, "cht_map: read with mismatched VAL_BITS must throw");
    }
    {   // errors and set_value_at
        cht_map_t<uint64_t, cht_splitmix64_hash, 8> m(4);
        bool threw0 = false, threwbig = false, threwfull = false;
        try { m.assign(1, 0); } catch (const std::invalid_argument &) { threw0 = true; }
        try { m.assign(1, 256); } catch (const std::invalid_argument &) { threwbig = true; }
        for (uint64_t k = 0; k < 16; ++k) m.assign(k, 1 + k);
        try { m.assign(99, 3); } catch (const std::overflow_error &) { threwfull = true; }
        CHECK(threw0 && threwbig && threwfull, "cht_map: range/full errors must throw");
        m.set_value_at(m.find_slot(5), 200);
        CHECK(m.get(5) == 200 && m.get(6) == 7, "cht_map: set_value_at wrong");
    }
}

// EXACT MODE COUNTS (home_exact): add()/count()/count_simd on narrow keys at
// high fill must equal reference counts exactly -- the run-mate case (equal
// remainders, different homes) that remainder-only matching merged.
static void test_exact_counts() {
    using T = cht_t<uint64_t, cht_identity_hash<32>, 16>;
    static_assert(T::home_exact && !T::exact, "identity<32>: exact on the 32-bit domain only");
    static_assert(cht_t<uint64_t, cht_splitmix64_hash, 16>::exact, "splitmix64 exact mode");
    static_assert(!cht_t<uint64_t, cht_splitmix64_hash, 16, 0, true, 0>::home_exact, "no disp array: not exact");
    const uint32_t q = 16;
    T f(q);
    std::mt19937_64 rng(61);
    std::unordered_map<uint64_t, uint64_t> ref;
    std::vector<uint64_t> stream;
    while (ref.size() < 59000) {                       // fill 0.90
        const uint64_t k = uint32_t(rng());
        const int reps = 1 + int(rng() % 3);
        for (int r = 0; r < reps; ++r) {f.add(k); ++ref[k]; stream.push_back(k);}
    }
    CHECK(f.size() == ref.size(), "exact add merged keys: %llu != %zu", (unsigned long long)f.size(), ref.size());
    uint64_t bad = 0, bads = 0, fp = 0;
    for (const auto &kv: ref) {bad += f.count(kv.first) != kv.second; bads += f.count_simd(kv.first) != kv.second;}
    for (unsigned t = 0; t < 200000; ++t) {
        const uint64_t k = uint32_t(rng());
        if (!ref.count(k)) fp += f.count(k) != 0 || f.count_simd(k) != 0;
    }
    CHECK(bad == 0 && bads == 0 && fp == 0, "exact counts: %llu wrong (scalar), %llu wrong (simd), %llu false positives",
          (unsigned long long)bad, (unsigned long long)bads, (unsigned long long)fp);
    // the sorted bulk build lands in the same exact state
    std::vector<uint64_t> sorted(stream);
    T hs(q);
    std::sort(sorted.begin(), sorted.end(), [&](uint64_t a, uint64_t b) {return hs.hash(a) < hs.hash(b);});
    T fb(q);
    fb.add_bulk_sorted(sorted.begin(), sorted.end());
    uint64_t badb = 0;
    for (const auto &kv: ref) badb += fb.count(kv.first) != kv.second;
    CHECK(fb.size() == ref.size() && badb == 0, "exact bulk build: size %llu, %llu wrong counts", (unsigned long long)fb.size(), (unsigned long long)badb);
    CHECK(f.structure_check() && fb.structure_check(), "exact counts: structure_check");
    std::fprintf(stderr, "cht exact-mode counts on 32-bit keys at load %.2f done (%zu keys, %zu adds)\n", f.load_factor(), ref.size(), stream.size());
}

int main() {
    test_exact_counts();
    test_cht_map();
    test_assign_map();
    test_hash_roundtrips();
    test_exact_vs_cqf();
    test_decode_dump();
    test_truncated_fpr();
    test_bulk_equiv();
    test_bulk_highfill();
    test_edges();
    test_bytes_accounting();
    test_batch_simd();
    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::fprintf(stderr, "all cht tests passed\n");
    return 0;
}
