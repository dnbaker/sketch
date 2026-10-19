#ifndef SKETCH_CHT_H__
#define SKETCH_CHT_H__
// Compact hash table, jellyfish-style (Marcais & Kingsford, Bioinformatics
// 2011), generic over REVERSIBLE hashes: the table is templated on
// <class Key, class RevHash, VAL_BITS, RSTORE_BITS> where RevHash is a
// BIJECTION on Key providing hash(key) (slotting/probing) and invert(h)
// (recovery of the original key). The home slot is the top q bits of
// hash(key), so each entry stores only key_bits - q remainder bits plus a
// VAL_BITS counter. Because hash() is invertible, the table is a DECODABLE
// map, jellyfish's dumper property: begin()/end()/dump_keys() reconstruct
// every original key from (home, stored remainder) via invert() and yield
// them in home-slot (storage) order.
//
// THE DISPLACEMENT ARRAY (the dumper's price, stated plainly): linear
// probing does not place colliding keys at their home, and a displaced
// key's home is NOT recoverable from (slot, remainder) alone -- nor from
// any placement discipline under colliding runs. Exact decode therefore
// records the displacement d = slot - home per slot, exactly jellyfish's
// reprobe count (which jellyfish stores IN-BAND: reprobe_len = 7 bits per
// entry at its default reprobe_limit = 126; see offsets_key_value.hpp and
// large_hash_array.hpp in gmarcais/Jellyfish). cht keeps the same
// information in a side structure: DECODE_BITS = 8 (default) allocates one
// u8 per slot -- d < 255 inline, d >= 255 marked 255 with the true value
// in a tiny sorted spill vector. The spill is small: measured at fill 0.9
// (random 48-bit keys, -O3 g++-15), max displacement is 451 at q=16, 970 at
// q=20, 1148 at q=22, with 0.07% of slots past 255 (2.7k entries at
// 2^22) -- about 8.03 bits/slot all in. Set DECODE_BITS = 0 for
// membership-only deployments: the structure is gone and
// dump_keys()/begin()/end() fail at COMPILE time. bytes_used() counts
// every array of the instantiation, so both configurations' footprints
// are directly comparable. A displacement >= 65535 needs a 64k-long probe
// run (never observed; the max at 2^22, fill 0.9 is 1148): insert throws
// std::overflow_error rather than silently corrupting decode.
//
// TRUNCATED-MODE FPR, HONESTLY: a probe compares ONLY the stored rs_-wide
// remainder, across EVERY slot in the probed span until the first empty
// one -- not just same-home slots (cht has no per-home run metadata,
// unlike cqfbase_t). The effective fingerprint of an absent key is
// therefore its remainder alone within a span of expected length
// L(a) = (1 + 1/(1-a)^2)/2, and FPR ~ 1 - exp(-L/2^rs_) at load a --
// idealized-per-(home,fp) bounds UNDERSTATE a linear-probing table by the
// probe-span factor (at a = 0.9, ~50x at q+rs fixed). fpr_bound() reports
// this clustering-aware value; chttest checks measured FPR against it.
// Same factor applies in exact mode with key_bits < 64 (full remainder
// compared across runs): L/2^key_bits.
//
// Struct-of-arrays throughout: packed remainder array, packed counter
// array, 1-bit occupancy bitmap, u8 displacement array (+ spill). Probe
// scans stay in the dense occupancy stream and touch payload only on a
// fingerprint match; displacement metadata is never on the probe path.
//
// TEMPLATE PARAMETERS
//   Key      uint64_t or unsigned __int128. Determines the arithmetic width
//            of the hash stream and of decode().
//   RevHash  bijection on Key: `Key hash(Key) const`, `Key invert(Key)
//            const`, and `static constexpr size_t key_bits` (the hash
//            width actually used; <= sizeof(Key)*CHAR_BIT). Instantiations
//            below; each is round-trip tested (hash o invert == identity)
//            in testsrc/chttest.cpp on a large random sample plus the edge
//            values 0, 1, and the domain maximum.
//   VAL_BITS counter width, packed per slot.
//   RSTORE_BITS 0 = exact mode; > 0 stores only the top RSTORE_BITS of the
//            remainder field as a fingerprint (see MODES).
//   OCC_BITMAP (default true) 1-bit occupancy bitmap; false selects the
//            zero-counter sentinel (cnt == 0 <=> empty).
//   DECODE_BITS 8 (default) = displacement side array (decode/dump on);
//            0 = membership-only, dump is a compile error.
//
// LAYOUT/MODES (mirrors cqfbase_t's contract):
//   nslots = 2^q homes == physical slots; linear probing with wraparound;
//   no overflow region and no grow -- fixed capacity, a full table throws
//   std::overflow_error from add() (bcf_t semantics). q is a runtime
//   constructor argument (2 <= q <= 40).
//   EXACT (RSTORE_BITS = 0): the full remainder (key_bits - q bits) is
//   stored. A 64-bit-or-wider hash with no truncation is exact up to the
//   open-addressing run-mate class: an absent key false-positives iff a
//   stored run-mate collides in the stored remainder, ~ used x E[miss
//   probe] / 2^key_bits (measured 3 / 3.77M at q=22, fill 0.9, key48,
//   matching the analytic 2.5-event expectation; cqfbase_t<48,...> shows 0
//   there only because its runs are per-home, the same 2^-48 class without
//   the clustering multiplier). key_bits < 64 (e.g. the identity hash on
//   reduced keys) is exact in the reduced-key fingerprint sense, exactly
//   like cqfbase_t KEY_BITS < 64.
//   TRUNCATED (RSTORE_BITS > 0): rs_ = min(RSTORE_BITS, r_). A query key
//   reports present iff some slot in its probe span stores its remainder,
//   so FPR ~ 1 - exp(-L(a)/2^rs_) with L(a) the expected unsuccessful-
//   probe span (see TRUNCATED-MODE FPR, HONESTLY above). Colliding
//   fingerprints merge into one slot and count (saturating).
//   WHAT INVERTIBILITY BUYS IN TRUNCATED MODE: decode() is still defined
//   -- invert() always exists and the displacement byte recovers the true
//   home -- but it is NOT a recovery of a stored key: fingerprint
//   collisions mean many keys share one (home, fp) pair, and decode()
//   yields the canonical representative of that pair ((home, fp) with the
//   fingerprint at the top of the remainder field and zeros below, then
//   inverted). Truncated mode is MEMBERSHIP-ONLY: the representative set
//   and order are deterministic, and every representative re-probes to its
//   own (home, fp) bucket, but those keys were never inserted. Only exact
//   mode is a decodable map.
//   KEY WIDTH CELL: every stored fingerprint lives in a u64 cell, so
//   construction throws when the stored width rs_ > 64. Consequences:
//   exact 128-bit keys would need q >= 64 (2^64 slots -- not buildable
//   within q <= 40), so unsigned __int128 keys are a TRUNCATED-mode
//   configuration; their decode yields representatives per the paragraph
//   above.
//   OCCUPANCY (knob OCC_BITMAP): bitmap probe scans are 16x denser than a
//   u16 counter stream at val16; the sentinel saves 1 bit/slot. Measured
//   on this host (q=22, key48/val16, fill 0.9, -O3 -march=native g++-15):
//   bitmap insert 1.22x, qhit 1.04x, qmiss 1.66x the sentinel, for +2.4%
//   footprint (47.8 vs 46.7 bits/entry, decode array excluded both ways) --
//   bitmap is the default.
//   PROBING: linear. Robin-hood was rejected without an A/B: displacement
//   comparison needs each slot's home, and the home is exactly what the
//   compact layout refuses to store in the payload (the decode side array
//   is dumper-only metadata, never on the probe path).
//
// SIMD (simde, PI directive 2026-10-17): every vector op in this file goes
// through the vendored simde headers (include/simde, single source, no ISA
// ifdef ladders; on AVX2 hosts simde__m256i IS __m256i and the simde
// functions inline to the native instructions). Where hash.h already
// provides an SIMD operator() whose vector type matches simde's native lane
// type (AVX2: vec::SIMDTypes<uint64_t>::Type == simde__m256i), cht_wang64_
// hash reuses it; the other hashers compose their 4-lane chain locally.
// Measurable surfaces:
//   - batch hashing: hashes_many/add_many_simd run keys through hash4()
//     (4 x u64 lanes) with a scalar tail. The hash4 lane-identity lives in
//     hash.h (hashtest); cht's own batch paths are bench-side
//     (cqf_ht_bench), not covered by the test gate.
//   - the probe: count_simd finds the probe-span end on the occupancy
//     bitmap (simde 4-word all-ones scan) and compares remainders 8 lanes
//     at a time (simde_mm256_cmpeq_epi32 + movemask_epi8 over the packed
//     rs_ <= 8 / rs_ <= 16 byte-and-word expand paths, an unpack-buffer
//     path for other packed widths, or the direct u32 stream when
//     REM_PACKED = false). Scalar-equivalent results; measured side by
//     side in benchmark/cqf_ht_bench.cpp.
//   - batch APIs: count_many / count_many_simd / count_many_sorted (input
//     sorted by hash: line reuse) / count_many_prefetch (random order with
//     simde_mm_prefetch on the home's occupancy/payload lines `ahead` keys
//     early); add_many / add_many_simd / add_many_sorted.
//   - occupancy popcounts go through cqf::detail (integral.h/libpopcnt
//     harley-seal machinery); see structure_check().
// No gather/scatter anywhere: this uarch (Coffee Lake) turns them into
// microcoded loops, and the probe's data-dependent walk does not vectorize
// that way regardless.
//
// REM_PACKED knob (default true): bit-packed rs_-wide remainders minimize
// footprint; REM_PACKED = false stores each remainder in its own u32 for
// the direct 8-lane compare (requires rs_ <= 32). The bench measures the
// crossover (unpacked wins the compare, packed wins memory).
//
// API CONTRACT: add(Key)/count(Key)/contains(Key) take the CALLER'S KEY and
// apply hf_.hash internally. For cht_identity_hash that masking is exactly
// the cqfbase_t/bcqf_t pre-hashed reduced-key contract (hash(k & mask) ==
// hash(k) & mask), so the identity instantiation is a drop-in at every
// cqfbase_t call site, including the head-to-head bench which feeds
// pre-hashed 48-bit keys (bits above key_bits may be garbage; they are
// masked off in hash()).
//
// BULK BUILD: add_bulk_sorted(first, last) consumes keys sorted by
// hf_.hash(key) ascending -- jellyfish's sorted-by-hash contract (sorting
// by hash is sorting by the (home, fingerprint) pair lexicographically,
// since both are top-bit extractions of the hash). Each home's run is
// written contiguously starting at max(home, next_empty), equal (adjacent)
// fingerprints merged into the run slot (saturating): no probing, one
// pass. The placements form one cyclically contiguous frontier, so a build
// whose tail crosses the table end wraps into the hole below the first
// home -- lookup-equivalent to add() on any order of the same multiset
// (every key sits at or after its home with all intervening slots
// occupied; the wrap preserves that cyclically). Caller contract: the
// table must be EMPTY (fresh init) and hold every distinct fingerprint;
// throws std::overflow_error when the frontier wraps the whole table.
// The bench uses fill 0.9. The displacement metadata is written from the
// same placement rule, so bulk-built tables decode exactly like
// single-insert builds.
#include "cqf.h" // detail::{packed_get,packed_set,bv_get,bv_set} reuse; contract parity with cqfbase_t
#include "simde/x86/avx2.h" // simde__m256i, cmpeq/movemask/cvt/mullo compose (single-source SIMD)
#include "simde/x86/sse.h"  // simde_mm_prefetch + SIMDE_MM_HINT_T0
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sketch {
inline namespace cht {

namespace detail {
// Shared 64-bit mixing core: the splitmix64 finalizer, REUSED from hash.h's
// audited SplitMix64Hash (identical stage chain: xorshift-multiply with two
// odd multiplies and interleaved xorshifts; hash.h carries the audited
// scalar/4-lane implementations and their inverse). Used directly by
// cht_splitmix64_hash and as the Feistel round function by
// cht_feistel128_hash.
INLINE uint64_t mix64(uint64_t z) {
    return uint64_t(SplitMix64Hash()(z));
}
// The batch-SIMD hash contract: a hasher MAY provide
//   static simde__m256i hash4(simde__m256i)   // 4 x u64 lanes, == hash() per lane
// detected here; hashes_many/add_many_simd/count_many_simd use it when Key
// is u64 and fall back to the scalar chain otherwise.
template<typename H, typename = void>
struct has_hash4: std::false_type {};
template<typename H>
struct has_hash4<H, std::void_t<decltype(H::hash4(std::declval<simde__m256i>()))>>: std::true_type {};
} // namespace detail

// ---- Reversible hash instantiations ------------------------------------
// Every instantiation must provide:
//   static constexpr size_t key_bits   -- hash width used by the table
//   Key hash(Key) const                -- forward bijection (slotting)
//   Key invert(Key) const              -- inverse (decode/dump)
// chttest.cpp round-trips each on a large random sample plus 0, 1, and the
// domain maximum.

// Identity on BITS-wide reduced keys: preserves the cqfbase_t/bcqf_t
// PRE-HASHED contract. hash() masks; invert() is the identity on the
// masked domain, so decode() recovers the REDUCED key -- the value the
// caller inserted -- never a pre-image of it. BITS < 64 makes the table
// exact in the reduced-key fingerprint sense (same class as
// cqfbase_t<KEY_BITS=48,...>); BITS = 64 is exact up to the 2^-64
// run-mate class.
template<size_t BITS>
struct cht_identity_hash {
    static_assert(BITS >= 1 && BITS <= 64, "cht_identity_hash: BITS in [1, 64]");
    static constexpr size_t key_bits = BITS;
    static constexpr uint64_t MASK = BITS >= 64 ? ~uint64_t(0): ((uint64_t(1) << BITS) - 1);
    cht_identity_hash() = default;
    template<typename...Args> cht_identity_hash(Args &&...) {}
    INLINE uint64_t hash(uint64_t v) const {return v & MASK;}
    INLINE uint64_t invert(uint64_t h) const {return h & MASK;}
    // 4-lane batch: mask the lanes (the simde face of hash(); lane-verified
    // against the scalar form in chttest)
    static simde__m256i hash4(simde__m256i v) {
        return simde_mm256_and_si256(v, simde_mm256_set1_epi64x(MASK >= (~uint64_t(0) >> 1) ? -1ll: static_cast<long long>(MASK)));
    }
};

// Invertible mixer for full 64-bit keys: hash.h's audited SplitMix64Hash
// (the splitmix64 finalizer, Sebastiano Vigna, public domain;
// http://xoshiro.di.unimi.it/splitmix64.c) composed with a +GOLDEN Weyl
// pre-add. Every stage is a mod-2^64 bijection -- translation; multiply by
// an odd constant; xorshift-right -- so hash() is a bijection and invert()
// recovers the original key exactly: SplitMix64Hash::inverse inverts the
// finalizer and the pre-add is undone by subtracting GOLDEN. This is what
// makes a cht_t over RAW 64-bit keys a decodable map with no pre-hashing on
// the caller side. The stage constants and their multiplicative inverses
// mod 2^64 (in application order) stay exposed as M1/IM1/M2/IM2; chttest
// verifies M_i * IM_i == 1.
struct cht_splitmix64_hash {
    static constexpr size_t key_bits = 64;
    static constexpr uint64_t GOLDEN = 0x9e3779b97f4a7c15ull;
    static constexpr uint64_t M1 = 0xbf58476d1ce4e5b9ull;
    static constexpr uint64_t M2 = 0x94d049bb133111ebull;
    // multiplicative inverses mod 2^64 of M2 and M1 (in application
    // order): M_i * IM_i == 1 (mod 2^64), verified in chttest.
    static constexpr uint64_t IM2 = 0x319642b2d24d8ec3ull;
    static constexpr uint64_t IM1 = 0x96de1b173f119089ull;
    cht_splitmix64_hash() = default;
    template<typename...Args> cht_splitmix64_hash(Args &&...) {}
    INLINE uint64_t hash(uint64_t x) const {return detail::mix64(x + GOLDEN);}
    INLINE uint64_t invert(uint64_t h) const {return uint64_t(SplitMix64Hash().inverse(h)) - GOLDEN;}
    // 4-lane batch: REUSES hash.h's audited SplitMix64Hash::hash4 (lane-
    // identical to its operator(), verified in hashtest) on the +GOLDEN-
    // pre-added lanes, so hash4 matches hash() bit for bit per lane -- the
    // has_hash4 contract hashes_many/add_many_simd/count_many_simd rely on.
    // (The previous local compose mis-clamped GOLDEN: a >= 2^63 guard in
    // the set1 ternary mapped it to -1, so the SIMD lanes diverged from
    // hash(). Delegating removes the duplicate chain and the bug together.)
    static simde__m256i hash4(simde__m256i z) {
        return SplitMix64Hash::hash4(simde_mm256_add_epi64(z, simde_mm256_set1_epi64x(static_cast<long long>(GOLDEN))));
    }
};

// Adapter putting hash.h's WangHash (a bijection with a published inverse,
// see hash.h) behind the cht RevHash contract. hash4 REUSES hash.h's
// existing SIMD operator() overload where its vector type lines up with
// simde's native lane type (AVX2: vec::SIMDTypes<uint64_t>::Type ==
// simde__m256i -- checked at compile time) and composes the chain from
// simde primitives otherwise; scalar hash()/invert() forward to WangHash.
// hash.h is audit-frozen for the concurrent hash-audit lane; this adapter
// (not any edit to hash.h) is the reconciliation point for cht.
struct cht_wang64_hash {
    static constexpr size_t key_bits = 64;
    cht_wang64_hash() = default;
    template<typename...Args> cht_wang64_hash(Args &&...) {}
    INLINE uint64_t hash(uint64_t v) const {return uint64_t(WangHash::hash(v));}
    INLINE uint64_t invert(uint64_t h) const {return uint64_t(WangHash().inverse(h));}
    static simde__m256i hash4(simde__m256i v) {
#if !defined(VEC_DISABLED__) && defined(__AVX2__)
        static_assert(std::is_same<vec::SIMDTypes<uint64_t>::Type, simde__m256i>::value,
                      "hash.h SIMD vector type and simde's native lane type diverged");
        return WangHash()(v); // reuse: identical chain, already lane-shaped
#else
        alignas(32) uint64_t in[4], out[4];
        simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(in), v);
        for (unsigned i = 0; i < 4; ++i) out[i] = uint64_t(WangHash::hash(in[i]));
        return simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(out));
#endif
    }
};

// 4-round Feistel on two u64 halves for 128-bit keys (unsigned __int128).
// Round function F(b, rk) = mix64(b + rk); round keys rk_i = mix64(seed +
// (i+1) * Golden). Each round (a, b) -> (b, a ^ F(b, rk)) is a bijection
// with inverse (A, B) -> (B ^ F(A, rk), A), so hash() and invert() both
// round-trip exactly for ANY F -- distribution is the point, not security
// (4 rounds against a 64-bit half is not cryptographic). Makes
// cht_t<unsigned __int128, ...> a membership table over 128-bit keys;
// exact-mode decode is unreachable for 128-bit keys (see KEY WIDTH CELL
// above), so decode yields fingerprint representatives there.
class cht_feistel128_hash {
    uint64_t rk_[4];
public:
    static constexpr size_t key_bits = 128;
    explicit cht_feistel128_hash(uint64_t seed = 0x9e3779b97f4a7c15ull) {
        for (unsigned i = 0; i < 4; ++i)
            rk_[i] = detail::mix64(seed + cht_splitmix64_hash::GOLDEN * (i + 1));
    }
    INLINE unsigned __int128 hash(unsigned __int128 x) const {
        uint64_t a = uint64_t(x >> 64), b = uint64_t(x);
        for (unsigned i = 0; i < 4; ++i) {
            const uint64_t na = b, nb = a ^ detail::mix64(b + rk_[i]);
            a = na; b = nb;
        }
        return (static_cast<unsigned __int128>(a) << 64) | b;
    }
    INLINE unsigned __int128 invert(unsigned __int128 y) const {
        uint64_t a = uint64_t(y >> 64), b = uint64_t(y);
        for (unsigned i = 4; i-- > 0;) {
            const uint64_t pa = b ^ detail::mix64(a + rk_[i]), pb = a;
            a = pa; b = pb;
        }
        return (static_cast<unsigned __int128>(a) << 64) | b;
    }
};

// ---- the table ----------------------------------------------------------
template<class Key, class RevHash, size_t VAL_BITS, size_t RSTORE_BITS=0, bool OCC_BITMAP=true, size_t DECODE_BITS=8, bool REM_PACKED=true>
class cht_t {
    static_assert(std::is_same<Key, uint64_t>::value || std::is_same<Key, unsigned __int128>::value,
                  "cht_t: Key must be uint64_t or unsigned __int128");
    static_assert(VAL_BITS >= 1 && VAL_BITS <= 64, "VAL_BITS in [1, 64]");
    static_assert(RSTORE_BITS <= 64, "the stored fingerprint cell is a u64");
    static_assert(DECODE_BITS == 0 || DECODE_BITS == 8, "DECODE_BITS is 0 (membership-only) or 8 (u8 displacement array)");
    static constexpr size_t KH_ = RevHash::key_bits; // effective hash width
    static_assert(KH_ >= 4 && KH_ <= sizeof(Key) * CHAR_BIT, "RevHash::key_bits must lie in [4, sizeof(Key)*CHAR_BIT]");
public:
    static constexpr size_t key_bits = KH_;
    static constexpr size_t val_bits = VAL_BITS;
    // exact mirrors cqfbase_t: the hash spans the key AND the full
    // remainder is stored. Truncation (RSTORE_BITS > 0) is a separate
    // fingerprint mode.
    static constexpr bool exact = (RSTORE_BITS == 0) && (KH_ >= 64);
    static constexpr uint64_t SAT = VAL_BITS >= 64 ? ~uint64_t(0): ((uint64_t(1) << VAL_BITS) - 1);
private:
    uint32_t q_ = 0, r_ = 0;
    uint32_t rs_ = 0;         // STORED fingerprint width: r_ (exact) or
                              // min(RSTORE_BITS, r_) (truncated)
    uint64_t nslots_ = 0;     // 2^q physical slots
    uint64_t used_ = 0;       // occupied slots == distinct fingerprints
    Key rm_ = 0;              // low-r_ mask (Key width)
    uint64_t fpm_ = 0;        // low-rs_ mask (cell width)
    std::vector<uint64_t, common::Allocator<uint64_t>> ocraw_, remraw_, cntraw_;
    // unpacked remainder stream (REM_PACKED = false, rs_ <= 32 only): one
    // u32 per slot for the direct 8-lane compare; packed wins memory, see
    // the header REM_PACKED note and the bench crossover rows
    std::vector<uint32_t, common::Allocator<uint32_t>> remu32_;
    // dumper-only displacement metadata (never on the probe path): d < 255
    // inline; 255 marks a spill entry (sorted by slot) holding d in u16
    std::vector<uint8_t, common::Allocator<uint8_t>> dispraw_;
    std::vector<std::pair<uint32_t, uint16_t>, common::Allocator<std::pair<uint32_t, uint16_t>>> dispill_;
    RevHash hf_;

    INLINE uint64_t rem(uint64_t i) const {
        return REM_PACKED ? cqf::detail::packed_get(remraw_.data(), i, rs_): uint64_t(remu32_[i]);
    }
    INLINE void set_rem(uint64_t i, uint64_t v) {
        if(REM_PACKED) cqf::detail::packed_set(remraw_.data(), i, rs_, v);
        else remu32_[i] = uint32_t(v);
    }
    INLINE uint64_t cnt(uint64_t i) const {return cqf::detail::packed_get(cntraw_.data(), i, unsigned(VAL_BITS));}
    INLINE bool occupied(uint64_t i) const {
        return OCC_BITMAP ? cqf::detail::bv_get(ocraw_.data(), i): cnt(i) != 0;
    }
    // displacement d = (slot - home) mod nslots at placement; read only by
    // decode(). Inline u8 covers d < 255; d >= 255 marks the byte and
    // spills (sorted-by-slot binary search). A >= 65535 displacement means
    // a 64k probe run -- never observed (max 1148 at q=22, fill 0.9).
    INLINE void disp_place(uint64_t i, uint64_t d) {
        if (DECODE_BITS) {
            if (HEDLEY_LIKELY(d < 255)) {
                dispraw_[i] = uint8_t(d);
            } else {
                if (HEDLEY_UNLIKELY(d >= 65535))
                    throw std::overflow_error("[cht] displacement exceeds 65535; decode metadata cannot represent this probe run");
                dispraw_[i] = 255;
                dispill_.insert(std::lower_bound(dispill_.begin(), dispill_.end(), std::make_pair(uint32_t(i), uint16_t(0))),
                                std::make_pair(uint32_t(i), uint16_t(d)));
            }
        }
    }
    INLINE uint64_t disp_at(uint64_t i) const {
        const uint64_t d = dispraw_[i];
        if (HEDLEY_LIKELY(d < 255)) return d;
        const auto it = std::lower_bound(dispill_.begin(), dispill_.end(), std::make_pair(uint32_t(i), uint16_t(0)));
        return it != dispill_.end() && it->first == i ? uint64_t(it->second): 255;
    }
    // payload write; liveness is the bitmap bit (bitmap mode) or the
    // nonzero counter itself (sentinel mode), so no ordering constraint
    // between the two packed arrays
    void place(uint64_t i, uint64_t f, uint64_t c) {
        set_rem(i, f);
        cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), c);
        if(OCC_BITMAP) cqf::detail::bv_set(ocraw_.data(), i);
    }
public:
    using key_type = Key;
    using mapped_type = uint64_t;
    using hasher_type = RevHash;

    explicit cht_t(uint32_t q, RevHash &&hf=RevHash()): hf_(std::move(hf)) {init(q);}
    cht_t() = delete;

    void init(uint32_t q) {
        if(q < 2 || q > 40 || KH_ - q < 1)
            throw std::invalid_argument(std::string("[cht] need 2 <= q <= 40 and 1 <= key_bits - q; got q=") + std::to_string(q));
        q_ = q; r_ = uint32_t(KH_) - q;
        rs_ = RSTORE_BITS ? unsigned(std::min<size_t>(size_t(RSTORE_BITS), size_t(r_))): r_;
        if(rs_ > 64) // u64 fingerprint cell: exact wide keys need q >= key_bits - 64
            throw std::invalid_argument(std::string("[cht] stored remainder width ") + std::to_string(rs_)
                                        + " exceeds the u64 cell; use a larger q or truncated mode (RSTORE_BITS <= 64)");
        rm_ = (Key(1) << r_) - Key(1);
        fpm_ = rs_ >= 64 ? ~uint64_t(0): ((uint64_t(1) << rs_) - 1);
        if(!REM_PACKED && rs_ > 32) // u32 remainder cells
            throw std::invalid_argument(std::string("[cht] REM_PACKED=false requires stored remainder width <= 32; got ") + std::to_string(rs_));
        nslots_ = uint64_t(1) << q;
        if(OCC_BITMAP) ocraw_.assign(ceil_div(nslots_, uint64_t(word_bits)), 0);
        else decltype(ocraw_)().swap(ocraw_);
        if(REM_PACKED) {
            remraw_.assign(ceil_div(nslots_ * rs_, uint64_t(word_bits)) + 1, 0);
            decltype(remu32_)().swap(remu32_);
        } else {
            remu32_.assign(nslots_, 0);
            decltype(remraw_)().swap(remraw_);
        }
        cntraw_.assign(VAL_BITS >= word_bits ? nslots_: ceil_div(nslots_ * VAL_BITS, uint64_t(word_bits)) + 1, 0);
        if(DECODE_BITS) {dispraw_.assign(nslots_, 0); dispill_.clear();}
        else {decltype(dispraw_)().swap(dispraw_); decltype(dispill_)().swap(dispill_);}
        used_ = 0;
    }
    uint32_t q() const {return q_;}
    uint32_t r() const {return r_;}
    uint64_t nslots() const {return nslots_;}
    uint64_t size() const {return used_;}   // distinct fingerprints
    bool empty() const {return used_ == 0;}
    double load_factor() const {return double(used_) / double(nslots_);}
    // data footprint in bytes (every array of this instantiation,
    // displacement metadata included when DECODE_BITS=8); bits/entry is
    // bytes_used()*8 / size() at the measured fill
    size_t bytes_used() const {
        return (OCC_BITMAP ? ocraw_.size(): 0) * sizeof(uint64_t)
             + (REM_PACKED ? remraw_.size() * sizeof(uint64_t): remu32_.size() * sizeof(uint32_t))
             + cntraw_.size() * sizeof(uint64_t)
             + (DECODE_BITS ? dispraw_.size() + dispill_.size() * sizeof(std::pair<uint32_t, uint16_t>): 0);
    }
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this), bytes_used());
    }
    // False-positive bound, clustering-aware (see TRUNCATED-MODE FPR,
    // HONESTLY above): an absent key collides iff its rs_-wide remainder
    // appears anywhere in its probe span of expected length
    // L(a) = (1 + 1/(1-a)^2)/2 at load a. Exact mode with key_bits >= 64
    // reports 0 (the class is L/2^64). NOTE this is NOT the same basis as
    // cqfbase_t::fpr_bound (idealized per-(home,fp)); the measured fp
    // columns in the bench are the apples-to-apples comparison.
    double fpr_bound() const {
        if (exact) return 0.;
        const double a = load_factor();
        const double L = (1. + 1. / (1. - a) / (1. - a)) / 2.;
        return 1. - std::exp(-L / std::ldexp(1., int(rs_)));
    }

    INLINE Key hash(Key v) const {return hf_.hash(v);}
    INLINE Key invert(Key h) const {return hf_.invert(h);}
    // home: top q bits of the key_bits-wide hash
    INLINE uint64_t home(Key h) const {return uint64_t(h >> r_);}
    // stored fingerprint: exact = full remainder (low r_ bits); truncated =
    // top rs_ bits of the remainder field (cqfbase_t::frs() convention)
    INLINE uint64_t fp(Key h) const {
        return rs_ == r_ ? uint64_t(h & rm_): (uint64_t(h >> (r_ - rs_)) & fpm_);
    }
    // reconstruct the caller key stored at slot i: the key's HOME (slot
    // minus the recorded displacement) as the top bits, stored fingerprint
    // re-embedded at the TOP of the remainder field with zeros below
    // (cqfbase_t::key_of convention), then invert(). Exact mode: the
    // original key. Truncated mode: the canonical representative of a
    // (home, fp) pair -- see header contract.
    INLINE Key decode(uint64_t i) const {
        static_assert(DECODE_BITS != 0, "decode/dump require DECODE_BITS=8 (membership-only table)");
        const uint64_t hpos = (i + nslots_ - disp_at(i)) & (nslots_ - 1);
        return hf_.invert((Key(hpos) << r_) | (Key(rem(i)) << (r_ - rs_)));
    }

    // single-key insert given the key's HASH (add_slot hashes). Placement
    // rule identical to add(); shared by add(), the high-fill bulk fallback,
    // and the batch add APIs.
    uint64_t add_hashed(Key h) {
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            if(!occupied(i)) {
                if(used_ == nslots_) throw std::overflow_error("[cht] table full");
                place(i, f, 1);
                disp_place(i, probe);
                ++used_; return i;
            }
            if(rem(i) == f) {
                const uint64_t c = cnt(i);
                if(c < SAT) cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), c + 1);
                return i;
            }
            if(++probe == nslots_) throw std::overflow_error("[cht] table full");
        }
    }
    // single-key insert (probing with wraparound); returns the slot the key
    // landed in (or merged at).
    uint64_t add_slot(Key k) {return add_hashed(hf_.hash(k));}
    void add(Key k) {add_slot(k);}
    // MAP USE: assign(k, v) inserts k with value v or overwrites its value,
    // in the VAL_BITS field add() uses as a counter; get(k) returns it (0 =
    // absent). v must lie in [1, SAT]: 0 is the empty value (sentinel
    // occupancy, and structure_check's live <=> nonzero invariant), so maps
    // onto a range that includes 0 store an offset (e.g. v + 1).
    //
    // HOME-EXACT MATCHING: count()/add() compare only the stored remainder
    // across the whole probe span, so two keys with different homes but equal
    // remainders merge when one is displaced into the other's run (the
    // run-mate class, ~2^-(key_bits - q) per comparison: negligible for 64-bit
    // hashes, NOT for narrow keys such as 32-bit ids at q ~ 20). The map API
    // also requires the slot's recorded displacement to equal the current
    // probe distance, i.e. the slot's key has the SAME home -- so with the full
    // remainder stored (RSTORE_BITS = 0) assign/get are exact for any
    // key_bits. The displacement byte is read only on a remainder match. Needs
    // the displacement array (DECODE_BITS = 8). Placement is add_hashed()'s,
    // so decode/iteration are unaffected; mixing assign() with add() on one
    // key is allowed but add() keeps its run-mate semantics.
    uint64_t assign(Key k, uint64_t v) {
        static_assert(DECODE_BITS != 0, "assign/get match by home and need DECODE_BITS=8");
        if(HEDLEY_UNLIKELY(v == 0 || v > SAT))
            throw std::invalid_argument("[cht] assign: value must lie in [1, 2^VAL_BITS - 1]");
        const Key h = hf_.hash(k);
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            if(!occupied(i)) {
                if(used_ == nslots_) throw std::overflow_error("[cht] table full");
                place(i, f, v);
                disp_place(i, probe);
                ++used_; return i;
            }
            if(rem(i) == f && disp_at(i) == probe) {
                cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), v);
                return i;
            }
            if(++probe == nslots_) throw std::overflow_error("[cht] table full");
        }
    }
    uint64_t get(Key k) const {return get_hashed(hf_.hash(k));}
    uint64_t get_hashed(Key h) const {
        static_assert(DECODE_BITS != 0, "assign/get match by home and need DECODE_BITS=8");
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            if(!occupied(i)) return 0;
            if(rem(i) == f && disp_at(i) == probe) return cnt(i);
            if(++probe == nslots_) return 0;
        }
    }
    // Prefetch every array a lookup at home slot hh touches: occupancy word,
    // remainder word, displacement byte, value word.
    INLINE void prefetch_home(uint64_t hh) const {
        if(OCC_BITMAP)
            simde_mm_prefetch(reinterpret_cast<const char*>(ocraw_.data() + hh / word_bits), SIMDE_MM_HINT_T0);
        if(REM_PACKED)
            simde_mm_prefetch(reinterpret_cast<const char*>(remraw_.data() + (hh * rs_) / word_bits), SIMDE_MM_HINT_T0);
        else
            simde_mm_prefetch(reinterpret_cast<const char*>(remu32_.data() + hh), SIMDE_MM_HINT_T0);
        if(DECODE_BITS)
            simde_mm_prefetch(reinterpret_cast<const char*>(dispraw_.data() + hh), SIMDE_MM_HINT_T0);
        simde_mm_prefetch(reinterpret_cast<const char*>(cntraw_.data() + (hh * VAL_BITS) / word_bits), SIMDE_MM_HINT_T0);
    }
    // Batch get (home-exact) with software prefetch: each key is hashed ONCE
    // into a small ring, its home's lines are prefetched `ahead` keys before
    // use, and results are written in input order -- identical to calling
    // get() on each key. The batch shape for lookup-bound callers (e.g. all
    // k-mers of one read).
    template<typename It, typename Ot>
    void get_many_prefetch(It f, It l, Ot o, size_t ahead = 16) const {
        static_assert(DECODE_BITS != 0, "assign/get match by home and need DECODE_BITS=8");
        constexpr size_t RING = 64;
        if(ahead > RING) ahead = RING;
        if(ahead == 0) ahead = 1;
        Key ring[RING];
        size_t issued = 0, used = 0;
        It pf = f;
        auto issue = [&]() {
            const Key h = hf_.hash(Key(*pf++));
            prefetch_home(home(h));
            ring[issued++ % RING] = h;
        };
        while(pf != l && issued < ahead) issue();
        for(; f != l; ++f, ++o) {
            const Key h = ring[used++ % RING];
            if(pf != l) issue();
            *o = get_hashed(h);
        }
    }
    // SLOT-LEVEL MAP ACCESS (home-exact like get): slots are stable -- linear
    // probing without deletion never moves a placed entry -- so a slot index
    // is a durable handle, e.g. for storing references between entries (a
    // tree's parent pointer as the parent's slot). find_slot returns nslots()
    // when k is absent; value_at/set_value_at read/write the VAL_BITS field of
    // an OCCUPIED slot (set_value_at's value must lie in [1, SAT]); decode(i)
    // recovers the slot's key.
    uint64_t find_slot(Key k) const {
        static_assert(DECODE_BITS != 0, "find_slot matches by home and needs DECODE_BITS=8");
        const Key h = hf_.hash(k);
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            if(!occupied(i)) return nslots_;
            if(rem(i) == f && disp_at(i) == probe) return i;
            if(++probe == nslots_) return nslots_;
        }
    }
    bool occupied_at(uint64_t i) const {return i < nslots_ && occupied(i);}
    uint64_t value_at(uint64_t i) const {return cnt(i);}
    void set_value_at(uint64_t i, uint64_t v) {
        if(HEDLEY_UNLIKELY(v == 0 || v > SAT || !occupied_at(i)))
            throw std::invalid_argument("[cht] set_value_at: occupied slot and a value in [1, 2^VAL_BITS - 1] required");
        cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), v);
    }
    uint64_t count(Key k) const {
        const Key h = hf_.hash(k);
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            if(!occupied(i)) return 0;
            if(rem(i) == f) return cnt(i);
            if(++probe == nslots_) return 0;
        }
    }
    bool contains(Key k) const {return count(k) != 0;}
    bool may_contain(Key k) const {return contains(k);}

    // ---- batch APIs (see the SIMD header section) ----
    // batch-hash [f,l) through the reversible chain: 4 u64 lanes at a time
    // via RevHash::hash4 when Key is u64 and the iterators are random
    // access, scalar tail/otherwise. OutIt receives uint64 hash values.
    template<typename It, typename Ot>
    void hashes_many(It f, It l, Ot o) const {
        if constexpr(sizeof(Key) == 8 && detail::has_hash4<RevHash>::value
                     && std::is_same<typename std::iterator_traits<It>::iterator_category,
                                     std::random_access_iterator_tag>::value) {
            for(; l - f >= 4; f += 4, o += 4) {
                const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(&*f));
                simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(&*o), RevHash::hash4(v));
            }
        }
        for(; f != l; ++f, ++o) *o = uint64_t(hf_.hash(Key(*f)));
    }
    // scalar batch count (one-by-one probe); the reference the SIMD paths
    // are checked against
    template<typename It, typename Ot>
    void count_many(It f, It l, Ot o) const {for(; f != l; ++f, ++o) *o = count(Key(*f));}
    // sorted-order batch count: input sorted by hf_.hash(key) ascending (the
    // add_bulk_sorted order); consecutive keys share probe-span cache lines.
    // Same results as count_many.
    template<typename It, typename Ot>
    void count_many_sorted(It f, It l, Ot o) const {for(; f != l; ++f, ++o) *o = count(Key(*f));}
    // random-order batch count with software prefetch: `ahead` keys before
    // use, issue simde_mm_prefetch on the home's occupancy + payload words.
    template<typename It, typename Ot>
    void count_many_prefetch(It f, It l, Ot o, size_t ahead = 8) const {
        It pf = f;
        for(; f != l; ++f, ++o) {
            for(size_t d = 0; d < ahead && pf != l; ++d, ++pf) {
                const uint64_t hh = home(hf_.hash(Key(*pf)));
                if(OCC_BITMAP)
                    simde_mm_prefetch(reinterpret_cast<const char*>(ocraw_.data() + hh / word_bits), SIMDE_MM_HINT_T0);
                if(REM_PACKED)
                    simde_mm_prefetch(reinterpret_cast<const char*>(remraw_.data() + (hh * rs_) / word_bits), SIMDE_MM_HINT_T0);
                else
                    simde_mm_prefetch(reinterpret_cast<const char*>(remu32_.data() + hh), SIMDE_MM_HINT_T0);
                simde_mm_prefetch(reinterpret_cast<const char*>(cntraw_.data() + (hh * VAL_BITS) / word_bits), SIMDE_MM_HINT_T0);
            }
            *o = count(Key(*f));
        }
    }
    // scalar batch insert; add_many_sorted takes the same sorted input as
    // add_bulk_sorted (line reuse; identical placement to add())
    template<typename It>
    void add_many(It f, It l) {for(; f != l; ++f) add(Key(*f));}
    template<typename It>
    void add_many_sorted(It f, It l) {for(; f != l; ++f) add(Key(*f));}
    // SIMD batch insert: batch-hash via hash4, then the identical probe from
    // the hash (add_hashed)
    template<typename It>
    void add_many_simd(It f, It l) {
        if constexpr(sizeof(Key) == 8 && detail::has_hash4<RevHash>::value
                     && std::is_same<typename std::iterator_traits<It>::iterator_category,
                                     std::random_access_iterator_tag>::value) {
            for(; l - f >= 4; f += 4) {
                alignas(32) uint64_t hv[4];
                const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(&*f));
                simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(hv), RevHash::hash4(v));
                add_hashed(Key(hv[0])); add_hashed(Key(hv[1])); add_hashed(Key(hv[2])); add_hashed(Key(hv[3]));
            }
        }
        for(; f != l; ++f) add(Key(*f));
    }

    // ---- SIMD probe (simde; scalar-equivalent to count) ----
    // first empty slot at/after i == the probe-span end. Bitmap mode scans
    // the occupancy words four at a time (cmpeq_epi64 against all-ones +
    // movemask_pd to spot a word holding a zero bit, ctz inside it);
    // sentinel mode scans the counter stream. Returns nslots_ when nothing
    // is empty.
    uint64_t probe_end(uint64_t i) const {
        if(!OCC_BITMAP) {
            for(; i < nslots_ && cnt(i) != 0; ++i);
            return i;
        }
        const uint64_t nw = ceil_div(nslots_, uint64_t(word_bits));
        const uint64_t wi0 = i / word_bits;
        uint64_t wi = wi0;
        const simde__m256i ones = simde_mm256_set1_epi64x(-1);
        for(; wi + 4 <= nw; wi += 4) {
            const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(ocraw_.data() + wi));
            const unsigned full = unsigned(simde_mm256_movemask_pd(simde_mm256_castsi256_pd(simde_mm256_cmpeq_epi64(v, ones))));
            if(full != 15u) break; // a word in [wi, wi+4) holds an empty bit
        }
        for(; wi < nw; ++wi) {
            uint64_t w = ~ocraw_[wi];
            // first word: occupancy bits below the query index belong to
            // slots BEFORE i -- mask them off. probe_end(i) must be >= i; an
            // empty earlier in the same word ends a DIFFERENT probe's span.
            if(wi == wi0 && i % word_bits) w &= ~((uint64_t(1) << (i % word_bits)) - 1);
            if(wi == nw - 1 && nslots_ % word_bits) w &= (uint64_t(1) << (nslots_ % word_bits)) - 1; // phantom tail bits
            if(w) return wi * word_bits + ctz(w);
        }
        return nslots_;
    }
    // scalar remainder scan over the half-open slot range [lo, hi)
    uint64_t scan_rem_scalar(uint64_t lo, uint64_t hi, uint64_t f) const {
        for(uint64_t i = lo; i < hi; ++i) if(rem(i) == f) return i;
        return ~uint64_t(0);
    }
    // simde remainder scan over [lo, hi): 8 u32 lanes per step
    // (simde_mm256_cmpeq_epi32 against the broadcast fingerprint +
    // movemask_epi8; hit lane = ctz(movemask) >> 2). Packed remainders use
    // the byte/word expand paths at rs_ <= 8 / rs_ <= 16 and a scalar
    // unpack-buffer for other widths; REM_PACKED=false reads the u32 stream
    // directly. rs_ > 32 falls back to the scalar scan.
    uint64_t scan_rem_simd(uint64_t lo, uint64_t hi, uint64_t f) const {
        if(hi <= lo) return ~uint64_t(0);
        if(rs_ > 32) return scan_rem_scalar(lo, hi, f);
        const simde__m256i fv = simde_mm256_set1_epi32(static_cast<int>(f));
        uint64_t i = lo;
        if(REM_PACKED) {
            if(rs_ == 8) {
                for(; i + 8 <= hi; i += 8) {
                    const simde__m128i b = simde_mm_loadl_epi64(reinterpret_cast<const simde__m128i*>(
                        reinterpret_cast<const uint8_t*>(remraw_.data()) + i));
                    const unsigned m = unsigned(simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(simde_mm256_cvtepu8_epi32(b), fv)));
                    if(m) return i + (ctz(m) >> 2);
                }
            } else if(rs_ == 16) {
                for(; i + 8 <= hi; i += 8) {
                    const simde__m128i w = simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(
                        reinterpret_cast<const uint8_t*>(remraw_.data()) + 2 * i));
                    const unsigned m = unsigned(simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(simde_mm256_cvtepu16_epi32(w), fv)));
                    if(m) return i + (ctz(m) >> 2);
                }
            } else {
                for(; i + 8 <= hi; i += 8) {
                    alignas(32) uint32_t buf[8];
                    for(unsigned k = 0; k < 8; ++k) buf[k] = uint32_t(rem(i + k));
                    const unsigned m = unsigned(simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(simde_mm256_load_si256(reinterpret_cast<const simde__m256i*>(buf)), fv)));
                    if(m) return i + (ctz(m) >> 2);
                }
            }
        } else {
            for(; i + 8 <= hi; i += 8) {
                const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(remu32_.data() + i));
                const unsigned m = unsigned(simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(v, fv)));
                if(m) return i + (ctz(m) >> 2);
            }
        }
        return scan_rem_scalar(i, hi, f);
    }
    // SIMD count given the key's hash: find the probe-span end, compare
    // remainders in the span, and only when the span reaches the table end
    // without an empty slot continue the probe cyclically (scalar; the
    // table is then near-full). Identical results to count().
    uint64_t count_simd_hash(Key h) const {
        const uint64_t f = fp(h);
        const uint64_t h0 = home(h);
        const uint64_t e = probe_end(h0);
        const uint64_t s = scan_rem_simd(h0, e, f);
        if(s != ~uint64_t(0)) return cnt(s);
        if(e < nslots_) return 0;
        for(uint64_t i = 0;; i = (i + 1) & (nslots_ - 1)) { // cyclically continued probe
            if(!occupied(i)) return 0;
            if(rem(i) == f) return cnt(i);
        }
    }
    uint64_t count_simd(Key k) const {return count_simd_hash(hf_.hash(k));}
    // SIMD batch count: batch-hash via hash4 + per-key SIMD probe.
    // Identical results to count_many (chttest asserts).
    template<typename It, typename Ot>
    void count_many_simd(It f, It l, Ot o) const {
        if constexpr(sizeof(Key) == 8 && detail::has_hash4<RevHash>::value
                     && std::is_same<typename std::iterator_traits<It>::iterator_category,
                                     std::random_access_iterator_tag>::value) {
            for(; l - f >= 4; f += 4, o += 4) {
                alignas(32) uint64_t hv[4];
                const simde__m256i v = simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(&*f));
                simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(hv), RevHash::hash4(v));
                o[0] = count_simd_hash(Key(hv[0])); o[1] = count_simd_hash(Key(hv[1]));
                o[2] = count_simd_hash(Key(hv[2])); o[3] = count_simd_hash(Key(hv[3]));
            }
        }
        for(; f != l; ++f, ++o) *o = count(Key(*f));
    }

    // ---- test/debug helper ----
    // Verifies the table invariants and returns false on the first
    // violation: (1) the occupancy bitmap total (via the shared popcount
    // machinery, cqf::detail::popcount_words -> integral.h/libpopcnt
    // harley-seal) equals used_; (2) bitmap liveness and counter liveness
    // agree, with 0 < cnt <= SAT on every live slot; (3) sampled decode
    // round-trips: re-hashing a decoded key reproduces the slot's (home,
    // fingerprint). O(nslots/64 + sample).
    bool structure_check(size_t sample = 1000) const {
        if(OCC_BITMAP && cqf::detail::popcount_words(ocraw_.data(), ocraw_.size()) != used_) return false;
        for(uint64_t i = 0; i < nslots_; ++i) {
            const bool live = occupied(i);
            if(live != (cnt(i) != 0)) return false;
            if(live && cnt(i) > SAT) return false;
        }
        if(DECODE_BITS && used_) {
            std::mt19937_64 rng(97);
            for(size_t t = 0; t < sample; ++t) {
                const uint64_t i = rng() % nslots_;
                if(!occupied(i)) continue;
                const Key h = hf_.hash(decode(i));
                if(home(h) != ((i + nslots_ - disp_at(i)) & (nslots_ - 1)) || fp(h) != rem(i)) return false;
            }
        }
        return true;
    }


    // ---- decode iteration (the jellyfish dumper property) ----
    // Walks slots in home-slot (storage) order; operator* decodes the
    // caller key at the current slot. Exact mode: the inserted keys.
    // Truncated mode: canonical representatives, NOT the inserted keys
    // (membership-only; see header contract).
    class const_iterator {
        const cht_t *t_ = nullptr;
        uint64_t i_ = 0;
        void skip() {while(i_ < t_->nslots_ && !t_->occupied(i_)) ++i_;}
    public:
        static_assert(DECODE_BITS != 0, "iteration requires DECODE_BITS=8 (membership-only table)");
        const_iterator() = default;
        const_iterator(const cht_t *t, uint64_t i): t_(t), i_(i) {skip();}
        bool operator==(const const_iterator &o) const {return t_ == o.t_ && i_ == o.i_;}
        bool operator!=(const const_iterator &o) const {return !(*this == o);}
        const_iterator &operator++() {++i_; skip(); return *this;}
        const_iterator operator++(int) {const_iterator tmp(*this); ++*this; return tmp;}
        Key operator*() const {return t_->decode(i_);}
        uint64_t slot() const {return i_;}      // physical slot (debug/tests)
        uint64_t mapped() const {return t_->cnt(i_);} // counter at the slot
    };
    const_iterator begin() const {return const_iterator(this, 0);}
    const_iterator end() const {return const_iterator(this, nslots_);}
    // dump decoded keys in home-slot order (dumper)
    template<typename OutIt>
    void dump_keys(OutIt out) const {for(const_iterator it = begin(); it != end(); ++it) *out++ = *it;}
    // dump (key, count) pairs in home-slot order
    template<typename OutIt>
    void dump(OutIt out) const {for(const_iterator it = begin(); it != end(); ++it) *out++ = std::make_pair(*it, it.mapped());}

    // Bulk build from keys sorted by hf_.hash(key) ascending --
    // jellyfish's sorted-by-hash contract; see the header block. Runs are
    // written contiguously at max(home, next_empty) with equal
    // fingerprints merged into the run slot (saturating). Placements form
    // one cyclically contiguous frontier (absolute counter, slots mod
    // nslots), so a build whose tail crosses the table end wraps into the
    // hole below the first home. While the frontier stays below that wrap
    // the build is PLACEMENT-IDENTICAL to add() fed the same sorted stream
    // (induction on the first-empty invariant: every probe's intervening
    // slots are occupied, and holes sit behind every later home, so neither
    // add() nor the frontier ever places into one). When a run no longer
    // fits cyclically below the first home -- possible while ordinary holes
    // still exist, e.g. the first home is ~0 and the displacement creep
    // pushes the frontier a few slots past the table end -- the remaining
    // keys fall back to per-key probing (add_slot), which is exactly add()
    // on the same stream: add() fills those holes and succeeds wherever the
    // table still has room, so the bulk path throws only when the table is
    // genuinely full (used_ == nslots_), never at a fill factor. Measured
    // at fill 0.9/0.95, q=20, key48: the fallback absorbs the last few
    // groups only (test_bulk_highfill). Caller contract: the table must be
    // EMPTY (fresh init). The displacement byte is written from the same
    // placement rule (d = absolute distance), so wrapped slots decode
    // exactly.
    template<typename It>
    void add_bulk_sorted(It first, It last) {
        if(first == last) return;
        uint64_t next_empty = home(hf_.hash(Key(*first))); // absolute; first run starts at its home
        const uint64_t first_home = next_empty;
        while(first != last) {
            const uint64_t h = home(hf_.hash(Key(*first)));
            It gend = first;
            while(gend != last && home(hf_.hash(Key(*gend))) == h) ++gend;
            uint64_t i = h >= next_empty ? h: next_empty;
            uint64_t runlen = 0;  // distinct fingerprints in [first, gend)
            for(It j = first; j != gend;) {
                ++runlen;
                const uint64_t f = fp(hf_.hash(Key(*j)));
                do { ++j; } while(j != gend && fp(hf_.hash(Key(*j))) == f);
            }
            if(i + runlen - first_home > nslots_) {
                // no cyclic room below the first home for this run: probe
                // the rest per key, exactly add() on the same stream
                for(; first != last; ++first) add_slot(Key(*first));
                return;
            }
            for(It j = first; j != gend;) {
                const uint64_t f = fp(hf_.hash(Key(*j)));
                uint64_t c = 0;
                It k = j;
                for(; k != gend && fp(hf_.hash(Key(*k))) == f; ++k)
                    if(c < SAT) ++c;
                place(i & (nslots_ - 1), f, c);
                disp_place(i & (nslots_ - 1), i - h);
                ++used_; ++i;
                j = k;
            }
            next_empty = i;
            first = gend;
        }
    }
};

// ---------------------------------------------------------------------------
// cht_map_t: exact key -> value map, SINGLE-STREAM layout.
//
// The same compact-hash idea as cht_t: a reversible hash, home = the top q
// bits, and only the r = key_bits - q remainder bits stored. But every slot
// is ONE packed cell, [ remainder (r) | value (VAL_BITS) | displacement (8) ],
// in a single bit stream with value 0 = empty (no occupancy bitmap).
//
// A probe step is then one unaligned 16-byte load, and a lookup usually
// touches one cache line. cht_t's struct-of-arrays layout touches four on a
// hit (bitmap, remainder, displacement, value). Bits per slot are the same
// as cht_t minus its occupancy bit.
//
// Matching is home-exact, as in cht_t::get: remainder equal AND recorded
// displacement == probe distance. So with any key_bits there are no merged
// keys and no false positives. Displacements >= 255 spill to a sorted side
// vector (rare at fill <= 0.9), exactly like cht_t.
//
// Values must lie in [1, 2^VAL_BITS - 1]; store an offset for ranges that
// include 0. Fixed capacity (no grow); insert throws std::overflow_error
// when full. Slots are stable, so a slot index is a durable handle.
// Not for concurrent writers.
template<class Key, class RevHash, size_t VAL_BITS>
class cht_map_t {
    static_assert(std::is_same<Key, uint64_t>::value, "cht_map_t: Key must be uint64_t");
    static_assert(VAL_BITS >= 1 && VAL_BITS <= 48, "VAL_BITS in [1, 48]");
    static constexpr size_t KH_ = RevHash::key_bits;
    static_assert(KH_ >= 4 && KH_ <= 64, "RevHash::key_bits must lie in [4, 64]");
    using u128 = unsigned __int128;
public:
    static constexpr size_t key_bits = KH_;
    static constexpr size_t val_bits = VAL_BITS;
    static constexpr uint64_t SAT = (uint64_t(1) << VAL_BITS) - 1;
private:
    uint32_t q_ = 0, r_ = 0, w_ = 0;   // w_ = cell width r_ + VAL_BITS + 8
    uint64_t nslots_ = 0, used_ = 0, rm_ = 0;
    u128 cm_ = 0;                      // low-w_ mask
    std::vector<uint8_t, common::Allocator<uint8_t>> bits_;
    std::vector<std::pair<uint64_t, uint32_t>, common::Allocator<std::pair<uint64_t, uint32_t>>> spill_;
    RevHash hf_;

    INLINE u128 cell(uint64_t i) const {
        const uint64_t off = i * w_;
        u128 x;
        std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
        return (x >> (off & 7)) & cm_;
    }
    INLINE void set_cell(uint64_t i, u128 v) {
        const uint64_t off = i * w_;
        const unsigned s = unsigned(off & 7);
        u128 x;
        std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
        x = (x & ~(cm_ << s)) | ((v & cm_) << s);
        std::memcpy(bits_.data() + (off >> 3), &x, sizeof(x));
    }
    INLINE uint64_t c_rem(u128 c) const {return uint64_t(c) & rm_;}
    INLINE uint64_t c_val(u128 c) const {return uint64_t(c >> r_) & SAT;}
    INLINE uint64_t c_disp(u128 c) const {return uint64_t(c >> (r_ + VAL_BITS)) & 0xFFu;}
    INLINE u128 make_cell(uint64_t f, uint64_t v, uint64_t d8) const {
        return u128(f) | (u128(v) << r_) | (u128(d8) << (r_ + VAL_BITS));
    }
    uint64_t spilled_disp(uint64_t i) const {
        const auto it = std::lower_bound(spill_.begin(), spill_.end(), std::make_pair(i, uint32_t(0)));
        return it != spill_.end() && it->first == i ? uint64_t(it->second): 255;
    }
    INLINE uint64_t disp_of(uint64_t i, u128 c) const {
        const uint64_t d = c_disp(c);
        return HEDLEY_LIKELY(d < 255) ? d: spilled_disp(i);
    }
    // place a new entry at empty slot i, `probe` slots from its home
    void place_new(uint64_t i, uint64_t f, uint64_t v, uint64_t probe) {
        if(HEDLEY_LIKELY(probe < 255)) {
            set_cell(i, make_cell(f, v, probe));
        } else {
            if(HEDLEY_UNLIKELY(probe > 0xFFFFFFFFull))
                throw std::overflow_error("[cht_map] displacement exceeds 2^32");
            set_cell(i, make_cell(f, v, 255));
            spill_.insert(std::lower_bound(spill_.begin(), spill_.end(), std::make_pair(i, uint32_t(0))),
                          std::make_pair(i, uint32_t(probe)));
        }
        ++used_;
    }
public:
    using key_type = Key;
    using mapped_type = uint64_t;
    using hasher_type = RevHash;

    explicit cht_map_t(uint32_t q, RevHash &&hf=RevHash()): hf_(std::move(hf)) {init(q);}
    cht_map_t() = delete;

    void init(uint32_t q) {
        if(q < 2 || q > 40 || KH_ <= q)
            throw std::invalid_argument(std::string("[cht_map] need 2 <= q <= 40 and q < key_bits; got q=") + std::to_string(q));
        q_ = q; r_ = uint32_t(KH_) - q; w_ = r_ + uint32_t(VAL_BITS) + 8;
        if(w_ > 121) // one 16-byte load at a byte offset covers 121 bits
            throw std::invalid_argument("[cht_map] cell width (key_bits - q + VAL_BITS + 8) exceeds 121 bits");
        rm_ = r_ >= 64 ? ~uint64_t(0): ((uint64_t(1) << r_) - 1);
        cm_ = (u128(1) << w_) - 1;
        nslots_ = uint64_t(1) << q;
        bits_.assign(ceil_div(nslots_ * w_, uint64_t(8)) + 16, 0);
        spill_.clear();
        used_ = 0;
    }
    uint32_t q() const {return q_;}
    uint32_t r() const {return r_;}
    uint32_t cell_bits() const {return w_;}
    uint64_t nslots() const {return nslots_;}
    uint64_t size() const {return used_;}
    bool empty() const {return used_ == 0;}
    double load_factor() const {return double(used_) / double(nslots_);}
    size_t bytes_used() const {return bits_.size() + spill_.size() * sizeof(spill_[0]);}
    size_t num_spilled() const {return spill_.size();}   // entries displaced >= 255 slots

    INLINE Key hash(Key v) const {return hf_.hash(v);}
    INLINE Key invert(Key h) const {return hf_.invert(h);}
    INLINE uint64_t home(Key h) const {return uint64_t(h >> r_);}
    INLINE uint64_t fp(Key h) const {return uint64_t(h) & rm_;}

    // insert k with value v in [1, SAT], or overwrite its value
    uint64_t assign(Key k, uint64_t v) {
        if(HEDLEY_UNLIKELY(v == 0 || v > SAT))
            throw std::invalid_argument("[cht_map] assign: value must lie in [1, 2^VAL_BITS - 1]");
        const Key h = hf_.hash(k);
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            const u128 c = cell(i);
            if(c_val(c) == 0) {
                if(used_ == nslots_) throw std::overflow_error("[cht_map] table full");
                place_new(i, f, v, probe);
                return i;
            }
            if(c_rem(c) == f && disp_of(i, c) == probe) {
                set_cell(i, (c & ~(u128(SAT) << r_)) | (u128(v) << r_));
                return i;
            }
            if(++probe == nslots_) throw std::overflow_error("[cht_map] table full");
        }
    }
    // slot of k, or nslots() when absent
    uint64_t find_slot_hashed(Key h) const {
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            const u128 c = cell(i);
            if(c_val(c) == 0) return nslots_;
            if(c_rem(c) == f && disp_of(i, c) == probe) return i;
            if(++probe == nslots_) return nslots_;
        }
    }
    uint64_t find_slot(Key k) const {return find_slot_hashed(hf_.hash(k));}
    uint64_t get_hashed(Key h) const {
        const uint64_t f = fp(h);
        uint64_t i = home(h);
        for(uint64_t probe = 0;; i = (i + 1) & (nslots_ - 1)) {
            const u128 c = cell(i);
            const uint64_t v = c_val(c);
            if(v == 0) return 0;
            if(c_rem(c) == f && disp_of(i, c) == probe) return v;
            if(++probe == nslots_) return 0;
        }
    }
    uint64_t get(Key k) const {return get_hashed(hf_.hash(k));}
    bool contains(Key k) const {return get(k) != 0;}
    bool occupied_at(uint64_t i) const {return i < nslots_ && c_val(cell(i)) != 0;}
    uint64_t value_at(uint64_t i) const {return c_val(cell(i));}
    void set_value_at(uint64_t i, uint64_t v) {
        if(HEDLEY_UNLIKELY(v == 0 || v > SAT || !occupied_at(i)))
            throw std::invalid_argument("[cht_map] set_value_at: occupied slot and a value in [1, 2^VAL_BITS - 1] required");
        const u128 c = cell(i);
        set_cell(i, (c & ~(u128(SAT) << r_)) | (u128(v) << r_));
    }
    // the key stored at occupied slot i
    Key decode(uint64_t i) const {
        const u128 c = cell(i);
        const uint64_t hpos = (i + nslots_ - disp_of(i, c)) & (nslots_ - 1);
        return hf_.invert((Key(hpos) << r_) | Key(c_rem(c)));
    }
    INLINE void prefetch_home(uint64_t hh) const {
        simde_mm_prefetch(reinterpret_cast<const char*>(bits_.data() + ((hh * w_) >> 3)), SIMDE_MM_HINT_T0);
    }
    // batch get with software prefetch, results in input order (== get each)
    template<typename It, typename Ot>
    void get_many_prefetch(It f, It l, Ot o, size_t ahead = 16) const {
        constexpr size_t RING = 64;
        if(ahead > RING) ahead = RING;
        if(ahead == 0) ahead = 1;
        Key ring[RING];
        size_t issued = 0, used = 0;
        It pf = f;
        auto issue = [&]() {
            const Key h = hf_.hash(Key(*pf++));
            prefetch_home(home(h));
            ring[issued++ % RING] = h;
        };
        while(pf != l && issued < ahead) issue();
        for(; f != l; ++f, ++o) {
            const Key h = ring[used++ % RING];
            if(pf != l) issue();
            *o = get_hashed(h);
        }
    }

    class const_iterator {
        const cht_map_t *t_ = nullptr;
        uint64_t i_ = 0;
        void skip() {while(i_ < t_->nslots_ && !t_->occupied_at(i_)) ++i_;}
    public:
        const_iterator() = default;
        const_iterator(const cht_map_t *t, uint64_t i): t_(t), i_(i) {skip();}
        bool operator==(const const_iterator &o) const {return t_ == o.t_ && i_ == o.i_;}
        bool operator!=(const const_iterator &o) const {return !(*this == o);}
        const_iterator &operator++() {++i_; skip(); return *this;}
        Key operator*() const {return t_->decode(i_);}
        uint64_t slot() const {return i_;}
        uint64_t mapped() const {return t_->value_at(i_);}
    };
    const_iterator begin() const {return const_iterator(this, 0);}
    const_iterator end() const {return const_iterator(this, nslots_);}

    // Binary serialization: magic "CHTMAP01", key_bits, VAL_BITS, q, size,
    // the cell stream and the displacement spill. RevHash must be stateless
    // (or default-constructed identically) on both sides; read() checks the
    // key/value widths against this instantiation and throws on mismatch.
    void write(std::FILE *fp) const {
        static const char magic[8] = {'C','H','T','M','A','P','0','1'};
        const uint32_t hdr[3] = {uint32_t(KH_), uint32_t(VAL_BITS), q_};
        const uint64_t nb = bits_.size(), ns = spill_.size();
        if(std::fwrite(magic, 1, 8, fp) != 8 || std::fwrite(hdr, sizeof(hdr), 1, fp) != 1
           || std::fwrite(&used_, sizeof(used_), 1, fp) != 1 || std::fwrite(&nb, sizeof(nb), 1, fp) != 1
           || std::fwrite(bits_.data(), 1, nb, fp) != nb || std::fwrite(&ns, sizeof(ns), 1, fp) != 1
           || (ns && std::fwrite(spill_.data(), sizeof(spill_[0]), ns, fp) != ns))
            throw std::runtime_error("[cht_map] write failed");
    }
    static cht_map_t read(std::FILE *fp) {
        char magic[8];
        uint32_t hdr[3];
        uint64_t used, nb, ns;
        if(std::fread(magic, 1, 8, fp) != 8 || std::memcmp(magic, "CHTMAP01", 8))
            throw std::runtime_error("[cht_map] read: bad magic");
        if(std::fread(hdr, sizeof(hdr), 1, fp) != 1 || hdr[0] != KH_ || hdr[1] != VAL_BITS)
            throw std::runtime_error("[cht_map] read: key/value widths do not match this instantiation");
        cht_map_t ret(hdr[2]);
        if(std::fread(&used, sizeof(used), 1, fp) != 1 || std::fread(&nb, sizeof(nb), 1, fp) != 1 || nb != ret.bits_.size())
            throw std::runtime_error("[cht_map] read: truncated or inconsistent header");
        if(std::fread(ret.bits_.data(), 1, nb, fp) != nb || std::fread(&ns, sizeof(ns), 1, fp) != 1)
            throw std::runtime_error("[cht_map] read: truncated cell stream");
        ret.spill_.resize(ns);
        if(ns && std::fread(ret.spill_.data(), sizeof(ret.spill_[0]), ns, fp) != ns)
            throw std::runtime_error("[cht_map] read: truncated spill");
        ret.used_ = used;
        return ret;
    }
    // every occupied slot re-probes to itself from its decoded key's home
    bool structure_check() const {
        uint64_t n = 0;
        for(uint64_t i = 0; i < nslots_; ++i) {
            if(!occupied_at(i)) continue;
            ++n;
            if(find_slot(decode(i)) != i) return false;
        }
        return n == used_;
    }
};

} // namespace cht
} // namespace sketch

#endif // SKETCH_CHT_H__
