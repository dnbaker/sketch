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
// Exact mode does NOT have this class (see EXACT below): there a match also
// requires the slot's recorded displacement, i.e. the same home.
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
//   EXACT (RSTORE_BITS = 0 with DECODE_BITS = 8: `home_exact`): the full
//   remainder (key_bits - q bits) is stored, and every match -- add, count,
//   get, assign, find_slot, the SIMD span scan -- also requires the slot's
//   recorded displacement to equal the probe distance, i.e. the stored key
//   has the SAME home. (home, remainder) is the whole hash value, so
//   distinct keys never merge and absent keys never hit, for ANY key_bits:
//   exact on the key_bits-wide domain. key_bits >= 64 (`exact`) is exact on
//   raw 64-bit keys. key_bits < 64 (e.g. the identity hash on reduced keys)
//   is exact on the reduced keys, like cqfbase_t KEY_BITS < 64. The cost is
//   one displacement-byte read per remainder match, never per probe step.
//   (Before 2026-10-19, matching compared the remainder alone across the
//   span, so run-mates with different homes merged: 3 / 3.77M false
//   positives at q=22, fill 0.9, key48. At a 32-bit key width this was 11
//   merged keys of 200k at q=18.) With DECODE_BITS = 0 there is no
//   displacement to check, and matching is remainder-only (membership
//   filter semantics).
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
#include "policy.h" // policy::SizeDivPolicy (schism fastdiv) for non-power-of-two table sizes
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

// Invertible mixer on a RUNTIME-width domain [0, 2^bits), bits in [4, 64]:
// the splitmix64 finalizer stage chain (Weyl pre-add; xorshift-right;
// multiply by an odd constant; xorshift; multiply; xorshift), every stage
// taken mod 2^bits. Each stage is a bijection on the bits-wide domain: an odd
// multiplier is invertible mod 2^bits (its inverse mod 2^64, reduced), and
// x ^= x >> s is undone by repeated unshifting. Shifts scale with the width
// (30/27/31 at 64 bits: identical to splitmix64's stages). For keys that
// only span `bits` bits -- e.g. 2-bit-encoded k-mers, bits = 2k -- the table
// then stores bits - q remainder bits instead of 64 - q. key_bits = 0 marks
// the width as runtime (cht_core reads bits()). Inputs must lie in the domain.
namespace detail {
constexpr uint64_t inv_mul64(uint64_t a) {uint64_t x = a; for(int i = 0; i < 6; ++i) x *= 2 - a * x; return x;}  // a odd
} // namespace detail
class cht_mixbits_hash {
    static constexpr uint64_t GOLDEN = 0x9e3779b97f4a7c15ull;
    static constexpr uint64_t C1 = 0xbf58476d1ce4e5b9ull, C2 = 0x94d049bb133111ebull;
    static constexpr uint64_t I1 = detail::inv_mul64(C1), I2 = detail::inv_mul64(C2);
    static_assert(C1 * I1 == 1 && C2 * I2 == 1, "modular inverses");
    uint32_t b_;
    uint64_t m_;
    unsigned s1_, s2_, s3_;
    INLINE uint64_t unshift(uint64_t y, unsigned s) const {   // inverse of x ^= x >> s on b_ bits
        uint64_t x = y;
        for(unsigned k = s; k < b_; k += s) x = y ^ (x >> s);
        return x;
    }
public:
    static constexpr size_t key_bits = 0;   // runtime: bits()
    explicit cht_mixbits_hash(unsigned bits = 64): b_(bits), m_(bits >= 64 ? ~uint64_t(0): ((uint64_t(1) << bits) - 1)),
        s1_(std::max(1u, bits * 30 / 64)), s2_(std::max(1u, bits * 27 / 64)), s3_(std::max(1u, bits * 31 / 64)) {
        if(bits < 4 || bits > 64) throw std::invalid_argument("[cht] cht_mixbits_hash: bits in [4, 64]");
    }
    size_t bits() const {return b_;}
    uint64_t mask() const {return m_;}
    INLINE uint64_t hash(uint64_t x) const {
        x = (x + GOLDEN) & m_;
        x ^= x >> s1_; x = (x * C1) & m_;
        x ^= x >> s2_; x = (x * C2) & m_;
        x ^= x >> s3_;
        return x;
    }
    INLINE uint64_t invert(uint64_t h) const {
        uint64_t x = h & m_;
        x = unshift(x, s3_); x = (x * I2) & m_;
        x = unshift(x, s2_); x = (x * I1) & m_;
        x = unshift(x, s1_);
        return (x - GOLDEN) & m_;
    }
};

// Invertible mixer on a RUNTIME-width domain [0, 2^bits), bits in [65, 128],
// for unsigned __int128 keys (e.g. 2-bit k-mers with 32 < k <= 64, or
// rolling hashes wider than 64 bits): a 4-round UNBALANCED Feistel on (hi =
// the top bits - 64 bits, lo = the low 64 bits). Rounds alternate
// lo ^= mix64(hi ^ rk) and hi = (hi ^ mix64(lo ^ rk)) mod 2^(bits - 64); each
// is its own inverse, so invert() replays them in reverse. Any round
// function gives a bijection; mix64 (splitmix64) gives the distribution.
// key_bits = 0 marks the width as runtime (cht_core reads bits()).
class cht_mixbits128_hash {
    using u128 = unsigned __int128;
    uint32_t b_;
    uint64_t mhi_;
    uint64_t rk_[4];
public:
    static constexpr size_t key_bits = 0;   // runtime: bits()
    explicit cht_mixbits128_hash(unsigned bits = 128, uint64_t seed = 0x9e3779b97f4a7c15ull): b_(bits),
        mhi_(bits >= 128 ? ~uint64_t(0): ((uint64_t(1) << (bits - 64)) - 1)) {
        if(bits < 65 || bits > 128) throw std::invalid_argument("[cht] cht_mixbits128_hash: bits in [65, 128] (use cht_mixbits_hash below)");
        for(unsigned i = 0; i < 4; ++i) rk_[i] = detail::mix64(seed + 0x9e3779b97f4a7c15ull * (i + 1));
    }
    size_t bits() const {return b_;}
    INLINE u128 hash(u128 x) const {
        uint64_t lo = uint64_t(x), hi = uint64_t(x >> 64) & mhi_;
        lo ^= detail::mix64(hi ^ rk_[0]);
        hi = (hi ^ detail::mix64(lo ^ rk_[1])) & mhi_;
        lo ^= detail::mix64(hi ^ rk_[2]);
        hi = (hi ^ detail::mix64(lo ^ rk_[3])) & mhi_;
        return (u128(hi) << 64) | lo;
    }
    INLINE u128 invert(u128 y) const {
        uint64_t lo = uint64_t(y), hi = uint64_t(y >> 64) & mhi_;
        hi = (hi ^ detail::mix64(lo ^ rk_[3])) & mhi_;
        lo ^= detail::mix64(hi ^ rk_[2]);
        hi = (hi ^ detail::mix64(lo ^ rk_[1])) & mhi_;
        lo ^= detail::mix64(hi ^ rk_[0]);
        return (u128(hi) << 64) | lo;
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

// ---- shared core ---------------------------------------------------------
//
// cht_core is the ONE implementation of the compact-hash table, shared by
// two storage specializations:
//   cht_t      struct-of-arrays storage (occupancy bitmap, packed remainder,
//              packed value, u8 displacement as separate arrays): counting,
//              truncated/membership-filter modes, 128-bit keys, the SIMD span
//              scan and the sorted bulk build;
//   cht_map_t  single-stream storage (one packed cell per slot:
//              [remainder | value | displacement], value 0 = empty): the
//              exact key -> value map, where a lookup touches one cache line.
// The core holds the hashing (home = top q bits, stored fingerprint), the
// probe loops with home-exact matching (one rule for add/count/assign/get/
// find_slot), the displacement spill, decode/iteration, prefetching batch
// lookups and the structure check. A storage policy supplies per-slot
// primitives through a load handle: cht_t's handle is the slot index (lazy:
// a probe step reads occupancy and remainder only), cht_map_t's is the loaded
// cell (one load serves every field). Both are zero-cost.
//
// Storage hooks a Derived class provides (private; Derived befriends the core):
//   void     init_storage_()                    size arrays for nslots_/rs_
//   size_t   storage_bytes_() const             bytes of the storage arrays
//   H        load_(uint64_t i) const            handle for slot i
//   bool     h_occ_(uint64_t i, H) const        slot live
//   uint64_t h_rem_(uint64_t i, H) const        stored fingerprint
//   uint64_t h_val_(uint64_t i, H) const        value / counter
//   uint64_t h_disp8_(uint64_t i, H) const      inline displacement byte
//   void     place_(uint64_t i, uint64_t f, uint64_t v, uint64_t d8)   write a NEW live slot
//   void     set_val_(uint64_t i, uint64_t v)
//   void     set_disp8_(uint64_t i, uint64_t d8)
//   void     prefetch_(uint64_t home) const
//   bool     extra_check_() const               storage-specific invariants
// ---- table sizing --------------------------------------------------------
// The hash h (key_bits wide) is split into a home slot and a stored
// remainder by multiply-high ("fastrange"): with n slots,
//     P = h * n,  home = P >> key_bits,  rem = (P mod 2^key_bits) div n,
// which is exact -- rem = h - ceil(home * 2^key_bits / n), so
// h = lo(home) + rem with lo(home) = ceil(home * 2^key_bits / n) (decode) --
// keeps homes monotone in h (the sorted bulk build relies on it), and stores
// rem in r = ceil(log2(ceil(2^key_bits / n))) bits: a fixed width per table,
// though no longer a bit field of h.
//   cht_pow2_sizing  n = 2^q: the split above IS home = top q bits, rem = low
//                    key_bits - q bits (shift/mask; the default; constructors
//                    take q).
//   cht_div_sizing   any n in [4, 2^40] (key_bits <= 64): one 64x64 -> 128
//                    multiply plus a schism fastdiv (policy::SizeDivPolicy)
//                    per hash; probe wraparound by compare. Sizes the table
//                    to the data instead of the next power of two.
// Construct with cht_slots{n} to give the slot count directly.
struct cht_pow2_sizing {static constexpr bool pow2 = true;};
struct cht_div_sizing {static constexpr bool pow2 = false;};
struct cht_slots {uint64_t n;};

namespace detail {

template<class Derived, class Key, class RevHash, size_t VAL_BITS, size_t RSTORE_BITS, size_t DECODE_BITS, class Sizing = cht_pow2_sizing,
         class Rem = uint64_t>
class cht_core {
    static_assert(std::is_same<Rem, uint64_t>::value || std::is_same<Rem, unsigned __int128>::value, "Rem: uint64_t or unsigned __int128");
    static_assert(std::is_same<Key, uint64_t>::value || std::is_same<Key, unsigned __int128>::value,
                  "cht: Key must be uint64_t or unsigned __int128");
    static_assert(VAL_BITS <= 64, "VAL_BITS in [1, 64], or 0 = chosen at construction (runtime width)");
    static_assert(RSTORE_BITS <= 64, "the stored fingerprint cell is a u64");
    static_assert(DECODE_BITS == 0 || DECODE_BITS == 8, "DECODE_BITS is 0 (membership-only) or 8 (u8 displacement array)");
protected:
    static constexpr size_t KH_ = RevHash::key_bits; // effective hash width; 0 = runtime (hasher's bits())
    static_assert(KH_ == 0 || (KH_ >= 4 && KH_ <= sizeof(Key) * CHAR_BIT), "RevHash::key_bits must lie in [4, sizeof(Key)*CHAR_BIT], or 0 (runtime)");
    static_assert(KH_ != 0 || std::is_same<Key, uint64_t>::value || std::is_same<Key, unsigned __int128>::value,
                  "runtime key widths need uint64_t or unsigned __int128 keys");
    static_assert(Sizing::pow2 || KH_ <= 64, "non-power-of-two sizing needs key_bits <= 64");
public:
    using sizing_type = Sizing;
    using rem_type = Rem;     // stored fingerprint type (u128 for remainders wider than 64 bits)
    static constexpr size_t key_bits = KH_;           // 0: runtime width, see kbits()
    static constexpr bool runtime_key_bits = KH_ == 0;
    static constexpr size_t val_bits = VAL_BITS;     // 0: runtime width, see vbits()
    static constexpr bool runtime_val_bits = VAL_BITS == 0;
    // home_exact: the full remainder is stored AND displacements are kept,
    // so every match (add/count/get/assign/find_slot and the SIMD scan)
    // also requires the slot's recorded displacement == the probe distance
    // (same home). Distinct keys of the key_bits-wide hash domain never
    // merge and absent keys never hit: exact on that domain, at one
    // displacement read per remainder match. exact: home_exact over the
    // full 64-bit key (key_bits >= 64). DECODE_BITS = 0 or truncation keeps
    // the remainder-only (run-mate class) matching of a membership filter.
    static constexpr bool home_exact = (RSTORE_BITS == 0) && (DECODE_BITS != 0);
    static constexpr bool exact = home_exact && (KH_ >= 64);
    // compile-time saturation value (0 when the width is runtime: use sat())
    static constexpr uint64_t SAT = VAL_BITS == 0 ? 0: VAL_BITS >= 64 ? ~uint64_t(0): ((uint64_t(1) << VAL_BITS) - 1);
    using key_type = Key;
    using mapped_type = uint64_t;
    using hasher_type = RevHash;

protected:
    uint32_t q_ = 0, r_ = 0;
    uint32_t rs_ = 0;         // STORED fingerprint width: r_ (exact) or min(RSTORE_BITS, r_) (truncated)
    uint64_t nslots_ = 0;     // 2^q physical slots
    uint64_t used_ = 0;       // occupied slots == distinct fingerprints
    Key rm_ = 0;              // low-r_ mask (Key width)
    Rem fpm_ = 0;             // low-rs_ mask
    policy::SizeDivPolicy<uint64_t> div_{uint64_t(4)};   // fastdiv by nslots_ (cht_div_sizing)
    uint32_t kh_ = uint32_t(KH_);        // key (hash) width (runtime when key_bits == 0)
    uint32_t vb_ = uint32_t(VAL_BITS);   // value width (runtime when VAL_BITS == 0)
    uint64_t sat_ = SAT;                 // 2^vb_ - 1
    // displacements >= 255 (inline byte 255): (slot, displacement), sorted by slot
    std::vector<std::pair<uint64_t, uint32_t>, common::Allocator<std::pair<uint64_t, uint32_t>>> dispill_;
    RevHash hf_;

    Derived &d() {return static_cast<Derived &>(*this);}
    const Derived &d() const {return static_cast<const Derived &>(*this);}
    explicit cht_core(RevHash &&hf): hf_(std::move(hf)) {
        if constexpr(runtime_key_bits) {
            kh_ = uint32_t(hf_.bits());
            if(kh_ < 4 || kh_ > sizeof(Key) * CHAR_BIT)
                throw std::invalid_argument("[cht] runtime key width must lie in [4, key type width]");
            if(!Sizing::pow2 && kh_ > 64) throw std::invalid_argument("[cht] non-power-of-two sizing needs key_bits <= 64");
        }
    }
    // runtime value width (VAL_BITS == 0 only); call before init_core
    void set_val_bits_(uint32_t vb, uint32_t max_vb) {
        if(!runtime_val_bits) {
            if(vb != VAL_BITS) throw std::invalid_argument("[cht] value width differs from the compile-time VAL_BITS");
            return;
        }
        if(vb < 1 || vb > max_vb)
            throw std::invalid_argument(std::string("[cht] runtime value width must lie in [1, ") + std::to_string(max_vb) + "]; got " + std::to_string(vb));
        vb_ = vb;
        sat_ = vb >= 64 ? ~uint64_t(0): ((uint64_t(1) << vb) - 1);
    }

    void init_core(uint32_t q) {
        if(q < 2 || q > 40 || kh_ <= q)
            throw std::invalid_argument(std::string("[cht] need 2 <= q <= 40 and 1 <= key_bits - q; got q=") + std::to_string(q));
        init_core_n(uint64_t(1) << q);
    }
    void init_core_n(uint64_t n) {
        uint32_t q = 0;
        while(q < 63 && (uint64_t(1) << q) < n) ++q;            // ceil(log2 n)
        if(n < 4 || n > (uint64_t(1) << 40))
            throw std::invalid_argument(std::string("[cht] need 4 <= nslots <= 2^40; got ") + std::to_string(n));
        if constexpr(Sizing::pow2) {
            if(n & (n - 1))
                throw std::invalid_argument("[cht] cht_pow2_sizing needs a power-of-two slot count; use cht_div_sizing");
            if(kh_ <= q) throw std::invalid_argument("[cht] need 1 <= key_bits - q");
            r_ = kh_ - q;
        } else {
            // remainder bound ceil(2^key_bits / n) - 1 -> its bit width
            const unsigned __int128 span = (((unsigned __int128)1 << kh_) + n - 1) / n;
            const unsigned __int128 mx = span - 1;
            r_ = 0;
            while(r_ < 64 && (mx >> r_) != 0) ++r_;
            if(r_ == 0) r_ = 1;
            div_ = policy::SizeDivPolicy<uint64_t>(n);
        }
        q_ = q;
        rs_ = RSTORE_BITS ? unsigned(std::min<size_t>(size_t(RSTORE_BITS), size_t(r_))): r_;
        if(rs_ > sizeof(Rem) * CHAR_BIT) // fingerprint cell width: exact wide keys need q >= key_bits - width
            throw std::invalid_argument(std::string("[cht] stored remainder width ") + std::to_string(rs_)
                                        + " exceeds the fingerprint cell; use a larger q, truncated mode (RSTORE_BITS <= 64), or cht_map_t");
        rm_ = (Key(1) << r_) - Key(1);
        fpm_ = rs_ >= sizeof(Rem) * CHAR_BIT ? ~Rem(0): ((Rem(1) << rs_) - 1);
        nslots_ = n;
        dispill_.clear();
        used_ = 0;
        d().init_storage_();
    }

    // ---- displacement: inline byte, >= 255 spilled -------------------------
    template<typename H>
    INLINE uint64_t disp_h(uint64_t i, const H &c) const {
        const uint64_t d8 = d().h_disp8_(i, c);
        if(HEDLEY_LIKELY(d8 < 255)) return d8;
        const auto it = std::lower_bound(dispill_.begin(), dispill_.end(), std::make_pair(i, uint32_t(0)));
        return it != dispill_.end() && it->first == i ? uint64_t(it->second): 255;
    }
    INLINE uint64_t disp_at(uint64_t i) const {return disp_h(i, d().load_(i));}
    void disp_place(uint64_t i, uint64_t dd) {
        if(!DECODE_BITS) return;
        if(HEDLEY_LIKELY(dd < 255)) {
            d().set_disp8_(i, dd);
        } else {
            if(HEDLEY_UNLIKELY(dd > 0xFFFFFFFFull))
                throw std::overflow_error("[cht] displacement exceeds 2^32; decode metadata cannot represent this probe run");
            d().set_disp8_(i, 255);
            dispill_.insert(std::lower_bound(dispill_.begin(), dispill_.end(), std::make_pair(i, uint32_t(0))),
                            std::make_pair(i, uint32_t(dd)));
        }
    }
    // a new live slot in ONE storage write (payload + inline displacement
    // byte); only displacements >= 255 touch the spill
    void place_new(uint64_t i, Rem f, uint64_t v, uint64_t probe) {
        if(HEDLEY_LIKELY(!DECODE_BITS || probe < 255)) {
            d().place_(i, f, v, DECODE_BITS ? probe: 0);
        } else {
            if(HEDLEY_UNLIKELY(probe > 0xFFFFFFFFull))
                throw std::overflow_error("[cht] displacement exceeds 2^32; decode metadata cannot represent this probe run");
            d().place_(i, f, v, 255);
            dispill_.insert(std::lower_bound(dispill_.begin(), dispill_.end(), std::make_pair(i, uint32_t(0))),
                            std::make_pair(i, uint32_t(probe)));
        }
        ++used_;
    }
    // the matching rule: remainder AND (home_exact or MapApi) recorded displacement == probe distance
    template<bool MapApi, typename H>
    INLINE bool match(uint64_t i, const H &c, Rem f, uint64_t probe) const {
        return d().h_rem_(i, c) == f && ((!home_exact && !MapApi) || disp_h(i, c) == probe);
    }

public:
    void init(uint32_t q) {init_core(q);}
    uint32_t q() const {return q_;}
    uint32_t r() const {return r_;}
    uint32_t kbits() const {return kh_;}       // key (hash) width in bits
    uint32_t vbits() const {return vb_;}       // value width in bits
    uint64_t sat() const {return sat_;}        // largest storable value
    uint64_t nslots() const {return nslots_;}
    uint64_t size() const {return used_;}   // distinct fingerprints
    bool empty() const {return used_ == 0;}
    double load_factor() const {return double(used_) / double(nslots_);}
    size_t num_spilled() const {return dispill_.size();}   // entries displaced >= 255 slots
    // data footprint in bytes: every storage array plus the displacement spill
    size_t bytes_used() const {
        return d().storage_bytes_() + (DECODE_BITS ? dispill_.size() * sizeof(dispill_[0]): 0);
    }
    std::pair<size_t, size_t> est_memory_usage() const {return std::make_pair(sizeof(Derived), bytes_used());}

    INLINE Key hash(Key v) const {return hf_.hash(v);}
    INLINE Key invert(Key h) const {return hf_.invert(h);}
    // split a hash into (home slot, full remainder) -- see "table sizing"
    INLINE void split(Key h, uint64_t &hh, Rem &rr) const {
        if constexpr(Sizing::pow2) {
            hh = uint64_t(h >> r_);
            rr = Rem(h & rm_);
        } else {
            const unsigned __int128 P = (unsigned __int128)uint64_t(h) * nslots_;
            hh = uint64_t(P >> kh_);
            const uint64_t low = kh_ >= 64 ? uint64_t(P): uint64_t(P & ((((unsigned __int128)1) << kh_) - 1));
            rr = div_.div(low);
        }
    }
    // the hash with home hh and remainder field rr (inverse of split; Key
    // width, since 128-bit keys have remainders wider than 64 bits)
    INLINE Key compose(uint64_t hh, Key rr) const {
        if constexpr(Sizing::pow2) {
            return (Key(hh) << r_) | rr;
        } else {
            const unsigned __int128 lo = ((((unsigned __int128)hh) << kh_) + nslots_ - 1) / nslots_;
            return Key(uint64_t(lo) + rr);
        }
    }
    // stored fingerprint from the full remainder: exact = all of it;
    // truncated = its top rs_ bits (cqfbase_t::frs() convention)
    INLINE Rem fp_of_rem(Rem rr) const {return rs_ == r_ ? rr: (rr >> (r_ - rs_)) & fpm_;}
    // probe start for hash h: home slot and stored fingerprint. pow2 works
    // in Key width (the remainder of a 128-bit key exceeds 64 bits; the
    // stored fingerprint never does).
    INLINE void probe_start(Key h, uint64_t &i, Rem &f) const {
        if constexpr(Sizing::pow2) {
            i = uint64_t(h >> r_);
            f = rs_ == r_ ? Rem(h & rm_): (Rem(h >> (r_ - rs_)) & fpm_);
        } else {
            Rem rr;
            split(h, i, rr);
            f = fp_of_rem(rr);
        }
    }
    INLINE uint64_t next_slot(uint64_t i) const {
        if constexpr(Sizing::pow2) return (i + 1) & (nslots_ - 1);
        else return i + 1 == nslots_ ? 0: i + 1;
    }
    // (i - d) mod nslots for 0 <= d < nslots
    INLINE uint64_t back(uint64_t i, uint64_t dd) const {
        if constexpr(Sizing::pow2) return (i + nslots_ - dd) & (nslots_ - 1);
        else return dd <= i ? i - dd: i + nslots_ - dd;
    }
    // home slot of hash h
    INLINE uint64_t home(Key h) const {uint64_t i; Rem f; probe_start(h, i, f); return i;}
    // stored fingerprint of hash h
    INLINE Rem fp(Key h) const {uint64_t i; Rem f; probe_start(h, i, f); return f;}
    // reconstruct the caller key stored at slot i: home (slot minus recorded
    // displacement) as the top bits, the stored fingerprint at the TOP of the
    // remainder field with zeros below, then invert(). Exact mode: the
    // original key. Truncated mode: the canonical (home, fp) representative.
    INLINE Key decode(uint64_t i) const {
        static_assert(DECODE_BITS != 0, "decode/dump require DECODE_BITS=8 (membership-only table)");
        const auto c = d().load_(i);
        const uint64_t hpos = back(i, disp_h(i, c));
        return hf_.invert(compose(hpos, Key(d().h_rem_(i, c)) << (r_ - rs_)));
    }

    // ---- counting --------------------------------------------------------
    // insert given the key's HASH; returns the slot it landed in or merged at
    uint64_t add_hashed(Key h) {
        uint64_t i; Rem f;
        probe_start(h, i, f);
        for(uint64_t probe = 0;; i = next_slot(i)) {
            const auto c = d().load_(i);
            if(!d().h_occ_(i, c)) {
                if(used_ == nslots_) throw std::overflow_error("[cht] table full");
                place_new(i, f, 1, probe);
                return i;
            }
            if(match<false>(i, c, f, probe)) {
                const uint64_t v = d().h_val_(i, c);
                if(v < sat_) d().set_val_(i, v + 1);
                return i;
            }
            if(++probe == nslots_) throw std::overflow_error("[cht] table full");
        }
    }
    uint64_t add_slot(Key k) {return add_hashed(hf_.hash(k));}
    void add(Key k) {add_slot(k);}
    uint64_t count_hashed(Key h) const {
        uint64_t i; Rem f;
        probe_start(h, i, f);
        for(uint64_t probe = 0;; i = next_slot(i)) {
            const auto c = d().load_(i);
            if(!d().h_occ_(i, c)) return 0;
            if(match<false>(i, c, f, probe)) return d().h_val_(i, c);
            if(++probe == nslots_) return 0;
        }
    }
    uint64_t count(Key k) const {return count_hashed(hf_.hash(k));}
    bool contains(Key k) const {return count(k) != 0;}
    bool may_contain(Key k) const {return contains(k);}

    // ---- map API (always home-matched; exact when home_exact) -------------
    // assign(k, v): insert k with value v in [1, SAT] or overwrite its value;
    // get(k) returns it (0 = absent). 0 is the empty value, so maps onto a
    // range that includes 0 store an offset (e.g. v + 1). add() still counts
    // on top of an assigned value.
    uint64_t assign(Key k, uint64_t v) {
        static_assert(DECODE_BITS != 0, "assign/get match by home and need DECODE_BITS=8");
        if(HEDLEY_UNLIKELY(v == 0 || v > sat_))
            throw std::invalid_argument("[cht] assign: value must lie in [1, 2^vbits - 1]");
        const Key h = hf_.hash(k);
        uint64_t i; Rem f;
        probe_start(h, i, f);
        for(uint64_t probe = 0;; i = next_slot(i)) {
            const auto c = d().load_(i);
            if(!d().h_occ_(i, c)) {
                if(used_ == nslots_) throw std::overflow_error("[cht] table full");
                place_new(i, f, v, probe);
                return i;
            }
            if(match<true>(i, c, f, probe)) {d().set_val_(i, v); return i;}
            if(++probe == nslots_) throw std::overflow_error("[cht] table full");
        }
    }
    // slot of the key with hash h, or nslots() when absent; slots are stable
    // (no deletion, linear probing never moves an entry), so a slot index is
    // a durable handle
    uint64_t find_slot_hashed(Key h) const {
        static_assert(DECODE_BITS != 0, "find_slot matches by home and needs DECODE_BITS=8");
        uint64_t i; Rem f;
        probe_start(h, i, f);
        for(uint64_t probe = 0;; i = next_slot(i)) {
            const auto c = d().load_(i);
            if(!d().h_occ_(i, c)) return nslots_;
            if(match<true>(i, c, f, probe)) return i;
            if(++probe == nslots_) return nslots_;
        }
    }
    uint64_t find_slot(Key k) const {return find_slot_hashed(hf_.hash(k));}
    uint64_t get_hashed(Key h) const {
        static_assert(DECODE_BITS != 0, "assign/get match by home and need DECODE_BITS=8");
        uint64_t i; Rem f;
        probe_start(h, i, f);
        for(uint64_t probe = 0;; i = next_slot(i)) {
            const auto c = d().load_(i);
            if(!d().h_occ_(i, c)) return 0;
            if(match<true>(i, c, f, probe)) return d().h_val_(i, c);
            if(++probe == nslots_) return 0;
        }
    }
    uint64_t get(Key k) const {return get_hashed(hf_.hash(k));}
    bool occupied_at(uint64_t i) const {return i < nslots_ && d().h_occ_(i, d().load_(i));}
    uint64_t value_at(uint64_t i) const {return d().h_val_(i, d().load_(i));}
    void set_value_at(uint64_t i, uint64_t v) {
        if(HEDLEY_UNLIKELY(v == 0 || v > sat_ || !occupied_at(i)))
            throw std::invalid_argument("[cht] set_value_at: occupied slot and a value in [1, 2^vbits - 1] required");
        d().set_val_(i, v);
    }

    // ---- batches ---------------------------------------------------------
    INLINE void prefetch_home(uint64_t hh) const {d().prefetch_(hh);}
    // Batch with software prefetch: each key hashed ONCE into a ring, its
    // home's lines prefetched `ahead` keys before use; results in input
    // order, identical to the scalar call per key.
    template<bool Get, typename It, typename Ot>
    void many_prefetch_(It f, It l, Ot o, size_t ahead) const {
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
            if constexpr(Get) *o = get_hashed(h);
            else *o = count_hashed(h);
        }
    }
    template<typename It, typename Ot>
    void get_many_prefetch(It f, It l, Ot o, size_t ahead = 16) const {many_prefetch_<true>(f, l, o, ahead);}
    template<typename It, typename Ot>
    void count_many_prefetch(It f, It l, Ot o, size_t ahead = 8) const {many_prefetch_<false>(f, l, o, ahead);}
    // scalar batch count (one-by-one probe); the reference batch paths are checked against
    template<typename It, typename Ot>
    void count_many(It f, It l, Ot o) const {for(; f != l; ++f, ++o) *o = count(Key(*f));}
    // sorted-order batch count: input sorted by hash ascending (the
    // add_bulk_sorted order); consecutive keys share probe-span lines
    template<typename It, typename Ot>
    void count_many_sorted(It f, It l, Ot o) const {for(; f != l; ++f, ++o) *o = count(Key(*f));}
    template<typename It>
    void add_many(It f, It l) {for(; f != l; ++f) add(Key(*f));}
    template<typename It>
    void add_many_sorted(It f, It l) {for(; f != l; ++f) add(Key(*f));}

    // ---- iteration -------------------------------------------------------
    class const_iterator {
        const Derived *t_ = nullptr;
        uint64_t i_ = 0;
        void skip() {while(i_ < t_->nslots() && !t_->occupied_at(i_)) ++i_;}
    public:
        static_assert(DECODE_BITS != 0, "iteration requires DECODE_BITS=8 (membership-only table)");
        const_iterator() = default;
        const_iterator(const Derived *t, uint64_t i): t_(t), i_(i) {skip();}
        bool operator==(const const_iterator &o) const {return t_ == o.t_ && i_ == o.i_;}
        bool operator!=(const const_iterator &o) const {return !(*this == o);}
        const_iterator &operator++() {++i_; skip(); return *this;}
        const_iterator operator++(int) {const_iterator tmp(*this); ++*this; return tmp;}
        Key operator*() const {return t_->decode(i_);}
        uint64_t slot() const {return i_;}                 // physical slot
        uint64_t mapped() const {return t_->value_at(i_);} // value/counter at the slot
    };
    const_iterator begin() const {return const_iterator(&d(), 0);}
    const_iterator end() const {return const_iterator(&d(), nslots_);}
    template<typename OutIt>
    void dump_keys(OutIt out) const {for(const_iterator it = begin(); it != end(); ++it) *out++ = *it;}
    template<typename OutIt>
    void dump(OutIt out) const {for(const_iterator it = begin(); it != end(); ++it) *out++ = std::make_pair(*it, it.mapped());}

    // ---- test/debug helper ------------------------------------------------
    // Returns false on the first violated invariant: storage-specific checks
    // (extra_check_), liveness <=> nonzero value with value <= SAT on every
    // slot, live count == size(), and on `sample` sampled live slots (every
    // slot when sample == 0) the decoded key re-hashes to the slot's (home,
    // fingerprint) -- and, home_exact, re-probes to exactly that slot.
    bool structure_check(size_t sample = 1000) const {
        if(!d().extra_check_()) return false;
        uint64_t live = 0;
        for(uint64_t i = 0; i < nslots_; ++i) {
            const auto c = d().load_(i);
            const bool occ = d().h_occ_(i, c);
            const uint64_t v = d().h_val_(i, c);
            if(occ != (v != 0)) return false;
            if(occ && v > sat_) return false;
            live += occ;
        }
        if(live != used_) return false;
        if constexpr(DECODE_BITS != 0) {
            if(!used_) return true;
            auto check_slot = [&](uint64_t i) {
                const Key h = hf_.hash(decode(i));
                if(home(h) != back(i, disp_at(i)) || fp(h) != d().h_rem_(i, d().load_(i))) return false;
                if constexpr(home_exact) if(find_slot_hashed(h) != i) return false;
                return true;
            };
            if(sample == 0) {
                for(uint64_t i = 0; i < nslots_; ++i) if(occupied_at(i) && !check_slot(i)) return false;
            } else {
                std::mt19937_64 rng(97);
                for(size_t t = 0; t < sample; ++t) {
                    const uint64_t i = rng() % nslots_;
                    if(occupied_at(i) && !check_slot(i)) return false;
                }
            }
        }
        return true;
    }
};

} // namespace detail

// ---- cht_t: struct-of-arrays storage -------------------------------------
// TEMPLATE PARAMETERS and modes: see the header comment (Key, RevHash,
// VAL_BITS, RSTORE_BITS, OCC_BITMAP, DECODE_BITS, REM_PACKED). Adds to the
// core: truncated-mode fpr_bound, batch hashing (hash4), the SIMD span scan
// (count_simd*), SIMD batch insert and the sorted bulk build.
template<class Key, class RevHash, size_t VAL_BITS, size_t RSTORE_BITS=0, bool OCC_BITMAP=true, size_t DECODE_BITS=8, bool REM_PACKED=true,
         class Sizing=cht_pow2_sizing>
class cht_t: public detail::cht_core<cht_t<Key, RevHash, VAL_BITS, RSTORE_BITS, OCC_BITMAP, DECODE_BITS, REM_PACKED, Sizing>,
                                     Key, RevHash, VAL_BITS, RSTORE_BITS, DECODE_BITS, Sizing> {
    static_assert(VAL_BITS >= 1, "cht_t: VAL_BITS in [1, 64] (runtime widths are cht_map_t only)");
    using Base = detail::cht_core<cht_t, Key, RevHash, VAL_BITS, RSTORE_BITS, DECODE_BITS, Sizing>;
    friend Base;
    using Base::q_; using Base::r_; using Base::rs_; using Base::nslots_; using Base::used_;
    using Base::rm_; using Base::fpm_; using Base::dispill_; using Base::hf_;
    using Base::disp_at; using Base::disp_place;
    std::vector<uint64_t, common::Allocator<uint64_t>> ocraw_, remraw_, cntraw_;
    // unpacked remainder stream (REM_PACKED = false, rs_ <= 32 only): one
    // u32 per slot for the direct 8-lane compare; packed wins memory
    std::vector<uint32_t, common::Allocator<uint32_t>> remu32_;
    // displacement bytes (DECODE_BITS = 8): d < 255 inline, 255 = spilled
    std::vector<uint8_t, common::Allocator<uint8_t>> dispraw_;

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
    // payload write; liveness is the bitmap bit (bitmap mode) or the nonzero counter (sentinel mode)
    void place(uint64_t i, uint64_t f, uint64_t c) {
        set_rem(i, f);
        cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), c);
        if(OCC_BITMAP) cqf::detail::bv_set(ocraw_.data(), i);
    }

    // ---- storage hooks (cht_core) ----
    void init_storage_() {
        if(!REM_PACKED && rs_ > 32) // u32 remainder cells
            throw std::invalid_argument(std::string("[cht] REM_PACKED=false requires stored remainder width <= 32; got ") + std::to_string(rs_));
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
        if(DECODE_BITS) dispraw_.assign(nslots_, 0);
        else decltype(dispraw_)().swap(dispraw_);
    }
    size_t storage_bytes_() const {
        return (OCC_BITMAP ? ocraw_.size(): 0) * sizeof(uint64_t)
             + (REM_PACKED ? remraw_.size() * sizeof(uint64_t): remu32_.size() * sizeof(uint32_t))
             + cntraw_.size() * sizeof(uint64_t)
             + (DECODE_BITS ? dispraw_.size(): 0);
    }
    INLINE uint64_t load_(uint64_t i) const {return i;}
    INLINE bool h_occ_(uint64_t i, uint64_t) const {return occupied(i);}
    INLINE uint64_t h_rem_(uint64_t i, uint64_t) const {return rem(i);}
    INLINE uint64_t h_val_(uint64_t i, uint64_t) const {return cnt(i);}
    INLINE uint64_t h_disp8_(uint64_t i, uint64_t) const {return dispraw_[i];}
    void place_(uint64_t i, uint64_t f, uint64_t v, uint64_t d8) {
        place(i, f, v);
        if(DECODE_BITS) dispraw_[i] = uint8_t(d8);
    }
    void set_val_(uint64_t i, uint64_t v) {cqf::detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), v);}
    void set_disp8_(uint64_t i, uint64_t d8) {dispraw_[i] = uint8_t(d8);}
    INLINE void prefetch_(uint64_t hh) const {
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
    // the occupancy bitmap total (shared popcount machinery) equals size()
    bool extra_check_() const {
        return !OCC_BITMAP || cqf::detail::popcount_words(ocraw_.data(), ocraw_.size()) == used_;
    }

public:
    using Base::SAT; using Base::home_exact; using Base::exact;
    using Base::home; using Base::fp; using Base::count; using Base::add_slot; using Base::load_factor;
    using Base::add_hashed; using Base::add; using Base::hash;
    explicit cht_t(uint32_t q, RevHash &&hf=RevHash()): Base(std::move(hf)) {this->init_core(q);}
    // exact slot count (any n for cht_div_sizing; a power of two otherwise)
    explicit cht_t(cht_slots s, RevHash &&hf=RevHash()): Base(std::move(hf)) {this->init_core_n(s.n);}
    cht_t() = delete;

    // False-positive bound, clustering-aware (see TRUNCATED-MODE FPR,
    // HONESTLY above): an absent key collides iff its rs_-wide remainder
    // appears anywhere in its probe span of expected length
    // L(a) = (1 + 1/(1-a)^2)/2 at load a. Exact mode reports 0. NOTE this is
    // NOT the same basis as cqfbase_t::fpr_bound (idealized per-(home,fp)).
    double fpr_bound() const {
        if (exact) return 0.;
        const double a = load_factor();
        const double L = (1. + 1. / (1. - a) / (1. - a)) / 2.;
        return 1. - std::exp(-L / std::ldexp(1., int(rs_)));
    }

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

    // ---- SIMD span scan (struct-of-arrays only) ----
    // first empty slot at or after i (end of the probe span), via the
    // occupancy bitmap 4 words at a time
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
    // remainders in the span (home_exact: a match must also sit at its own
    // probe distance), and only when the span reaches the table end without
    // an empty slot continue the probe cyclically. Identical results to count().
    uint64_t count_simd_hash(Key h) const {
        const uint64_t f = fp(h);
        const uint64_t h0 = home(h);
        const uint64_t e = probe_end(h0);
        // home_exact: a remainder match must also sit at its own probe distance
        // from h0; otherwise keep scanning the span past it
        for(uint64_t lo = h0;;) {
            const uint64_t s = scan_rem_simd(lo, e, f);
            if(s == ~uint64_t(0)) break;
            if(!home_exact || disp_at(s) == s - h0) return cnt(s);
            lo = s + 1;
        }
        if(e < nslots_) return 0;
        for(uint64_t i = 0;; i = this->next_slot(i)) { // cyclically continued probe
            if(!occupied(i)) return 0;
            if(rem(i) == f && (!home_exact || disp_at(i) == nslots_ - h0 + i)) return cnt(i);
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

    // Sorted bulk build: input sorted by hash ascending. Places each run of
    // equal (home, fingerprint) keys once with its multiplicity -- identical
    // placement to add() on the same stream; falls back to per-key add()
    // when a run would wrap past the first home.
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
                const uint64_t slot = i >= nslots_ ? i - nslots_: i;   // i < first_home + nslots <= 2 * nslots
                place(slot, f, c);
                disp_place(slot, i - h);
                ++used_; ++i;
                j = k;
            }
            next_empty = i;
            first = gend;
        }
    }
};

// ---- cht_map_t: single-stream storage, exact key -> value map ------------
// One packed cell per slot, [ remainder (r) | value (VAL_BITS) | displacement
// (8) ], value 0 = empty (no occupancy bitmap). A probe step is one unaligned
// 16-byte load; a lookup usually touches one cache line (cht_t touches four
// arrays on a hit). Exact mode only (full remainder, displacements kept);
// 64-bit keys; values in [1, 2^vbits - 1]. VAL_BITS = 0 selects the value
// width at construction (e.g. the minimum for a dense id range). Adds to the
// core: serialization (write/read, the width travels in the header) and
// cell_bits().
template<class Key, class RevHash, size_t VAL_BITS, class Sizing=cht_pow2_sizing>
class cht_map_t: public detail::cht_core<cht_map_t<Key, RevHash, VAL_BITS, Sizing>, Key, RevHash, VAL_BITS, 0, 8, Sizing,
                                         std::conditional_t<std::is_same<Key, unsigned __int128>::value, unsigned __int128, uint64_t>> {
    static_assert(std::is_same<Key, uint64_t>::value || std::is_same<Key, unsigned __int128>::value,
                  "cht_map_t: Key must be uint64_t or unsigned __int128");
    // 128-bit keys: remainders can exceed 64 bits and cells reach ~182 bits;
    // fields are then read/written as bit fields at slot i * cell_bits (the
    // load handle is the slot index). 64-bit keys keep the one-load cell paths.
    static constexpr bool k128 = std::is_same<Key, unsigned __int128>::value;
    using RemT = std::conditional_t<k128, unsigned __int128, uint64_t>;
    static_assert(VAL_BITS <= 48, "cht_map_t: VAL_BITS in [1, 48], or 0 = runtime width (constructor argument)");
    using Base = detail::cht_core<cht_map_t, Key, RevHash, VAL_BITS, 0, 8, Sizing, RemT>;
    using handle_t = std::conditional_t<k128, uint64_t, unsigned __int128>;
    friend Base;
    using u128 = unsigned __int128;
    using Base::r_; using Base::nslots_; using Base::used_; using Base::dispill_;
    using Base::vb_; using Base::sat_;
    uint32_t w_ = 0;    // cell width r_ + vb_ + 8
    u128 cm_ = 0;       // low-w_ mask
    // load width: cells of <= 57 bits fit one unaligned 8-byte load (bit
    // offset <= 7); a 64-bit cell is exactly one aligned word; wider cells
    // take a 16-byte load (up to 121 bits)
    enum CellMode: uint8_t {NARROW64, WORD64, WIDE128};
    CellMode mode_ = WIDE128;
    std::vector<uint8_t, common::Allocator<uint8_t>> bits_;

    INLINE u128 cell(uint64_t i) const {
        if(mode_ == WORD64) {
            uint64_t x;
            std::memcpy(&x, bits_.data() + (i << 3), sizeof(x));
            return x;
        }
        const uint64_t off = i * w_;
        if(mode_ == NARROW64) {
            uint64_t x;
            std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
            return (x >> (off & 7)) & uint64_t(cm_);
        }
        u128 x;
        std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
        return (x >> (off & 7)) & cm_;
    }
    INLINE void set_cell(uint64_t i, u128 v) {
        if(mode_ == WORD64) {
            const uint64_t x = uint64_t(v);
            std::memcpy(bits_.data() + (i << 3), &x, sizeof(x));
            return;
        }
        const uint64_t off = i * w_;
        const unsigned s = unsigned(off & 7);
        if(mode_ == NARROW64) {
            uint64_t x;
            std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
            const uint64_t m = uint64_t(cm_);
            x = (x & ~(m << s)) | ((uint64_t(v) & m) << s);
            std::memcpy(bits_.data() + (off >> 3), &x, sizeof(x));
            return;
        }
        u128 x;
        std::memcpy(&x, bits_.data() + (off >> 3), sizeof(x));
        x = (x & ~(cm_ << s)) | ((v & cm_) << s);
        std::memcpy(bits_.data() + (off >> 3), &x, sizeof(x));
    }
    INLINE void set_field(uint64_t i, unsigned shift, uint64_t width_mask, uint64_t v) {
        const u128 c = cell(i);
        set_cell(i, (c & ~(u128(width_mask) << shift)) | (u128(v & width_mask) << shift));
    }
    // width in [1, 64] bits at an arbitrary bit position (one 16-byte load)
    INLINE uint64_t get_bits(uint64_t pos, unsigned width) const {
        u128 x;
        std::memcpy(&x, bits_.data() + (pos >> 3), sizeof(x));
        const uint64_t v = uint64_t(x >> (pos & 7));
        return width >= 64 ? v: (v & ((uint64_t(1) << width) - 1));
    }
    INLINE void set_bits(uint64_t pos, unsigned width, uint64_t v) {
        const unsigned s = unsigned(pos & 7);
        const u128 m = (width >= 64 ? u128(~uint64_t(0)): u128((uint64_t(1) << width) - 1)) << s;
        u128 x;
        std::memcpy(&x, bits_.data() + (pos >> 3), sizeof(x));
        x = (x & ~m) | ((u128(v) << s) & m);
        std::memcpy(bits_.data() + (pos >> 3), &x, sizeof(x));
    }
    INLINE RemT get_rem128(uint64_t base) const {
        if(r_ <= 64) return RemT(get_bits(base, r_));
        return (RemT(get_bits(base + 64, r_ - 64)) << 64) | RemT(get_bits(base, 64));
    }
    INLINE void set_rem128(uint64_t base, RemT f) {
        if(r_ <= 64) {set_bits(base, r_, uint64_t(f)); return;}
        set_bits(base, 64, uint64_t(f));
        set_bits(base + 64, r_ - 64, uint64_t(f >> 64));
    }

    // ---- storage hooks (cht_core) ----
    void init_storage_() {
        w_ = r_ + vb_ + 8;
        if(!k128 && w_ > 121) // one 16-byte load at a byte offset covers 121 bits
            throw std::invalid_argument("[cht_map] cell width (key_bits - q + value bits + 8) exceeds 121 bits");
        cm_ = (u128(1) << w_) - 1;
        mode_ = w_ <= 57 ? NARROW64: w_ == 64 ? WORD64: WIDE128;
        bits_.assign(ceil_div(nslots_ * w_, uint64_t(8)) + 16, 0);
    }
    size_t storage_bytes_() const {return bits_.size();}
    INLINE handle_t load_(uint64_t i) const {
        if constexpr(k128) return i;
        else return cell(i);
    }
    INLINE uint64_t h_val_(uint64_t i, const handle_t &c) const {
        if constexpr(k128) return get_bits(i * w_ + r_, vb_);
        else {(void)i; return uint64_t(c >> r_) & sat_;}
    }
    INLINE bool h_occ_(uint64_t i, const handle_t &c) const {return h_val_(i, c) != 0;}
    INLINE RemT h_rem_(uint64_t i, const handle_t &c) const {
        if constexpr(k128) return get_rem128(i * w_);
        else {(void)i; return uint64_t(c) & uint64_t(this->rm_);}
    }
    INLINE uint64_t h_disp8_(uint64_t i, const handle_t &c) const {
        if constexpr(k128) return get_bits(i * w_ + r_ + vb_, 8);
        else {(void)i; return uint64_t(c >> (r_ + vb_)) & 0xFFu;}
    }
    void place_(uint64_t i, RemT f, uint64_t v, uint64_t d8) {
        if constexpr(k128) {
            const uint64_t base = i * w_;
            set_rem128(base, f);
            set_bits(base + r_, vb_, v);
            set_bits(base + r_ + vb_, 8, d8);
        } else {
            set_cell(i, u128(f) | (u128(v) << r_) | (u128(d8) << (r_ + vb_)));
        }
    }
    void set_val_(uint64_t i, uint64_t v) {
        if constexpr(k128) set_bits(i * w_ + r_, vb_, v);
        else set_field(i, r_, sat_, v);
    }
    void set_disp8_(uint64_t i, uint64_t d8) {
        if constexpr(k128) set_bits(i * w_ + r_ + vb_, 8, d8);
        else set_field(i, r_ + vb_, 0xFFu, d8);
    }
    INLINE void prefetch_(uint64_t hh) const {
        simde_mm_prefetch(reinterpret_cast<const char*>(bits_.data() + ((hh * w_) >> 3)), SIMDE_MM_HINT_T0);
    }
    bool extra_check_() const {return true;}

public:
    // val_bits: required for VAL_BITS == 0 (runtime width in [1, 48]); must
    // equal VAL_BITS otherwise
    explicit cht_map_t(uint32_t q, uint32_t val_bits = uint32_t(VAL_BITS), RevHash &&hf=RevHash()): Base(std::move(hf)) {
        this->set_val_bits_(val_bits, 48);
        this->init_core(q);
    }
    // exact slot count (any n in [4, 2^40] for cht_div_sizing; a power of two otherwise)
    explicit cht_map_t(cht_slots s, uint32_t val_bits = uint32_t(VAL_BITS), RevHash &&hf=RevHash()): Base(std::move(hf)) {
        this->set_val_bits_(val_bits, 48);
        this->init_core_n(s.n);
    }
    cht_map_t() = delete;
    uint32_t cell_bits() const {return w_;}
    // 0: one unaligned 8-byte load (cells <= 57 bits), 1: one aligned 64-bit
    // word (cells of exactly 64 bits), 2: a 16-byte load (58-63 or > 64 bits)
    unsigned cell_mode() const {return unsigned(mode_);}
    // full check by default (every live slot re-probes to itself)
    bool structure_check(size_t sample = 0) const {return Base::structure_check(sample);}

    // Binary serialization: magic "CHTMAP02", key bits, value bits, sizing
    // (1 = power of two, 0 = div), nslots, size,
    // the cell stream and the displacement spill. RevHash must be stateless
    // (or default-constructed identically) on both sides; read() checks the
    // key/value widths against this instantiation and throws on mismatch.
    void write(std::FILE *fp) const {
        static const char magic[8] = {'C','H','T','M','A','P','0','2'};
        const uint32_t hdr[3] = {this->kh_, vb_, uint32_t(Sizing::pow2)};
        const uint64_t nb = bits_.size(), ns = dispill_.size();
        if(std::fwrite(magic, 1, 8, fp) != 8 || std::fwrite(hdr, sizeof(hdr), 1, fp) != 1
           || std::fwrite(&nslots_, sizeof(nslots_), 1, fp) != 1
           || std::fwrite(&used_, sizeof(used_), 1, fp) != 1 || std::fwrite(&nb, sizeof(nb), 1, fp) != 1
           || std::fwrite(bits_.data(), 1, nb, fp) != nb || std::fwrite(&ns, sizeof(ns), 1, fp) != 1
           || (ns && std::fwrite(dispill_.data(), sizeof(dispill_[0]), ns, fp) != ns))
            throw std::runtime_error("[cht_map] write failed");
    }
    static cht_map_t make_for_read(uint64_t n, uint32_t vb, uint32_t kb) {
        if constexpr(Base::runtime_key_bits) return cht_map_t(cht_slots{n}, vb, RevHash(kb));
        else {(void)kb; return cht_map_t(cht_slots{n}, vb);}
    }
    static cht_map_t read(std::FILE *fp) {
        char magic[8];
        uint32_t hdr[3];
        uint64_t n, used, nb, ns;
        if(std::fread(magic, 1, 8, fp) != 8 || std::memcmp(magic, "CHTMAP02", 8))
            throw std::runtime_error("[cht_map] read: bad magic");
        if(std::fread(hdr, sizeof(hdr), 1, fp) != 1 || (!Base::runtime_key_bits && hdr[0] != Base::key_bits) || (VAL_BITS != 0 && hdr[1] != VAL_BITS))
            throw std::runtime_error("[cht_map] read: key/value widths do not match this instantiation");
        if(hdr[2] != uint32_t(Sizing::pow2))
            throw std::runtime_error("[cht_map] read: table sizing policy does not match this instantiation");
        if(std::fread(&n, sizeof(n), 1, fp) != 1)
            throw std::runtime_error("[cht_map] read: truncated header");
        cht_map_t ret = make_for_read(n, hdr[1], hdr[0]);
        if(std::fread(&used, sizeof(used), 1, fp) != 1 || std::fread(&nb, sizeof(nb), 1, fp) != 1 || nb != ret.bits_.size())
            throw std::runtime_error("[cht_map] read: truncated or inconsistent header");
        if(std::fread(ret.bits_.data(), 1, nb, fp) != nb || std::fread(&ns, sizeof(ns), 1, fp) != 1)
            throw std::runtime_error("[cht_map] read: truncated cell stream");
        ret.dispill_.resize(ns);
        if(ns && std::fread(ret.dispill_.data(), sizeof(ret.dispill_[0]), ns, fp) != ns)
            throw std::runtime_error("[cht_map] read: truncated spill");
        ret.used_ = used;
        return ret;
    }
};

} // namespace cht
} // namespace sketch

#endif // SKETCH_CHT_H__
