#ifndef SKETCH_CQF_H__
#define SKETCH_CQF_H__
// Struct-of-arrays counting quotient filter.
// Study reference for semantics: Pandey et al., SIGMOD 2017 (quotient filter).
//
// LAYOUT (hybrid: directories retained, single-register payload,
// co-located metadata):
//   nslots = 2^q homes == physical slots. No overflow region, no wraparound;
//   the table is linear [0, nslots). occ (home h nonempty; nslots bits) and
//   runend (slot ends a run; xnslots bits) live in ONE flat word allocation,
//   occ words first, each region backed by a Fenwick rank/select directory
//   (1024-bit blocks; ~16 KB total at q=20). Measured: the directories are
//   the difference between ~0.6 us and ~22 us per query at q=20.
//   Remainder (r = KEY_BITS-q bits) and counter (VAL_BITS) share ONE
//   register per slot whenever they fit (exact mode, r + VAL_BITS <= 64:
//   cqf64_16 at q=16 is exactly 48+16=64, at q=20 44+16=60); otherwise they
//   stay in separate packed arrays. cqf64_64's counter array is already
//   word-per-slot through the packed helpers' word-width path.
//
// INVARIANTS:
//   I1 inserts place home h's run at slots >= h (home-anchored); deletions
//      shift runs left, so a run may drift below its home -- navigation
//      never relies on the run being at/after its home
//   I2 a cluster is a maximal span of nonempty slots
//   I3 the k-th run of the table (slot order) pairs with the k-th occupied
//      home (index order); runs of a cluster are packed back-to-back in
//      increasing home order
//   I4 within a run, remainders are sorted ascending
//   I5 occ/runend/cnt semantics as standard QF: runend marks the last slot
//      of each run; cnt == 0 <=> slot empty; used_ == live slots
//
// Navigation is directory-assisted rank/select over the two metadata
// regions: rank1 in O(1 + intra-block scan), select1 in
// O(log nblocks + intra-block scan), with O(range/gran) local directory
// refreshes after every slot-range mutation. The flat bv_rank1/bv_select1
// scans (vectorized prefix popcount; per-lane movemask + PDEP/TZCNT) remain
// as the structure_check reference, probed against the directories at
// sampled ranks/selects. Locality comes from the 16j invariants:
// home-anchored placement bounds shifts at the first empty slot, and
// deletes keep clusters contiguous, so shifts stay inside the touched
// cluster. Shifts are whole-register memmove in the fused regime, byte-
// memmove or word-funnel otherwise. Load cap 0.9; grow() reinserts at q+1.
//
// MODES: exact when KEY_BITS spans the hash width (64) -- an STL-like
// integer->integer map; inexact otherwise, with fpr_bound() reporting the
// false-positive probability. Same API family in both modes.
//
// SIMD: SKETCH_CQF_NO_SIMD selects the scalar loops everywhere; correctness
// is identical either way. On-disk format is unchanged (CQF4: packed rem
// stream, packed cnt stream, occ words, runend words); the fused regime
// packs/unpacks at the I/O boundary.
#include "common.h"
#include "hash.h"
#if defined(__aarch64__) && !defined(SKETCH_CQF_NO_SIMD)
#include <arm_neon.h>
#endif
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <shared_mutex>
#include <thread>

namespace sketch {
inline namespace cqf {

// word granularity bound once from sizeof (shared with the packed helpers
// in detail below and the filter class above)
static constexpr unsigned word_bits = bits_per_word<uint64_t>;

namespace detail {

// ---- popcount kernels: single CQF call site, routed through the shared
// libpopcnt machinery (AVX2/AVX512/neon dispatch lives there); explicit
// scalar loop only for the SKETCH_CQF_NO_SIMD build.
static INLINE uint64_t popcount_words(const uint64_t *p, size_t n) {
#if defined(SKETCH_CQF_NO_SIMD)
    uint64_t s = 0;
    for(size_t i = 0; i < n; ++i) s += popcount(p[i]);
    return s;
#else
    return popcnt(p, n * sizeof(uint64_t));
#endif
}

// ---- flat bitvector primitives over a raw word range (no object, no
// per-block metadata; the occ and runend vectors are regions of one array)
static INLINE bool bv_get(const uint64_t *w, uint64_t i) {
    return (w[i / word_bits] >> (i % word_bits)) & 1;
}
static INLINE void bv_set(uint64_t *w, uint64_t i) {
    w[i / word_bits] |= uint64_t(1) << (i % word_bits);
}
static INLINE void bv_clear(uint64_t *w, uint64_t i) {
    w[i / word_bits] &= ~(uint64_t(1) << (i % word_bits));
}
static INLINE uint64_t bv_rank1(const uint64_t *w, uint64_t i) { // # set in [0, i)
    const uint64_t wi = i / word_bits, off = i % word_bits;
    uint64_t s = popcount_words(w, wi);
    if(off) s += popcount(w[wi] & ((uint64_t(1) << off) - 1));
    return s;
}
// position of the n-th (0-based) set bit within word
static INLINE unsigned ctz_nth(uint64_t word, uint64_t n) {
#if defined(__BMI2__)
    return unsigned(ctz(_pdep_u64(uint64_t(1) << n, word)));
#else
    // binary descent by popcount halves: O(6) dependent steps, branch-
    // predictable (each test is a uint32 compare); beats the bit-clearing
    // loop whenever the word is dense (runend/occ words at load >= 0.5)
    unsigned pos = 0;
    const uint32_t lo32 = uint32_t(word);
    const unsigned p32 = popcount(lo32);
    uint64_t v;
    if(p32 <= n) { n -= p32; pos = 32; v = word >> 32; } else v = lo32;
    const uint16_t lo16 = uint16_t(v);
    const unsigned p16 = popcount(lo16);
    if(p16 <= n) { n -= p16; pos += 16; v >>= 16; } else v = lo16;
    const uint8_t lo8 = uint8_t(v);
    const unsigned p8 = popcount(lo8);
    if(p8 <= n) { n -= p8; pos += 8; v >>= 8; }
    uint32_t b = uint32_t(v) & 0xff;
    const unsigned p4 = popcount(b & 0xf);
    if(p4 <= n) { n -= p4; pos += 4; b >>= 4; }
    const unsigned p2 = popcount(b & 0x3);
    if(p2 <= n) { n -= p2; pos += 2; b >>= 2; }
    const unsigned p1 = b & 1;
    if(p1 <= n) { n -= p1; pos += 1; b >>= 1; }
    return pos + unsigned(ctz(uint8_t(b)));
#endif
}
static INLINE uint64_t bv_select1(const uint64_t *w, uint64_t nw, uint64_t k) {
    // 0-indexed; requires k < total set bits. Coarse pass walks per-lane
    // popcount chunks (AVX2: inclusive-prefix movemask picks the lane);
    // refine pass is word-at-a-time; in-word bit via ctz_nth (PDEP/TZCNT).
    uint64_t wi = 0, c = 0;
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
    for(; wi + 4 <= nw; wi += 4) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(w + wi));
        const __m256i pc = popcnt_fn(v); // per-lane popcounts
        const __m128i lo = _mm256_castsi256_si128(pc);
        const __m128i hi = _mm256_extracti128_si256(pc, 1);
        const __m128i pf_lo = _mm_add_epi64(lo, _mm_slli_si128(lo, 8)); // [p0, p0+p1]
        const __m128i pf_hi = _mm_add_epi64(_mm_add_epi64(hi, _mm_slli_si128(hi, 8)),
                                            _mm_set1_epi64x(_mm_extract_epi64(pf_lo, 1)));
        const __m256i pfx = _mm256_inserti128_si256(_mm256_castsi128_si256(pf_lo), pf_hi, 1);
        const uint64_t tot = _mm_extract_epi64(pf_hi, 1);
        if(c + tot > k) { // k lands in this chunk: first lane whose inclusive prefix exceeds k - c
            const __m256i tgt = _mm256_set1_epi64x(k - c);
            const unsigned m = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpgt_epi64(pfx, tgt)));
            const unsigned lane = unsigned(ctz(m));
            uint64_t excl = 0;
            if(lane == 1) excl = _mm_extract_epi64(pf_lo, 0);
            else if(lane == 2) excl = _mm_extract_epi64(pf_lo, 1);
            else if(lane == 3) excl = _mm_extract_epi64(pf_hi, 0);
            return (wi + lane) * word_bits + ctz_nth(w[wi + lane], k - c - excl);
        }
        c += tot;
    }
#endif
    for(; wi < nw; ++wi) {
        const uint64_t pc = popcount(w[wi]);
        if(c + pc > k) return wi * word_bits + ctz_nth(w[wi], k - c);
        c += pc;
    }
    return nw * word_bits; // unreachable when k < total
}
// first set bit position >= i, or nw * word_bits if none
static INLINE uint64_t bv_next_set(const uint64_t *w, uint64_t nw, uint64_t i) {
    uint64_t wi = i / word_bits;
    if(wi >= nw) return nw * word_bits;
    uint64_t word = w[wi] & (~uint64_t(0) << (i % word_bits));
    while(true) {
        if(word) return wi * word_bits + ctz(word);
        if(++wi >= nw) return nw * word_bits;
        word = w[wi];
    }
}
static INLINE uint64_t bv_total(const uint64_t *w, uint64_t nw) {
    return popcount_words(w, nw);
}

// ---- packed fixed-width array helpers (bits in [1, word_bits]) ----
static INLINE uint64_t packed_get(const uint64_t *w, uint64_t i, unsigned bits) {
    const uint64_t b = i * bits;
    const unsigned sh = unsigned(b & (word_bits - 1));
    const uint64_t *p = w + (b / word_bits);
    uint64_t v;
    if(bits == word_bits) return p[0];
    if(sh + bits <= word_bits) v = p[0] >> sh;
    else v = (p[0] >> sh) | (p[1] << (word_bits - sh));
    return v & ((uint64_t(1) << bits) - 1);
}
static INLINE void packed_set(uint64_t *w, uint64_t i, unsigned bits, uint64_t v) {
    const uint64_t b = i * bits;
    const unsigned sh = unsigned(b & (word_bits - 1));
    uint64_t *p = w + (b / word_bits);
    if(bits == word_bits) {p[0] = v; return;}
    const uint64_t mask = ((uint64_t(1) << bits) - 1) << sh;
    p[0] &= ~mask;
    p[0] |= (v << sh) & mask;
    if(sh + bits > word_bits) {
        const uint64_t mask2 = ((uint64_t(1) << bits) - 1) >> (word_bits - sh);
        p[1] &= ~mask2;
        p[1] |= v >> (word_bits - sh);
    }
}
// move nbits starting at absolute bit B0 to B0+delta (delta > 0 moves toward
// higher addresses). Bits outside the moved window but sharing boundary words
// are preserved; the one-slot gap at the trailing end is the caller's to
// rewrite. Word-granular: byte-aligned fast path, funnel otherwise.
static INLINE void shift_bit_window(uint64_t *w, uint64_t B0, uint64_t nbits, int64_t delta) {
    const uint64_t D0 = uint64_t(int64_t(B0) + delta);
    const uint64_t dW = D0 / word_bits, dO = D0 % word_bits;
    const uint64_t nwords = ceil_div(dO + nbits, uint64_t(word_bits));
    for(uint64_t kk = 0; kk < nwords; ++kk) {
        const uint64_t k = delta > 0 ? nwords - 1 - kk: kk;
        const uint64_t absword = dW + k;
        const uint64_t wstart = absword * word_bits;
        const uint64_t lo = wstart > D0 ? wstart: D0;
        const uint64_t hi = wstart + word_bits < D0 + nbits ? wstart + word_bits: D0 + nbits;
        const uint64_t slo = uint64_t(int64_t(lo) - delta);
        const uint64_t *sp = w + slo / word_bits;
        const unsigned ssh = unsigned(slo % word_bits);
        const unsigned len = unsigned(hi - lo);
        uint64_t srcv = ssh + len <= word_bits ? sp[0] >> ssh: (sp[0] >> ssh) | (sp[1] << (word_bits - ssh));
        srcv &= len == word_bits ? ~uint64_t(0): (uint64_t(1) << len) - 1;
        const uint64_t dmask = (len == word_bits ? ~uint64_t(0): ((uint64_t(1) << len) - 1)) << (lo % word_bits);
        w[absword] = (w[absword] & ~dmask) | (srcv << (lo % word_bits));
    }
}
// move slots [s, e-1] into [s+1, e] (caller writes slot s). Linear s <= e.
static INLINE void packed_shift_right(uint64_t *w, uint64_t s, unsigned bits, uint64_t e) {
    if(s >= e) return;
    if(bits == word_bits) {
        std::memmove(w + s + 1, w + s, (e - s) * sizeof(uint64_t));
        return;
    }
    if(bits % 8 == 0) { // byte-aligned: memcpy-class bulk shift
        uint8_t *b = reinterpret_cast<uint8_t *>(w);
        const uint64_t B0 = s * bits / 8, nb = (e - s) * bits / 8, db = bits / 8;
        std::memmove(b + B0 + db, b + B0, nb);
        return;
    }
    shift_bit_window(w, s * bits, (e - s) * bits, int64_t(bits));
}
// move slots [s+1, e] into [s, e-1]; slot e is zeroed. Linear s <= e.
static INLINE void packed_shift_left(uint64_t *w, uint64_t s, unsigned bits, uint64_t e) {
    if(bits == word_bits) {
        std::memmove(w + s, w + s + 1, (e - s) * sizeof(uint64_t));
        packed_set(w, e, bits, 0);
        return;
    }
    if(bits % 8 == 0) {
        uint8_t *b = reinterpret_cast<uint8_t *>(w);
        const uint64_t B0 = (s + 1) * bits / 8, nb = (e - s) * bits / 8, db = bits / 8;
        std::memmove(b + B0 - db, b + B0, nb);
        packed_set(w, e, bits, 0);
        return;
    }
    shift_bit_window(w, (s + 1) * bits, (e - s) * bits, -int64_t(bits));
    packed_set(w, e, bits, 0);
}

// Rank/select directory over a raw bitvector word range: Fenwick tree of
// per-block popcounts (one block = dir_gran bits) supporting rank1 in
// O(1 + intra-block scan) and select1 in O(log nblocks + intra-block scan),
// with O(range/gran) local updates after a slot-range mutation. Blocks are
// intentionally large (16 words): intra-block scans stay SIMD-popcount fast
// and the directory costs 2 * 32 / 1024 = 0.0625 bit per slot per vector
// (~16 KB total at q=20). Measured carry: this structure is the difference
// between ~0.6 us and ~22 us per query at q=20 (three-way A/B).
class dir {
    static constexpr unsigned lg = 1024 / word_bits; // words per block
    std::vector<uint32_t, common::Allocator<uint32_t>> bsum_, fw_;
    uint64_t nblocks_ = 0;
    static void fw_add(std::vector<uint32_t, common::Allocator<uint32_t>> &fw, uint32_t i, int32_t d) {
        for(++i; i < fw.size(); i += i & -i) fw[i] = uint32_t(int64_t(fw[i]) + d);
    }
    uint32_t fw_sum(uint32_t i) const { // sum of bsum_[0, i)
        uint32_t s = 0;
        for(; i; i -= i & -i) s += fw_[i];
        return s;
    }
public:
    dir() = default;
    size_t bytes() const {return (bsum_.size() + fw_.size()) * sizeof(uint32_t);}
    void reset(uint64_t nbits) {
        nblocks_ = ceil_div(nbits, uint64_t(lg * word_bits));
        bsum_.assign(nblocks_, 0);
        fw_.assign(nblocks_ + 1, 0);
    }
    void rebuild(const uint64_t *w, uint64_t nw) {
        std::fill(bsum_.begin(), bsum_.end(), 0);
        std::fill(fw_.begin(), fw_.end(), 0);
        for(uint64_t b = 0; b < nblocks_; ++b) {
            const uint64_t lo = b * lg, hi = std::min<uint64_t>(lo + lg, nw);
            uint32_t s = 0;
            for(uint64_t widx = lo; widx < hi; ++widx) s += popcount(w[widx]);
            bsum_[b] = s;
            for(uint32_t i = b + 1; i < fw_.size(); i += i & -i) fw_[i] += s;
        }
    }
    // recompute the blocks intersecting [lo, hi] after a mutation there
    void refresh(const uint64_t *w, uint64_t nw, uint64_t lo, uint64_t hi) {
        if(nblocks_ == 0) return;
        const uint64_t b0 = std::min<uint64_t>(lo / (lg * word_bits), nblocks_ - 1);
        const uint64_t b1 = std::min<uint64_t>(hi / (lg * word_bits), nblocks_ - 1);
        for(uint64_t b = b0; b <= b1; ++b) {
            const uint64_t blo = b * lg, bhi = std::min<uint64_t>(blo + lg, nw);
            uint32_t ns = 0;
            for(uint64_t widx = blo; widx < bhi; ++widx) ns += popcount(w[widx]);
            if(ns != bsum_[b]) {
                fw_add(fw_, uint32_t(b), int32_t(ns) - int32_t(bsum_[b]));
                bsum_[b] = ns;
            }
        }
    }
    uint64_t rank1(const uint64_t *w, uint64_t i) const { // # set in [0, i)
        const uint64_t blk = i / (lg * word_bits), off = i % (lg * word_bits);
        uint64_t s = fw_sum(uint32_t(blk));
        const uint64_t wi = blk * lg + off / word_bits, wo = off % word_bits;
        s += popcount_words(w + blk * lg, wi - blk * lg);
        if(wo) s += popcount(w[wi] & ((uint64_t(1) << wo) - 1));
        return s;
    }
    uint64_t select1(const uint64_t *w, uint64_t k) const { // 0-indexed
        // Fenwick binary lifting: smallest block whose inclusive prefix > k
        uint32_t b = 0, acc = 0;
        for(uint32_t st = uint32_t(1) << (31 - clz(uint32_t(fw_.size()))); st; st >>= 1) {
            const uint32_t nb = b + st;
            if(nb < fw_.size() && acc + fw_[nb] <= k) {b = nb; acc += fw_[nb];}
        }
        uint64_t rem = k - acc; // target within block b
        const uint64_t *ww = w + b * lg;
        for(uint64_t j = 0;; ++j) {
            const uint64_t pc = popcount(ww[j]);
            if(pc > rem) return (b * lg + j) * word_bits + ctz_nth(ww[j], rem);
            rem -= pc;
        }
    }
};

} // namespace detail

template<size_t KEY_BITS, size_t VAL_BITS, class Storage=uint64_t, class Hash=WangHash>
class cqfbase_t {
    static_assert(KEY_BITS >= 4 && KEY_BITS <= 64, "KEY_BITS in [4, 64]");
    static_assert(VAL_BITS >= 1 && VAL_BITS <= 64, "VAL_BITS in [1, 64]");
public:
    static constexpr size_t key_bits = KEY_BITS;
    static constexpr size_t val_bits = VAL_BITS;
    static constexpr bool exact = (KEY_BITS >= 64);
    static constexpr uint64_t SAT = VAL_BITS >= 64 ? ~uint64_t(0): ((uint64_t(1) << VAL_BITS) - 1);
private:
    uint32_t q_ = 0, r_ = 0;
    uint64_t nslots_ = 0;     // homes (2^q)
    uint64_t xnslots_ = 0;    // physical slots: nslots + ~2*sqrt(nslots)
                              // overflow tail (linear, no wrap); absorbs the
                              // max-home run creep that a physical==homes
                              // table cannot (the classic QF capacity cliff)
    uint64_t used_ = 0;       // live slots
    uint64_t ndistinct_ = 0;  // occupied homes
    uint64_t occ_words_ = 0, runend_words_ = 0; // word counts of the two
                              // bitvector regions inside metaraw_
    bool fuse_ = false;       // rem+cnt share one register per slot
    bool fuse_pref_ = true;   // policy: allow fusion when it fits (grow/read
                              // re-apply it); false selects the packed arrays
    uint64_t remmask_ = 0;    // (1 << r_) - 1 (fuse_ regime)
    // payload: fused single-register slots, or packed rem/cnt arrays
    std::vector<uint64_t, common::Allocator<uint64_t>> slot_, remraw_, cntraw_;
    // occ words first, then runend words: one allocation, directory-assisted
    // rank/select over the two regions (flat scans remain only as the
    // structure_check reference)
    std::vector<uint64_t, common::Allocator<uint64_t>> metaraw_;
    detail::dir occ_dir_, runend_dir_;
    Hash hf_;
    static constexpr uint32_t MAGIC = 0x43514634; // "CQF4" (home-anchored layout)
public:
    using final_type = cqfbase_t<KEY_BITS, VAL_BITS, Storage, Hash>;
    using HashType = Hash;
    using key_type = uint64_t;
    using mapped_type = uint64_t;

    explicit cqfbase_t(uint32_t q, Hash &&hf=Hash()): hf_(std::move(hf)) {
        if(q < 2 || KEY_BITS - q < 2 || q > 40)
            throw std::invalid_argument(std::string("[cqf] need 2 <= q <= 40 and 2 <= KEY_BITS - q; got q=") + std::to_string(q));
        init(q);
    }
    // Payload-regime policy: fused single-register slots (default) or the
    // packed rem/cnt arrays. Fusion only engages when rem+cnt fit one
    // register anyway; passing fuse=false forces packing even then -- the
    // packed layout is narrower when rem+cnt use well under 64 bits (e.g.
    // cqf48_16 at q=20: 44 payload bits -> 0.99x the reference footprint vs
    // 1.42x fused), at the cost of funnel-class shifts. The preference
    // survives grow() and selects the unpack target in read().
    explicit cqfbase_t(uint32_t q, bool fuse, Hash &&hf=Hash()): fuse_pref_(fuse), hf_(std::move(hf)) {
        if(q < 2 || KEY_BITS - q < 2 || q > 40)
            throw std::invalid_argument(std::string("[cqf] need 2 <= q <= 40 and 2 <= KEY_BITS - q; got q=") + std::to_string(q));
        init(q);
    }
    cqfbase_t() = delete;
    // File ctors take an explicit tag so a literal 0 can never bind to a
    // pointer overload (the old null-pointer-constant hijack).
    struct from_path_t {explicit from_path_t() = default;};
    struct from_gz_t {explicit from_gz_t() = default;};
    static constexpr from_path_t from_path{};
    static constexpr from_gz_t from_gz{};
    cqfbase_t(from_path_t, const std::string &path, bool fuse=true): cqfbase_t(2, fuse) {read(path);}
    cqfbase_t(from_gz_t, gzFile fp, bool fuse=true): cqfbase_t(2, fuse) {read(fp);}

    void init(uint32_t q) {
        q_ = q; r_ = uint32_t(KEY_BITS) - q;
        nslots_ = uint64_t(1) << q;
        xnslots_ = nslots_ + 2 * (uint64_t(1) << (q / 2)) + 8; // ~2 sqrt(nslots)
        fuse_ = fuse_pref_ && (KEY_BITS - q + VAL_BITS <= word_bits);
        remmask_ = r_ >= word_bits ? ~uint64_t(0): ((uint64_t(1) << r_) - 1);
        if(fuse_) {
            slot_.assign(xnslots_, 0);
            decltype(remraw_)().swap(remraw_);
            decltype(cntraw_)().swap(cntraw_);
        } else {
            remraw_.assign(ceil_div(xnslots_ * r_, uint64_t(word_bits)) + 1, 0);
            cntraw_.assign(ceil_div(xnslots_ * VAL_BITS, uint64_t(word_bits)) + 1, 0);
            decltype(slot_)().swap(slot_);
        }
        occ_words_ = ceil_div(nslots_, uint64_t(word_bits));
        runend_words_ = ceil_div(xnslots_, uint64_t(word_bits));
        metaraw_.assign(occ_words_ + runend_words_, 0);
        occ_dir_.reset(nslots_);
        runend_dir_.reset(xnslots_);
        used_ = ndistinct_ = 0;
    }
    uint32_t q() const {return q_;}
    uint32_t r() const {return r_;}
    uint64_t nslots() const {return nslots_;}
    uint64_t size() const {return used_;}           // distinct keys (live slots)
    bool empty() const {return used_ == 0;}
    double load_factor() const {return double(used_) / double(nslots_);}
    std::pair<size_t, size_t> est_memory_usage() const {
        const size_t payload = fuse_ ? slot_.size(): remraw_.size() + cntraw_.size();
        return std::make_pair(sizeof(*this), (payload + metaraw_.size()) * sizeof(uint64_t)
                              + occ_dir_.bytes() + runend_dir_.bytes());
    }
    bool fused() const {return fuse_;} // rem+cnt in one register per slot
    // false-positive bound for inexact mode
    double fpr_bound() const {
        return exact ? 0.: 1. - std::exp(-double(std::max<uint64_t>(used_, 1)) / std::ldexp(1., int(KEY_BITS)));
    }

    INLINE uint64_t hash(uint64_t v) const {return hf_(v);}
    INLINE void addh(uint64_t e) {add(hf_(e));}
    INLINE std::pair<uint64_t, uint64_t> parts(uint64_t hv) const {
        return {(hv >> r_) & (nslots_ - 1), hv & ((uint64_t(1) << r_) - 1)};
    }
    uint64_t key_of(uint64_t fq, uint64_t fr) const {return (fq << r_) | fr;}

    // ---- metadata regions (one allocation; occ words then runend words) ----
    uint64_t *occ_w() {return metaraw_.data();}
    const uint64_t *occ_w() const {return metaraw_.data();}
    uint64_t *re_w() {return metaraw_.data() + occ_words_;}
    const uint64_t *re_w() const {return metaraw_.data() + occ_words_;}
    bool occ_get(uint64_t h) const {return detail::bv_get(occ_w(), h);}
    void occ_set(uint64_t h) {detail::bv_set(occ_w(), h);}
    void occ_clear(uint64_t h) {detail::bv_clear(occ_w(), h);}
    bool re_get(uint64_t i) const {return detail::bv_get(re_w(), i);}
    void re_set(uint64_t i) {detail::bv_set(re_w(), i);}
    void re_clear(uint64_t i) {detail::bv_clear(re_w(), i);}

    // ---- slot accessors ----
    uint64_t rem(uint64_t i) const {
        return fuse_ ? slot_[i] & remmask_: detail::packed_get(remraw_.data(), i, r_);
    }
    void set_rem(uint64_t i, uint64_t v) {
        if(fuse_) slot_[i] = (slot_[i] & ~remmask_) | (v & remmask_);
        else detail::packed_set(remraw_.data(), i, r_, v);
    }
    uint64_t cnt(uint64_t i) const {
        return fuse_ ? slot_[i] >> r_: detail::packed_get(cntraw_.data(), i, unsigned(VAL_BITS));
    }
    void set_cnt(uint64_t i, uint64_t v) {
        if(fuse_) slot_[i] = (slot_[i] & remmask_) | ((v & SAT) << r_);
        else detail::packed_set(cntraw_.data(), i, unsigned(VAL_BITS), v);
    }
    void set_slot(uint64_t i, uint64_t fr, uint64_t c) { // fused: one store
        if(fuse_) slot_[i] = (fr & remmask_) | ((c & SAT) << r_);
        else {set_rem(i, fr); set_cnt(i, c);}
    }
    bool slot_live(uint64_t i) const {return cnt(i) != 0;}

    // ---- navigation: the k-th occupied home (index order) owns the run
    // ending at the k-th runend (slot order) -- the global run/home
    // correspondence, robust to runs drifting left of their homes after
    // deletions. Hybrid: directory-assisted rank/select (Fenwick over
    // 1024-bit blocks, ~16 KB at q=20) does the navigation; the flat
    // bv_rank1/bv_select1 scans remain as the structure_check reference.
    // Measured at q=20: dir-assisted query ~0.6 us vs ~22 us flat.
    uint64_t next_runend_from(uint64_t p) const {
        return detail::bv_next_set(re_w(), runend_words_, p);
    }
    // run [s, e] (inclusive) of quotient fq; requires occ[fq].
    std::pair<uint64_t, uint64_t> run_of(uint64_t fq) const {
        const uint64_t rnk = occ_dir_.rank1(occ_w(), fq + 1); // 1-indexed home position
        const uint64_t e = runend_dir_.select1(re_w(), rnk - 1);
        uint64_t s = e;
        while(s > 0 && slot_live(s - 1) && !re_get(s - 1)) --s;
        return {s, e};
    }
    uint64_t first_empty(uint64_t s) const {
        // scan RIGHT only (no wrap); the 0.9 load cap guarantees one exists
        if(fuse_) { // cnt == 0 <=> register < 2^r_ (high bits stay zero)
            const uint64_t *sl = slot_.data();
            uint64_t i = s;
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
            const __m256i z = _mm256_setzero_si256();
            const __m128i sh = _mm_cvtsi32_si128(int(r_));
            for(; i + 4 <= xnslots_; i += 4) {
                const __m256i v = _mm256_loadu_si256((const __m256i *)(sl + i));
                const unsigned m = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(_mm256_srl_epi64(v, sh), z)));
                if(m) return i + unsigned(ctz(m));
            }
#endif
            for(; i < xnslots_; ++i) if(sl[i] < remmask_ + 1) return i;
            return ~uint64_t(0); // full (unreachable: ceiling + overflow)
        }
        const uint64_t *cw = cntraw_.data();
        if(VAL_BITS == word_bits) {
            uint64_t i = s;
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
            const __m256i z = _mm256_setzero_si256();
            for(; i + 4 <= xnslots_; i += 4) {
                const __m256i v = _mm256_loadu_si256((const __m256i *)(cw + i));
                const unsigned m = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(v, z)));
                if(m) return i + unsigned(ctz(m));
            }
#endif
            for(; i < xnslots_; ++i) if(cw[i] == 0) return i;
            return ~uint64_t(0); // full (unreachable: ceiling + overflow)
        }
        for(uint64_t j = s; j < xnslots_; ++j)
            if(!slot_live(j)) return j;
        return ~uint64_t(0);
    }
    // single leftward pass from the runend: membership and run-start bound
    // in one loop (the sorted run exits at the first rem < fr scanning
    // right-to-left; the run-start test stops at holes and runends). One
    // payload pass instead of walk-left-then-scan.
    uint64_t find_slot(uint64_t fq, uint64_t fr) const {
        if(!occ_get(fq)) return ~uint64_t(0);
        const uint64_t rnk = occ_dir_.rank1(occ_w(), fq + 1); // 1-indexed home position
        uint64_t i = runend_dir_.select1(re_w(), rnk - 1);
        for(;;) {
            const uint64_t rv = rem(i);
            if(rv == fr) return i;
            if(rv < fr || i == 0 || re_get(i - 1) || !slot_live(i - 1)) return ~uint64_t(0);
            --i;
        }
    }
    bool contains(uint64_t hv) const {const auto p = parts(hv); return find_slot(p.first, p.second) != ~uint64_t(0);}
    bool may_contain(uint64_t hv) const {return contains(hv);}
    uint64_t count(uint64_t hv) const {const auto p = parts(hv); const uint64_t s = find_slot(p.first, p.second); return s == ~uint64_t(0) ? 0: cnt(s);}

    // shift slots [pos, empty-1] into [pos+1, empty] over rem/cnt/runend
    void shift_range_right(uint64_t pos, uint64_t empty) {
        if(fuse_)
            std::memmove(slot_.data() + pos + 1, slot_.data() + pos, (empty - pos) * sizeof(uint64_t));
        else {
            detail::packed_shift_right(remraw_.data(), pos, r_, empty);
            detail::packed_shift_right(cntraw_.data(), pos, unsigned(VAL_BITS), empty);
        }
        detail::packed_shift_right(re_w(), pos, 1, empty);
        re_clear(pos);
        runend_dir_.refresh(re_w(), runend_words_, pos, empty);
    }
    // shift slots [pos+1, empty] into [pos, empty-1]; slot empty freed
    void shift_range_left(uint64_t pos, uint64_t empty) {
        if(fuse_) {
            std::memmove(slot_.data() + pos, slot_.data() + pos + 1, (empty - pos) * sizeof(uint64_t));
            slot_[empty] = 0;
        } else {
            detail::packed_shift_left(remraw_.data(), pos, r_, empty);
            detail::packed_shift_left(cntraw_.data(), pos, unsigned(VAL_BITS), empty);
        }
        detail::packed_shift_left(re_w(), pos, 1, empty);
        runend_dir_.refresh(re_w(), runend_words_, pos, empty);
    }

    // grow when a further insert would push past the 0.9 load cap
    uint64_t ceiling() const {return nslots_ - nslots_ / 10;}
    void grow_if_full() {if(HEDLEY_UNLIKELY(used_ >= ceiling())) grow();}

    void add(uint64_t hv) {
        grow_if_full(); // grow first: parts() below must use the new r_
        const auto [fq, fr] = parts(hv);
        if(occ_get(fq)) {
            // one run_of serves membership check and sorted insert position
            auto [s, e] = run_of(fq);
            uint64_t p = s;
            for(;;) {
                const uint64_t rv = rem(p);
                if(rv == fr) { // membership: bump the counter
                    if(cnt(p) < SAT) set_cnt(p, cnt(p) + 1);
                    return;
                }
                if(p == e || rv > fr) break;
                ++p;
            }
            if(p == e && rem(p) < fr) ++p; // append beyond runend
            uint64_t empty = first_empty(p);
            if(HEDLEY_UNLIKELY(empty == ~uint64_t(0))) { grow(); return add(hv); }
            shift_range_right(p, empty);
            set_slot(p, fr, 1); ++used_;
            if(p == e + 1) { // appended past old runend: move runend
                re_clear(e);
                re_set(p);
                runend_dir_.refresh(re_w(), runend_words_, e, p);
            }
            return;
        }
        insert_new_key(hv, 1); // known-absent new home; grows+retries on tail-full
    }

    // ---- bulk load (paper sec 3.5 observation on this rank-ordered core):
    // keys presented in ASCENDING full-hash order have ascending homes, so
    // each new home appends its run at the live-region end -- no shifting.
    // *It dereferences to (hash, count) pairs sorted by hash ascending.
    // Existing homes take the normal local in-run path (bump / sorted
    // insert, which may shift locally); a mid-batch grow resumes sorted.
    // The append cursor is validated before each use; any violation falls
    // back to the general rank/select position.
    template<typename It>
    void insert_sorted_counts(It first, It last) {
        uint64_t cursor = ~uint64_t(0); // next append slot; ~0 = derive
        for(; first != last; ++first) {
            grow_if_full();
            const uint64_t hv = first->first;
            const uint64_t count_ = first->second;
            const auto [fq, fr] = parts(hv);
            if(occ_get(fq)) {
                auto [s, e] = run_of(fq);
                uint64_t p = s;
                for(;;) {
                    const uint64_t rv = rem(p);
                    if(rv == fr) { // membership: bump the counter
                        if(cnt(p) < SAT) set_cnt(p, std::min<uint64_t>(SAT, cnt(p) + count_));
                        break;
                    }
                    if(p == e || rv > fr) {
                        if(p == e && rem(p) < fr) ++p; // append beyond runend
                        const uint64_t empty = first_empty(p);
                        shift_range_right(p, empty); // refreshes the runend dir
                        set_slot(p, fr, std::min<uint64_t>(count_, SAT)); ++used_;
                        if(p == e + 1) { // appended past old runend: move runend
                            re_clear(e);
                            re_set(p);
                            runend_dir_.refresh(re_w(), runend_words_, e, p);
                        }
                        break;
                    }
                    ++p;
                }
                cursor = ~uint64_t(0); // a shift may have moved the end
                continue;
            }
            uint64_t pos;
            if(cursor != ~uint64_t(0) && cursor < xnslots_ && !slot_live(cursor)
               && (cursor == 0 || slot_live(cursor - 1))) {
                pos = cursor; // validated append point: no shift needed
            } else {
                const uint64_t rnk = occ_dir_.rank1(occ_w(), fq); // occupied homes before fq
                pos = rnk == 0 ? 0: runend_dir_.select1(re_w(), rnk - 1) + 1;
            }
            if(HEDLEY_UNLIKELY(pos >= xnslots_)) { grow(); continue; }
            if(slot_live(pos)) { // defensive: not a clean append; general path
                const uint64_t empty = first_empty(pos);
                if(HEDLEY_UNLIKELY(empty == ~uint64_t(0))) { grow(); continue; }
                shift_range_right(pos, empty); // refreshes the runend dir
            }
            set_slot(pos, fr, std::min<uint64_t>(count_, SAT)); ++used_;
            ++ndistinct_;
            occ_set(fq);
            occ_dir_.refresh(occ_w(), occ_words_, fq, fq);
            re_set(pos);
            runend_dir_.refresh(re_w(), runend_words_, pos, pos);
            cursor = pos + 1;
        }
    }
    // hash-only convenience wrapper (count 1 per hash)
    template<typename It>
    void insert_sorted(It first, It last) {
        std::vector<std::pair<uint64_t, uint64_t>> tmp;
        tmp.reserve(std::distance(first, last));
        for(; first != last; ++first) tmp.emplace_back(*first, 1);
        insert_sorted_counts(tmp.begin(), tmp.end());
    }
    // buffered streaming wrapper: aggregate a random-order stream of KEYS
    // (hashed once), flush sorted at the threshold. The buffer never
    // outlives the call, so queries are consistent when it returns.
    template<typename It>
    void insert_buffered(It first, It last, size_t threshold = 8192) {
        std::unordered_map<uint64_t, uint64_t> buf; // hash -> pending count
        buf.reserve(threshold * 2);
        auto flush = [&]() {
            if(buf.empty()) return;
            std::vector<std::pair<uint64_t, uint64_t>> keys;
            keys.reserve(buf.size());
            for(auto &kv: buf) keys.emplace_back(kv.first, kv.second);
            std::sort(keys.begin(), keys.end());
            insert_sorted_counts(keys.begin(), keys.end());
            buf.clear();
        };
        for(; first != last; ++first) {
            ++buf[hf_(*first)];
            if(buf.size() >= threshold) flush();
        }
        flush();
    }

    // insert a known-absent KEY with count: decomposes per attempt and
    // grows+retries if the table tail is full (no empty slot to shift into)
    void insert_new_key(uint64_t hv, uint64_t count_) {
        for(;;) {
            const auto [fq, fr] = parts(hv);
            if(!occ_get(fq)) {
                // place the new run right after the previous home's run
                // (runs are stored in home order table-wide); the shift
                // stops at the first empty slot, so holes between clusters
                // keep shifts local
                const uint64_t rnk = occ_dir_.rank1(occ_w(), fq); // occupied homes before fq
                const uint64_t pos = rnk == 0 ? 0: runend_dir_.select1(re_w(), rnk - 1) + 1;
                const uint64_t empty = first_empty(pos);
                if(HEDLEY_UNLIKELY(empty == ~uint64_t(0))) { grow(); continue; }
                shift_range_right(pos, empty); // refreshes the runend dir
                set_slot(pos, fr, count_); ++used_;
                ++ndistinct_;
                occ_set(fq);
                occ_dir_.refresh(occ_w(), occ_words_, fq, fq);
                re_set(pos);
                runend_dir_.refresh(re_w(), runend_words_, pos, pos);
                return;
            }
            auto [s, e] = run_of(fq);
            uint64_t p = s;
            while(p != e && rem(p) <= fr) ++p;
            if(p == e && rem(p) < fr) ++p; // append beyond runend
            const uint64_t empty = first_empty(p);
            if(HEDLEY_UNLIKELY(empty == ~uint64_t(0))) { grow(); continue; }
            shift_range_right(p, empty); // refreshes the runend dir
            set_slot(p, fr, count_); ++used_;
            if(p == e + 1) { // appended past old runend: move runend
                re_clear(e);
                re_set(p);
                runend_dir_.refresh(re_w(), runend_words_, e, p);
            }
            return;
        }
    }

    // decrement one occurrence; erases the key when the counter hits zero
    bool dec(uint64_t hv) {
        const auto [fq, fr] = parts(hv);
        const uint64_t s = find_slot(fq, fr);
        if(s == ~uint64_t(0)) return false;
        const uint64_t c = cnt(s);
        if(c > 1) {set_cnt(s, c - 1); return true;}
        return erase_found(fq, fr, s);
    }
    bool del(uint64_t hv) {return dec(hv);}
    // ---- STL-style map surface (exact mode is an integer->integer map) ----
    void insert(uint64_t hv) {add(hv);}
    void insert(const std::pair<uint64_t, uint64_t> &kv) {add_n(kv.first, kv.second);}
    void emplace(uint64_t hv, uint64_t v = 1) {add_n(hv, v);}
    void add_n(uint64_t hv, uint64_t v) {
        while(v--) add(hv);
    }
    uint64_t at(uint64_t hv) const {return count(hv);}
    class counted_ref {
        cqfbase_t *f_;
        uint64_t hv_;
    public:
        counted_ref(cqfbase_t *f, uint64_t hv): f_(f), hv_(hv) {}
        operator uint64_t() const {return f_->count(hv_);}
        counted_ref &operator=(uint64_t v) {
            uint64_t cur = f_->count(hv_);
            if(v > cur) {f_->add_n(hv_, v - cur);}
            else {while(v < cur--) f_->dec(hv_);}
            return *this;
        }
        counted_ref &operator+=(uint64_t v) {f_->add_n(hv_, v); return *this;}
        counted_ref &operator-=(uint64_t v) {while(v--) f_->dec(hv_); return *this;}
        counted_ref &operator++() {f_->add(hv_); return *this;}
        counted_ref &operator--() {f_->dec(hv_); return *this;}
    };
    counted_ref operator[](uint64_t hv) {return counted_ref(this, hv);}
    // STL-style erase: remove the key entirely (frees its slot)
    bool erase(uint64_t hv) {
        const auto [fq, fr] = parts(hv);
        const uint64_t s = find_slot(fq, fr);
        return s == ~uint64_t(0) ? false: erase_found(fq, fr, s);
    }
private:
    bool erase_found(uint64_t fq, uint64_t, uint64_t s) {
        auto [rs, re] = run_of(fq);
        const bool was_single = rs == re;
        // shift boundary = the cluster's LAST live slot (its last runend),
        // not first_empty: drift holes beyond the cluster end must not be
        // pulled into the shifted range.
        uint64_t ce = re;
        while(ce + 1 < xnslots_ && slot_live(ce + 1)) ce = next_runend_from(ce + 1);
        shift_range_left(s, ce); // refreshes the runend dir
        --used_;
        if(was_single) { // run emptied: home loses its occupied bit
            occ_clear(fq);
            occ_dir_.refresh(occ_w(), occ_words_, fq, fq);
            --ndistinct_;
        } else if(s == re) { // erased the run's last element: move the
            re_set(re - 1);  // runend to the new last slot
            runend_dir_.refresh(re_w(), runend_words_, re - 1, re - 1);
        }
        // cluster may split: no metadata change needed (clusters are defined
        // by nonempty spans)
        return true;
    }
public:
    void clear() {
        if(fuse_) std::fill(slot_.begin(), slot_.end(), 0);
        else {
            std::fill(remraw_.begin(), remraw_.end(), 0);
            std::fill(cntraw_.begin(), cntraw_.end(), 0);
        }
        std::fill(metaraw_.begin(), metaraw_.end(), 0);
        occ_dir_.reset(nslots_);
        runend_dir_.reset(xnslots_);
        used_ = ndistinct_ = 0;
    }
    void reset() {clear();}
    void free() {
        auto tmp(std::move(slot_));
        auto tmp1(std::move(remraw_));
        auto tmp2(std::move(cntraw_));
        auto tmp3(std::move(metaraw_));
        q_ = r_ = 0; nslots_ = used_ = ndistinct_ = 0;
    }

    // 2x grow: q+1, r-1; reinsert all live (key, count) via the insert path
    void grow() {
        if(q_ + 1 > 40 || KEY_BITS - (q_ + 1) < 2)
            throw std::overflow_error("[cqf] cannot grow: remainder bits exhausted");

        std::vector<std::pair<uint64_t, uint64_t>> els;
        els.reserve(used_);
        for(uint64_t i = 0; i < xnslots_; ++i)
            if(cnt(i)) els.push_back({key_of_home(i), cnt(i)});
        init(q_ + 1);
        for(const auto &kv: els)
            insert_new_key(kv.first, kv.second); // re-decomposes; grows+retries
        occ_dir_.rebuild(occ_w(), occ_words_);
        runend_dir_.rebuild(re_w(), runend_words_);
    }
    // multiset-union of counters (identical parameters): reinsert all
    // (home, rem, cnt) entries of the SMALLER into a copy of the LARGER.
    cqfbase_t &operator+=(const cqfbase_t &o) {
        if(q_ != o.q_ || r_ != o.r_)
            throw std::invalid_argument(std::string("[cqf] merge requires matching q (") + std::to_string(q_) + " vs " + std::to_string(o.q_) + ")");
        if(o.used_ == 0) return *this;
        if(used_ == 0) {*this = o; return *this;}
        const cqfbase_t *small = &o, *large = this;
        if(o.used_ > used_) std::swap(small, large);
        cqfbase_t dst(*large);
        for(uint64_t i = 0; i < small->nslots_; ++i) {
            if(small->cnt(i)) {
                const uint64_t hv = small->key_of_home(i);
                dst.grow_if_full(); // grow BEFORE decomposing hv with the new r_
                const auto p = dst.parts(hv);
                const uint64_t ex = dst.find_slot(p.first, p.second);
                if(ex != ~uint64_t(0)) dst.set_cnt(ex, std::min<uint64_t>(SAT, dst.cnt(ex) + small->cnt(i)));
                else dst.insert_new_key(hv, small->cnt(i)); // re-decomposes after any tail-full grow
            }
        }
        *this = std::move(dst);
        return *this;
    }
    cqfbase_t operator+(const cqfbase_t &o) const {auto r = *this; r += o; return r;}
    // compose with mismatched (KEY_BITS, VAL_BITS): rehash the other here
    template<size_t KB2, size_t VB2>
    void compose(const cqfbase_t<KB2, VB2, Storage, Hash> &o) {
        for(uint64_t i = 0; i < o.nslots(); ++i) {
            if(o.cnt(i)) {
                const uint64_t hv = o.key_of_home(i); // KB2-bit reduced key
                grow_if_full();                       // grow BEFORE decomposing hv with the new r_
                const auto p = parts(hv);             // rehashed into this layout
                const uint64_t ex = find_slot(p.first, p.second);
                if(ex != ~uint64_t(0) && rem(ex) == p.second)
                    set_cnt(ex, std::min<uint64_t>(SAT, cnt(ex) + o.cnt(i)));
                else insert_new_key(hv, o.cnt(i));
            }
        }
    }
public:
    // the reduced key of the element at slot i: the k-th run of the table
    // (in slot order) pairs with the k-th occupied home (in index order) --
    // the global run/home correspondence, robust to runs drifting left of
    // their homes after deletions.
    uint64_t key_of_home(uint64_t slot) const {
        uint64_t t = slot;
        while(t > 0 && slot_live(t - 1) && !re_get(t - 1)) --t; // run start of slot's run
        const uint64_t e = next_runend_from(t);
        const uint64_t ridx = runend_dir_.rank1(re_w(), e + 1); // 1-based global run index
        return key_of(occ_dir_.select1(occ_w(), ridx - 1), rem(slot));
    }
public:
    // ---- iteration over (key, count) pairs ----
    class iterator {
        const cqfbase_t *f_;
        uint64_t i_;
        void advance() {while(f_ && i_ < f_->xnslots_ && !f_->cnt(i_)) ++i_;}
    public:
        iterator(): f_(nullptr), i_(0) {}
        iterator(const cqfbase_t *f, uint64_t i): f_(f), i_(i) {advance();}
        iterator &operator++() {++i_; advance(); return *this;}
        bool operator!=(const iterator &o) const {return i_ != o.i_ || f_ != o.f_;}
        bool operator==(const iterator &o) const {return !(*this != o);}
        std::pair<uint64_t, uint64_t> operator*() const {return {f_->key_of_home(i_), f_->cnt(i_)};}
    };
    iterator begin() const {return iterator(this, 0);}
    iterator end() const {return iterator(this, xnslots_);}
    // find returns an iterator to the key's slot, or end()
    iterator find(uint64_t hv) const {
        const auto p = parts(hv);
        const uint64_t s = find_slot(p.first, p.second);
        return s == ~uint64_t(0) ? end(): iterator(this, s);
    }

    // ---- serialization (wire format unchanged: packed rem stream, packed
    // cnt stream, occ words, runend words; the fused regime packs/unpacks
    // at this boundary) ----
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[3]{MAGIC, q_, uint32_t(VAL_BITS)};
        CW(fp, header, sizeof(header));
        CW(fp, &used_, sizeof(used_));
        CW(fp, &ndistinct_, sizeof(ndistinct_));
        if(fuse_) {
            std::vector<uint64_t, common::Allocator<uint64_t>> rt(ceil_div(xnslots_ * r_, uint64_t(word_bits)) + 1, 0);
            std::vector<uint64_t, common::Allocator<uint64_t>> ct(ceil_div(xnslots_ * VAL_BITS, uint64_t(word_bits)) + 1, 0);
            for(uint64_t i = 0; i < xnslots_; ++i) {
                detail::packed_set(rt.data(), i, r_, slot_[i] & remmask_);
                detail::packed_set(ct.data(), i, unsigned(VAL_BITS), slot_[i] >> r_);
            }
            CW(fp, rt.data(), rt.size() * sizeof(uint64_t));
            CW(fp, ct.data(), ct.size() * sizeof(uint64_t));
        } else {
            CW(fp, remraw_.data(), remraw_.size() * sizeof(uint64_t));
            CW(fp, cntraw_.data(), cntraw_.size() * sizeof(uint64_t));
        }
        CW(fp, metaraw_.data(), metaraw_.size() * sizeof(uint64_t));
#undef CW
    }
    void write(const char *path) const {
        gzFile fp(gzopen(path, "wb"));
        if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at '") + path + "' for writing");
        write(fp);
        gzclose(fp);
    }
    void write(const std::string &path) const {write(path.data());}
    void read(gzFile fp) {
#define CR(fp, dst, len) do {if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) throw ZlibError(std::string("[cqf] Error reading from file"));} while(0)
        uint32_t header[3];
        CR(fp, header, sizeof(header));
        if(header[0] != MAGIC) throw std::runtime_error("[cqf] bad magic: not a CQF4 (home-anchored) sketch");
        if(header[1] != q_ || header[2] != VAL_BITS) init(header[1]);
        CR(fp, &used_, sizeof(used_));
        CR(fp, &ndistinct_, sizeof(ndistinct_));
        if(fuse_) {
            std::vector<uint64_t, common::Allocator<uint64_t>> rt(ceil_div(xnslots_ * r_, uint64_t(word_bits)) + 1);
            std::vector<uint64_t, common::Allocator<uint64_t>> ct(ceil_div(xnslots_ * VAL_BITS, uint64_t(word_bits)) + 1);
            CR(fp, rt.data(), rt.size() * sizeof(uint64_t));
            CR(fp, ct.data(), ct.size() * sizeof(uint64_t));
            slot_.assign(xnslots_, 0);
            for(uint64_t i = 0; i < xnslots_; ++i)
                slot_[i] = (detail::packed_get(ct.data(), i, unsigned(VAL_BITS)) << r_)
                         | detail::packed_get(rt.data(), i, r_);
        } else {
            CR(fp, remraw_.data(), remraw_.size() * sizeof(uint64_t));
            CR(fp, cntraw_.data(), cntraw_.size() * sizeof(uint64_t));
        }
        CR(fp, metaraw_.data(), metaraw_.size() * sizeof(uint64_t));
        occ_dir_.rebuild(occ_w(), occ_words_);
        runend_dir_.rebuild(re_w(), runend_words_);
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}

    // ---- invariant checker: throws on violation (active in all builds) ----
    // Walks runs in slot order, pairing each with the next occupied home in
    // index order (the global correspondence). Run interiors must be live
    // and sorted; totals and used_ must agree. Runs may sit below their
    // homes (delete drift) -- the correspondence does not rely on position.
    // Also probes the navigation on BOTH paths: directory-assisted and flat
    // scans must agree with a running word-popcount reference (intra-word
    // offsets included) at sampled ranks, and with an incremental set-bit
    // walk at sampled selects -- allocation-free.
    void structure_check() const {
        uint64_t live = 0, distinct = 0, home = 0;
        for(uint64_t h = 0; h < nslots_; ++h) distinct += occ_get(h);
        if(detail::bv_total(re_w(), runend_words_) != distinct)
            throw std::runtime_error("[cqf] I5 violated: runends != occupied");
        if(ndistinct_ != distinct)
            throw std::runtime_error("[cqf] I5 violated: ndistinct != occupied");
        uint64_t s = 0;
        while(s < xnslots_) {
            if(!slot_live(s)) {
                if(re_get(s)) throw std::runtime_error("[cqf] I5 violated: runend on empty slot");
                ++s;
                continue;
            }
            uint64_t e = s;
            while(!re_get(e)) { // run end (must exist: live slot)
                ++e;
                if(e >= xnslots_) throw std::runtime_error("[cqf] I5 violated: runend missing");
            }
            while(home < nslots_ && !occ_get(home)) ++home; // this run's home
            if(home == nslots_)
                throw std::runtime_error("[cqf] I3 violated: run without occupied home");
            // I4: remainders strictly increasing, all interior slots live
            for(uint64_t i = s + 1; i <= e; ++i) {
                if(!slot_live(i))
                    throw std::runtime_error(std::string("[cqf] I5 violated: hole inside run of home ") + std::to_string(home));
                if(rem(i) <= rem(i - 1))
                    throw std::runtime_error(std::string("[cqf] I4 violated: run of home ") + std::to_string(home)
                        + " remainders not increasing at slot " + std::to_string(i));
            }
            live += e - s + 1;
            ++home;
            s = e + 1;
        }
        if(live != used_) throw std::runtime_error("[cqf] I5 violated: used count mismatch");
        // navigation probes, allocation-free: sampled ranks against a running
        // word-popcount scan (probe words spread across the region, offsets
        // exercise the intra-word mask), each checked on BOTH the flat scan
        // and the directory; sampled selects against an incremental set-bit
        // walk, also on both
        {
            const uint64_t *ow = occ_w();
            static constexpr unsigned offs[] = {0, 5, 27, 63};
            const uint64_t step = std::max<uint64_t>(1, occ_words_ / 47);
            uint64_t acc = 0, next_probe = 0;
            for(uint64_t w = 0; w < occ_words_; ++w) {
                if(w >= next_probe) {
                    for(unsigned o: offs) {
                        const uint64_t i = w * word_bits + o;
                        const uint64_t want = acc + (o ? popcount(ow[w] & ((uint64_t(1) << o) - 1)): 0);
                        if(detail::bv_rank1(ow, i) != want || occ_dir_.rank1(ow, i) != want)
                            throw std::runtime_error("[cqf] NAV violated: occ rank mismatch");
                    }
                    next_probe += step;
                }
                acc += popcount(ow[w]);
            }
        }
        // runend ranks likewise, then sampled selects against a set-bit walk
        {
            const uint64_t *rw = re_w();
            static constexpr unsigned offs[] = {0, 5, 27, 63};
            const uint64_t step = std::max<uint64_t>(1, runend_words_ / 47);
            uint64_t acc = 0, next_probe = 0;
            for(uint64_t w = 0; w < runend_words_; ++w) {
                if(w >= next_probe) {
                    for(unsigned o: offs) {
                        const uint64_t i = w * word_bits + o;
                        const uint64_t want = acc + (o ? popcount(rw[w] & ((uint64_t(1) << o) - 1)): 0);
                        if(detail::bv_rank1(rw, i) != want || runend_dir_.rank1(rw, i) != want)
                            throw std::runtime_error("[cqf] NAV violated: runend rank mismatch");
                    }
                    next_probe += step;
                }
                acc += popcount(rw[w]);
            }
            uint64_t k = 0, next_k = 0;
            const uint64_t kstep = std::max<uint64_t>(1, distinct / 31);
            for(uint64_t w = 0; w < runend_words_ && k < distinct; ++w) {
                uint64_t word = rw[w];
                while(word) {
                    const uint64_t pos = w * word_bits + ctz(word);
                    if(k == next_k) {
                        if(detail::bv_select1(rw, runend_words_, k) != pos || runend_dir_.select1(rw, k) != pos)
                            throw std::runtime_error("[cqf] NAV violated: runend select mismatch");
                        next_k += kstep;
                    }
                    ++k;
                    word &= word - 1;
                }
            }
        }
    }
};

// Block storage layouts (round 16n item 2): FUSED keeps the 16m
// [hash|counter] register per slot; SOA splits parallel arrays
// (hashes | counters). Both keep the per-block live array and the
// superblock rollup (sums over 64 blocks) for metadata-first scans and
// range skipping. Measured A/B.
struct bcqf_fused_layout {};
struct bcqf_soa_layout {};

// ---- block-confined counting filter (bcqf_t) ----
// DESIGN RECORD
//
// What this is. A counting quotient filter for exact reduced-key
// spaces (KEY_BITS + VAL_BITS <= 64; the flagship is 48+16 = 64 bits,
// the exact config the literature's filters cannot serve). The slot
// stores the FULL reduced hash plus the counter. Measured against
// Reference CQF at q=20 (measured comparisons in the bench notes;
// bench/cqf_restructure.md): single insert 88-97 ns vs their 193-475,
// builds 14-16 vs 5.9-8.4 M/s, delete 22-27 ns vs 104-175, query-hit at
// or above parity, memory 1.38x, query-miss 1.5-2.3x behind (the one
// honest loss).
//
// Why full-hash storage, not quotient/remainder. All three reference
// papers split the hash: quotient selects a home, the remainder is
// stored, and the filter spends metadata (occ/runend/offsets, 2.125-
// 3.125 bits/slot) to recover the home at query time. Storing the
// whole reduced hash instead has four consequences, each worth more
// than the extra stored bits at KEY_BITS <= 48:
//   1. The stored hash IS the routing information. Both candidate
//      blocks recompute from it (VQF S3.4's pattern strengthened); no
//      occ/runend bitvectors and no rank/select directories exist at
//      all. The only metadata left is one live counter per block and a
//      superblock rollup.
//   2. Iteration enumerates exact keys. AQF's reverse map -- a whole
//      on-disk auxiliary structure with a map update per insert -- is
//      subsumed; merge/resize/grow enumerate and reinsert with zero
//      bookkeeping (union measured at 37-49 M keys/s, ~4,700x the
//      reference qf_merge at this configuration).
//   3. The filter is false-positive-free within its key space: the
//      compare is exact. AQF's adaptivity machinery exists only to
//      repair false positives; there are none to repair. The width
//      knob (item 3) trades the size of the exact key space, not
//      correctness within it.
//   4. Counter typing is free: the counter field is dedicated (0 =
//      empty sentinel, 1..SAT = count). AQF spends a third metadata
//      bit to distinguish extension from counter slots; here the
//      distinction is the value range itself.
//
// Why no grouping, no run structure, no slot ordering. The CQF paper's
// own S6.8 diagnosis is that inserts are bottlenecked by shifting
// elements around; rank/select optimization sped lookups 2-4x and
// inserts barely moved, because displacement length is placement
// geometry (their Theorem 1 blows up as load approaches 1). Blocking
// attacks the geometry: with power-of-two-choices among two candidate
// blocks (VQF S3.1, Berenbrink et al. max-load bound), slots are
// UNSORTED within a 32-slot block, so an insert never displaces
// anything -- it writes any free slot of the chosen block, and a
// delete clears a slot. Displacement is removed by construction, not
// accelerated. What is given up, deliberately:
//   - Block order is NOT hash order: a key may reside in its secondary
//     block (blk2 is a multiply-mix, non-monotone). Exact windowed
//     enumeration therefore scans in block order with an hv test,
//     bulk-skipping only empty superblocks via the rollup; window
//     sharding of set ops is measured counterproductive at n = 2^20
//     (per-shard rescans) and documented as such.
//   - The 0.86-0.9 ceiling: past ~0.9 block load, both-candidates-full
//     events begin. There is no overflow structure (round 16n deleted
//     it): both full -> grow() and retry, and a max_load knob (0.9
//     default) grows proactively. Amortized insert at sustained 0.95
//     with grows: 4.5-4.7 M/s, worst-case single-insert latency = the
//     grow spike (74-79 ms at q~20; incremental/lazy rehash is the
//     recorded future option).
//   - Query-miss touches two candidate blocks (two dependent cache
//     lines) instead of the reference's metadata-first short circuit;
//     the metadata-first empty-block skip (item 2) recovers part of
//     it. Measured residual: 1.5-2.3x.
//
// Storage and width parameterization. The slot word is width-selected
// (u16/u32/u64 by KEY_BITS+VAL_BITS) with sentinel liveness in every
// width and width-specialized AVX2 block scans (4x64/8x32/16x16 lane
// groups, movemask lane locate; aarch64 NEON for u64; scalar under
// SKETCH_CQF_NO_SIMD). An SoA variant (parallel hashes/counters
// arrays) was implemented and A/B-measured (item 2): it wins only
// query-miss and costs 25% memory plus hit/dec regression, so the
// fused register layout is the default and SoA remains available as
// bcqf_t<..., bcqf_soa_layout>.
//
// Concurrency. add_mt implements the VQF S6.3 lock-bit pattern with
// the locks in a cache-line padded side array (the live word stays a
// plain counter): the shortcut path takes ONE block lock, two-choice
// takes both in increasing order, grow runs under an exclusive table
// mutex, and readers stay lock-free (a slot write is a single word
// store, so a racing reader sees absence or a complete slot). Measured
// (item 6): exact contents at T = 1/2/4/8; the stress phase forced a
// yield backoff in the spin (oversubscribed cores starve the holder
// otherwise) and explicit copy/move semantics.
//
// Iteration contract. begin()/end() and every range API walk blocks in
// ascending order, slots ascending within a block, skipping empty
// blocks and empty superblocks; single-pass, forward-only. No ordering
// guarantee across blocks by hash value (see above).
//
// Wire format. CQF6: logical (hash, count) u64 stream + live counts;
// CQF5 files (the 16m layout with an overflow map) remain readable by
// re-inserting their overflow entries.

template<size_t KEY_BITS, size_t VAL_BITS, class Hash=WangHash, class Layout=bcqf_fused_layout>
class bcqf_t {
    static constexpr bool SOA = std::is_same<Layout, bcqf_soa_layout>::value;
    static_assert(KEY_BITS >= 8 && KEY_BITS <= 48, "bcqf stores the full reduced hash: KEY_BITS in [8, 48]");
    static_assert(VAL_BITS >= 1 && VAL_BITS <= 16, "VAL_BITS in [1, 16]");
    static_assert(KEY_BITS + VAL_BITS <= 64, "hash + counter must fit one slot register");
public:
    static constexpr size_t key_bits = KEY_BITS;
    static constexpr size_t val_bits = VAL_BITS;
    static constexpr bool exact = true; // full reduced hash stored: FP-free in key space
    static constexpr uint64_t SAT = (uint64_t(1) << VAL_BITS) - 1;
    static constexpr uint64_t HVMASK = (uint64_t(1) << KEY_BITS) - 1;
    static constexpr unsigned BLK = 32;   // slots per block
    static constexpr unsigned BLK_SHIFT = 5;
    static constexpr unsigned SUPER = 64; // blocks per rollup superblock
    static constexpr unsigned SLOT_BITS = unsigned(KEY_BITS + VAL_BITS);
    // slot word width: u16/u32/u64 by need; liveness is the zero counter
    // sentinel in every width, so scans stay branchless single-pass
    using W = typename std::conditional<SLOT_BITS <= 16, uint16_t,
              typename std::conditional<SLOT_BITS <= 32, uint32_t, uint64_t>::type>::type;
public:
    static constexpr unsigned slot_bits = SLOT_BITS;
    static constexpr unsigned slot_bytes = sizeof(W);
private:
    uint32_t q_ = 0;
    uint64_t nslots_ = 0, nblocks_ = 0; // nslots_ = nblocks_ * BLK = 2^q
    std::atomic<uint64_t> used_ {0};    // live keys
    double max_load_ = 0.9;             // proactive grow threshold (1.0 = grow on full only)
    std::vector<W, common::Allocator<W>> slots_; // [hash | cnt] (fused)
    std::vector<uint64_t, common::Allocator<uint64_t>> hashes_; // hash per slot (SoA)
    std::vector<uint16_t, common::Allocator<uint16_t>> cnts_;   // counter per slot (SoA)
    std::vector<uint16_t, common::Allocator<uint16_t>> live_;  // per-block live counts
    std::vector<std::atomic<uint32_t>> rollup_; // std::allocator: atomics are not copyable // live sums over SUPER blocks
    // block locks (item 6): one per block, cache-line padded (16 x u32
    // stride) so CAS traffic on one block's lock does not invalidate
    // neighboring blocks' locks; 1 = held
    mutable std::vector<std::atomic<uint32_t>> locks_;
    mutable std::shared_mutex grow_mu_; // add_mt: shared for placement, unique for grow
    Hash hf_;
    static constexpr unsigned LOCK_PAD = 16; // u32 words per lock (64B); one lock per SUPERBLOCK blocks(per-block locks cost 2 MB at q=20 and erased the compact memory win)
    static constexpr uint32_t MAGIC = 0x43514636; // "CQF6" (block layout, no overflow)
    static constexpr uint32_t MAGIC5 = 0x43514635; // "CQF5" (legacy, overflow map)
    static constexpr uint64_t PHI = 0x9E3779B97F4A7C15ULL;

    INLINE uint64_t blk1(uint64_t hv) const { return hv >> (KEY_BITS - (q_ - BLK_SHIFT)); }
    INLINE uint64_t blk2(uint64_t hv) const {
        const uint64_t m = hv * PHI;
        return q_ > BLK_SHIFT ? (m >> (64 - (q_ - BLK_SHIFT))) & (nblocks_ - 1): 0;
    }
    // ---- storage accessors (layout- and width-dispatched; dead
    // branches fold away)
    INLINE uint64_t get_hv(uint64_t i) const {
        return SOA ? (hashes_[i] & HVMASK): (uint64_t(slots_[i]) & HVMASK);
    }
    INLINE uint64_t get_cnt(uint64_t i) const {
        return SOA ? uint64_t(cnts_[i]): (uint64_t(slots_[i]) >> KEY_BITS);
    }
    INLINE bool slot_live(uint64_t i) const {return get_cnt(i) != 0;}
    INLINE void put(uint64_t i, uint64_t hv, uint64_t c) {
        if(SOA) {hashes_[i] = hv & HVMASK; cnts_[i] = uint16_t(c);}
        else slots_[i] = W((hv & HVMASK) | (c << KEY_BITS));
    }
    static INLINE bool slot_match(uint64_t hv, uint64_t hv_field, uint64_t cnt) {
        return hv_field == hv && cnt != 0;
    }
    template<typename T>
    static unsigned block_free_fused_t(const T *blk) {
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
        // zero-compare of the counter field, width-specialized lanes
        if(sizeof(T) == 8) {
            const __m256i cm = _mm256_set1_epi64x(~uint64_t(0) << KEY_BITS);
            for(unsigned i = 0; i < BLK; i += 4) {
                const __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(blk + i)), cm);
                const unsigned m = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(v, _mm256_setzero_si256())));
                if(m) return i + ctz(m);
            }
            return BLK;
        } else if(sizeof(T) == 4) {
            const __m256i cm = _mm256_set1_epi32(~uint32_t(0) << KEY_BITS);
            for(unsigned i = 0; i < BLK; i += 8) {
                const __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(blk + i)), cm);
                const unsigned m = unsigned(_mm256_movemask_ps((__m256)_mm256_cmpeq_epi32(v, _mm256_setzero_si256())));
                if(m) return i + ctz(m);
            }
            return BLK;
        } else {
            const __m256i cm = _mm256_set1_epi16(~uint16_t(0) << KEY_BITS);
            for(unsigned i = 0; i < BLK; i += 16) {
                const __m256i v = _mm256_and_si256(_mm256_loadu_si256((const __m256i *)(blk + i)), cm);
                const unsigned m8 = unsigned(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, _mm256_setzero_si256())));
                if(m8) return i + unsigned(ctz(m8) >> 1);
            }
            return BLK;
        }
#else
        for(unsigned i = 0; i < BLK; ++i)
            if(T(blk[i] >> KEY_BITS) == 0) return i;
        return BLK;
#endif
    }
    static unsigned block_free_fused(const W *blk) {return block_free_fused_t<W>(blk);}
    static unsigned block_free_soa(const uint16_t *cnt) {
        // 16-bit lanes: two blocks of 16 per 256-bit vector, zero-compare
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
        for(unsigned i = 0; i < BLK; i += 16) {
            const __m256i v = _mm256_loadu_si256((const __m256i *)(cnt + i));
            const unsigned m = unsigned(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, _mm256_setzero_si256())));
            if(m) return i + (ctz(m) >> 1);
        }
        return BLK;
#else
        for(unsigned i = 0; i < BLK; ++i)
            if(cnt[i] == 0) return i;
        return BLK;
#endif
    }
    unsigned block_free(uint64_t b) const {
        return SOA ? block_free_soa(cnts_.data() + (b << BLK_SHIFT))
                   : block_free_fused(slots_.data() + (b << BLK_SHIFT));
    }
    template<typename T>
    static bool block_find_fused_t(const T *blk, uint64_t hv, uint64_t &pos) {
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
        // per-group branchless vector compare, early exit between groups;
        // the movemask itself locates the lane. Width-specialized:
        // 4/8/16 lanes per group for u64/u32/u16 slots.
        if(sizeof(T) == 8) {
            const __m256i vh = _mm256_set1_epi64x(hv);
            const __m256i hvm = _mm256_set1_epi64x(HVMASK);
            const __m256i cm = _mm256_set1_epi64x(~uint64_t(0) << KEY_BITS);
            const __m256i ones = _mm256_set1_epi64x(~uint64_t(0));
            const __m256i zero = _mm256_setzero_si256();
            for(unsigned i = 0; i < BLK; i += 8) {
                const __m256i s0 = _mm256_loadu_si256((const __m256i *)(blk + i));
                const __m256i s1 = _mm256_loadu_si256((const __m256i *)(blk + i + 4));
                const unsigned m0 = unsigned(_mm256_movemask_pd((__m256d)_mm256_and_si256(_mm256_cmpeq_epi64(_mm256_and_si256(s0, hvm), vh), _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(s0, cm), zero), ones))));
                const unsigned m1 = unsigned(_mm256_movemask_pd((__m256d)_mm256_and_si256(_mm256_cmpeq_epi64(_mm256_and_si256(s1, hvm), vh), _mm256_andnot_si256(_mm256_cmpeq_epi64(_mm256_and_si256(s1, cm), zero), ones))));
                const unsigned m = m0 | (m1 << 4);
                if(m) { pos = i + ctz(m); return true; }
            }
            return false;
        } else if(sizeof(T) == 4) {
            const __m256i vh = _mm256_set1_epi32(uint32_t(hv));
            const __m256i hvm = _mm256_set1_epi32(uint32_t(HVMASK));
            const __m256i cm = _mm256_set1_epi32(~uint32_t(0) << KEY_BITS);
            const __m256i ones = _mm256_set1_epi32(~uint32_t(0));
            const __m256i zero = _mm256_setzero_si256();
            for(unsigned i = 0; i < BLK; i += 8) {
                const __m256i s = _mm256_loadu_si256((const __m256i *)(blk + i));
                const unsigned m = unsigned(_mm256_movemask_ps((__m256)_mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(s, hvm), vh), _mm256_andnot_si256(_mm256_cmpeq_epi32(_mm256_and_si256(s, cm), zero), ones))));
                if(m) { pos = i + ctz(m); return true; }
            }
            return false;
        } else {
            const __m256i vh = _mm256_set1_epi16(uint16_t(hv));
            const __m256i hvm = _mm256_set1_epi16(uint16_t(HVMASK));
            const __m256i cm = _mm256_set1_epi16(~uint16_t(0) << KEY_BITS);
            const __m256i ones = _mm256_set1_epi16(~uint16_t(0));
            const __m256i zero = _mm256_setzero_si256();
            for(unsigned i = 0; i < BLK; i += 16) {
                const __m256i s = _mm256_loadu_si256((const __m256i *)(blk + i));
                const __m256i eq = _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(s, hvm), vh),
                                                    _mm256_andnot_si256(_mm256_cmpeq_epi16(_mm256_and_si256(s, cm), zero), ones));
                const unsigned m8 = unsigned(_mm256_movemask_epi8(eq));
                if(m8) { pos = i + unsigned(ctz(m8) >> 1); return true; }
            }
            return false;
        }
#elif defined(__ARM_NEON) && !defined(SKETCH_CQF_NO_SIMD)
        if(sizeof(T) == 8) { // aarch64 NEON two-lane (u64 slots only)
            const uint64x2_t vh = vdupq_n_u64(hv);
            const uint64x2_t hvm = vdupq_n_u64(HVMASK);
            const uint64x2_t cm = vdupq_n_u64(~uint64_t(0) << KEY_BITS);
            const uint64x2_t zero = vdupq_n_u64(0);
            uint64x2_t a0 = vdupq_n_u64(0), a1 = vdupq_n_u64(0), a2 = vdupq_n_u64(0), a3 = vdupq_n_u64(0);
            for(unsigned i = 0; i < BLK; i += 8) {
                const uint64x2_t s0 = vld1q_u64((const uint64_t *)(blk + i));
                const uint64x2_t s1 = vld1q_u64((const uint64_t *)(blk + i + 2));
                const uint64x2_t s2 = vld1q_u64((const uint64_t *)(blk + i + 4));
                const uint64x2_t s3 = vld1q_u64((const uint64_t *)(blk + i + 6));
                a0 = vorrq_u64(a0, vandq_u64(vceqq_u64(vandq_u64(s0, hvm), vh), vbicq_u64(vdupq_n_u64(~uint64_t(0)), vceqq_u64(vandq_u64(s0, cm), zero))));
                a1 = vorrq_u64(a1, vandq_u64(vceqq_u64(vandq_u64(s1, hvm), vh), vbicq_u64(vdupq_n_u64(~uint64_t(0)), vceqq_u64(vandq_u64(s1, cm), zero))));
                a2 = vorrq_u64(a2, vandq_u64(vceqq_u64(vandq_u64(s2, hvm), vh), vbicq_u64(vdupq_n_u64(~uint64_t(0)), vceqq_u64(vandq_u64(s2, cm), zero))));
                a3 = vorrq_u64(a3, vandq_u64(vceqq_u64(vandq_u64(s3, hvm), vh), vbicq_u64(vdupq_n_u64(~uint64_t(0)), vceqq_u64(vandq_u64(s3, cm), zero))));
            }
            uint64_t r[2];
            vst1q_u64(r, vorrq_u64(vorrq_u64(a0, a1), vorrq_u64(a2, a3)));
            if((r[0] | r[1]) == 0) return false;
            for(unsigned i = 0; i < BLK; ++i)
                if(slot_match(hv, uint64_t(blk[i]) & HVMASK, uint64_t(blk[i]) >> KEY_BITS)) { pos = i; return true; }
            return false;
        }
        // narrower slots on aarch64: scalar
        for(unsigned i = 0; i < BLK; ++i)
            if(slot_match(hv, uint64_t(blk[i]) & HVMASK, uint64_t(blk[i]) >> KEY_BITS)) { pos = i; return true; }
        return false;
#else
        for(unsigned i = 0; i < BLK; ++i)
            if(slot_match(hv, uint64_t(blk[i]) & HVMASK, uint64_t(blk[i]) >> KEY_BITS)) { pos = i; return true; }
        return false;
#endif
    }
    static bool block_find_fused(const W *blk, uint64_t hv, uint64_t &pos) {return block_find_fused_t<W>(blk, hv, pos);}
    static bool block_find_soa(const uint64_t *hashes, const uint16_t *cnts, uint64_t hv, uint64_t &pos) {
#if defined(__AVX2__) && !defined(SKETCH_CQF_NO_SIMD)
        const __m256i vh = _mm256_set1_epi64x(hv);
        const __m256i hvm = _mm256_set1_epi64x(HVMASK);
        for(unsigned i = 0; i < BLK; i += 8) {
            const unsigned m0 = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(hashes + i)), hvm), vh)));
            const unsigned m1 = unsigned(_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_loadu_si256((const __m256i *)(hashes + i + 4)), hvm), vh)));
            const unsigned m = m0 | (m1 << 4);
            if(m) {
                const unsigned j = i + ctz(m);
                if(cnts[j]) { pos = j; return true; } // hash-equal slot: live check
            }
        }
        return false;
#elif defined(__ARM_NEON) && !defined(SKETCH_CQF_NO_SIMD)
        const uint64x2_t vh = vdupq_n_u64(hv);
        const uint64x2_t hvm = vdupq_n_u64(HVMASK);
        uint64x2_t a0 = vdupq_n_u64(0), a1 = vdupq_n_u64(0), a2 = vdupq_n_u64(0), a3 = vdupq_n_u64(0);
        for(unsigned i = 0; i < BLK; i += 8) {
            a0 = vorrq_u64(a0, vceqq_u64(vandq_u64(vld1q_u64(hashes + i), hvm), vh));
            a1 = vorrq_u64(a1, vceqq_u64(vandq_u64(vld1q_u64(hashes + i + 2), hvm), vh));
            a2 = vorrq_u64(a2, vceqq_u64(vandq_u64(vld1q_u64(hashes + i + 4), hvm), vh));
            a3 = vorrq_u64(a3, vceqq_u64(vandq_u64(vld1q_u64(hashes + i + 6), hvm), vh));
        }
        uint64_t r[2];
        vst1q_u64(r, vorrq_u64(vorrq_u64(a0, a1), vorrq_u64(a2, a3)));
        if((r[0] | r[1]) == 0) return false;
        for(unsigned i = 0; i < BLK; ++i)
            if(slot_match(hv, hashes[i] & HVMASK, cnts[i])) { pos = i; return true; }
        return false; // unreachable
#else
        for(unsigned i = 0; i < BLK; ++i)
            if(slot_match(hv, hashes[i] & HVMASK, cnts[i])) { pos = i; return true; }
        return false;
#endif
    }
    bool block_find(uint64_t b, uint64_t hv, uint64_t &pos) const {
        return SOA ? block_find_soa(hashes_.data() + (b << BLK_SHIFT), cnts_.data() + (b << BLK_SHIFT), hv, pos)
                   : block_find_fused(slots_.data() + (b << BLK_SHIFT), hv, pos);
    }
    // locate hv's slot across its two candidate blocks; metadata first
    // (an empty block is a definite miss without touching the payload),
    // secondary block prefetched while the primary scans
    bool find_slot(uint64_t hv, uint64_t &b, uint64_t &pos) const {
        const uint64_t b1 = blk1(hv), b2 = blk2(hv);
        if(b2 != b1) __builtin_prefetch(payload(b2), 0, 2);
        // no live-word gating on the query path: at BLK=32 and load >=
        // 0.5 a candidate block is empty with probability ~e^-16 ~= 0,
        // so the metadata read is a guaranteed extra cache miss per
        // query (round-18 profile). Inserts keep their gated paths.
        if(block_find(b1, hv, pos)) { b = b1; return true; }
        if(b2 != b1 && block_find(b2, hv, pos)) { b = b2; return true; }
        return false;
    }
    const void *payload(uint64_t b) const {
        return SOA ? static_cast<const void *>(hashes_.data() + (b << BLK_SHIFT))
                   : static_cast<const void *>(slots_.data() + (b << BLK_SHIFT));
    }
    INLINE void bump_live(uint64_t b, int d) {
        live_[b] = uint16_t(int(live_[b]) + d);
        rollup_[b / SUPER].fetch_add(uint32_t(d), std::memory_order_relaxed);
    }
    // VQF S6.3 pattern, adapted: block locks live in their own cache-line
    // padded array (the live word stays a plain counter); a spin-CAS with
    // an acquire fence, release on unlock. A non-full block always has
    // its lock available (live <= BLK).
    void lock_block(uint64_t b) const {
        std::atomic<uint32_t> &w = locks_[(b / SUPER) * LOCK_PAD];
        uint32_t exp = 0;
        unsigned spins = 0;
        while(!w.compare_exchange_weak(exp, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            exp = 0; // contended: retry with backoff
            if(++spins > 64) { std::this_thread::yield(); spins = 0; } // oversubscribed cores: don't starve the holder
        }
    }
    void unlock_block(uint64_t b) const {
        locks_[(b / SUPER) * LOCK_PAD].store(0, std::memory_order_release);
    }
public:
    using key_type = uint64_t;
    using mapped_type = uint64_t;
    using HashType = Hash;

    explicit bcqf_t(uint32_t q, Hash &&hf=Hash()): hf_(std::move(hf)) {
        if(q < 6 || q > 40 || q > KEY_BITS + BLK_SHIFT)
            throw std::invalid_argument(std::string("[bcqf] need 6 <= q <= 40 and q <= KEY_BITS + 5; got q=") + std::to_string(q));
        init(q);
    }
    // max_load: proactive grow threshold as a load fraction (1.0 grows
    // only when both candidates are full)
    explicit bcqf_t(uint32_t q, double max_load, Hash &&hf=Hash()): max_load_(max_load), hf_(std::move(hf)) {
        if(q < 6 || q > 40 || q > KEY_BITS + BLK_SHIFT)
            throw std::invalid_argument(std::string("[bcqf] need 6 <= q <= 40 and q <= KEY_BITS + 5; got q=") + std::to_string(q));
        if(!(max_load > 0. && max_load <= 1.))
            throw std::invalid_argument(std::string("[bcqf] need 0 < max_load <= 1; got ") + std::to_string(max_load));
        init(q);
    }
    bcqf_t() = delete;
    // copy/move carry the logical state; locks, rollup atomics, and the
    // grow mutex are fresh in the destination
    bcqf_t(const bcqf_t &o): q_(o.q_), nslots_(o.nslots_), nblocks_(o.nblocks_),
        max_load_(o.max_load_), live_(o.live_), hf_(o.hf_) {
        if(!SOA) slots_ = o.slots_;
        else {hashes_ = o.hashes_; cnts_ = o.cnts_;}
        rollup_ = decltype(rollup_)(ceil_div(nblocks_, uint64_t(SUPER)));
        for(size_t s = 0; s < rollup_.size(); ++s) rollup_[s].store(0, std::memory_order_relaxed);
        for(uint64_t b = 0; b < nblocks_; ++b) rollup_[b / SUPER].fetch_add(uint32_t(live_[b]), std::memory_order_relaxed);
        locks_ = decltype(locks_)(size_t(ceil_div(nblocks_, uint64_t(SUPER))) * LOCK_PAD);
        for(auto &l: locks_) l.store(0, std::memory_order_relaxed);
        used_.store(o.used_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    bcqf_t(bcqf_t &&o) = default;
    bcqf_t &operator=(const bcqf_t &o) {
        if(this != &o) {
            q_ = o.q_; nslots_ = o.nslots_; nblocks_ = o.nblocks_; max_load_ = o.max_load_;
            hf_ = o.hf_;
            if(!SOA) slots_ = o.slots_;
            else {hashes_ = o.hashes_; cnts_ = o.cnts_;}
            live_ = o.live_;
            rollup_ = decltype(rollup_)(ceil_div(nblocks_, uint64_t(SUPER)));
            for(size_t s = 0; s < rollup_.size(); ++s) rollup_[s].store(0, std::memory_order_relaxed);
            for(uint64_t b = 0; b < nblocks_; ++b) rollup_[b / SUPER].fetch_add(uint32_t(live_[b]), std::memory_order_relaxed);
            locks_ = decltype(locks_)(size_t(ceil_div(nblocks_, uint64_t(SUPER))) * LOCK_PAD);
            for(auto &l: locks_) l.store(0, std::memory_order_relaxed);
            used_.store(o.used_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }
    bcqf_t &operator=(bcqf_t &&o) = default;
    struct from_path_t {explicit from_path_t() = default;};
    struct from_gz_t {explicit from_gz_t() = default;};
    static constexpr from_path_t from_path{};
    static constexpr from_gz_t from_gz{};
    bcqf_t(from_path_t, const std::string &path): bcqf_t(6) {read(path);}
    bcqf_t(from_gz_t, gzFile fp): bcqf_t(6) {read(fp);}

    void init(uint32_t q) {
        q_ = q;
        nslots_ = uint64_t(1) << q;
        nblocks_ = nslots_ / BLK;
        if(!SOA) slots_.assign(nslots_, 0);
        else {
            hashes_.assign(nslots_, 0);
            cnts_.assign(nslots_, 0);
        }
        live_.assign(nblocks_, 0);
        rollup_ = decltype(rollup_)(ceil_div(nblocks_, uint64_t(SUPER))); // fresh: atomics are not movable in-place
        for(auto &r: rollup_) r.store(0, std::memory_order_relaxed);
        locks_ = decltype(locks_)(size_t(ceil_div(nblocks_, uint64_t(SUPER))) * LOCK_PAD);
        for(auto &l: locks_) l.store(0, std::memory_order_relaxed);
        used_ = 0;
    }
    uint32_t q() const {return q_;}
    uint64_t nslots() const {return nslots_;}
    uint64_t size() const {return used_.load();}
    bool empty() const {return used_.load() == 0;}
    double load_factor() const {return double(used_.load()) / double(nslots_);}
    double max_load() const {return max_load_;}
    double fpr_bound() const {return 0.;} // full reduced hash stored
    std::pair<size_t, size_t> est_memory_usage() const {
        const size_t payload = SOA ? hashes_.size() * sizeof(uint64_t) + cnts_.size() * sizeof(uint16_t)
                                   : slots_.size() * sizeof(W);
        return std::make_pair(sizeof(*this),
                              payload + live_.size() * sizeof(uint16_t) + rollup_.size() * sizeof(uint32_t)
                              + locks_.size() * sizeof(uint32_t)); // lock array included
    }
    INLINE uint64_t hash(uint64_t v) const {return hf_(v);}
    INLINE void addh(uint64_t e) {add(hf_(e));}
    INLINE uint64_t reduce(uint64_t hv) const {return hv & HVMASK;}

    // raw read-only slot access (analytics/tooling; enumeration for
    // merge and compaction goes through the iterators/range instead)
    uint64_t slot_hash(uint64_t i) const {return get_hv(i);}
    uint64_t slot_count(uint64_t i) const {return get_cnt(i);}
    uint64_t count(uint64_t hv) const {
        hv &= HVMASK;
        uint64_t b, pos;
        if(!find_slot(hv, b, pos)) return 0;
        return get_cnt((b << BLK_SHIFT) + pos) & SAT;
    }
    bool contains(uint64_t hv) const {return count(hv) != 0;}
    bool may_contain(uint64_t hv) const {return contains(hv);}

    void add(uint64_t hv) {
        for(;;) {
            if(HEDLEY_UNLIKELY(used_ >= max_load_ * double(nslots_))) grow();
            hv &= HVMASK;
            const uint64_t b1 = blk1(hv);
            uint64_t pos;
            // presence always covers BOTH candidate blocks, whatever the
            // placement shortcut: a key may live in the secondary from an
            // earlier insert that found the primary fuller
            uint64_t fb;
            if(find_slot(hv, fb, pos)) {
                const uint64_t i = (fb << BLK_SHIFT) + pos;
                const uint64_t c = get_cnt(i);
                if(c < SAT) put(i, hv, c + 1);
                return;
            }
            if(live_[b1] < (BLK * 3) / 4) { // VQF S6.2 shortcut: primary only
                const unsigned f = block_free(b1);
                put((b1 << BLK_SHIFT) + f, hv, 1);
                bump_live(b1, 1); ++used_;
                return;
            }
            const uint64_t b2 = blk2(hv);
            const unsigned l1 = live_[b1], l2 = live_[b2];
            uint64_t target = ~uint64_t(0);
            if(l2 < BLK && (l1 >= BLK || l2 < l1)) target = b2;
            else if(l1 < BLK) target = b1;
            if(HEDLEY_UNLIKELY(target == ~uint64_t(0))) { grow(); continue; } // both full: grow + retry
            const unsigned f = block_free(target);
            put((target << BLK_SHIFT) + f, hv, 1);
            bump_live(target, 1); ++used_;
            return;
        }
    }

    // ---- concurrent insert (item 6). VQF S6.3 pattern: the block lock
    // lives in the live word's top bit; the shortcut path takes ONE
    // block lock, the two-choice path takes both in increasing order;
    // grow runs under an exclusive table mutex. Readers (count/contains,
    // range, iterators) stay lock-free: a slot write is a single word
    // store, so a racing reader sees either absence or a complete slot.
    // Do not mix add() and add_mt() concurrently (sequential is fine).
    void add_mt(uint64_t hv) {
        for(;;) {
            bool need_grow = false;
            {
                std::shared_lock<std::shared_mutex> g(grow_mu_);
                if(HEDLEY_UNLIKELY(used_.load() >= max_load_ * double(nslots_))) need_grow = true;
                else {
                    hv &= HVMASK;
                    const uint64_t b1 = blk1(hv);
                    const uint64_t b2 = blk2(hv);
                    uint64_t pos;
                    lock_block(b1);
                    if(block_find(b1, hv, pos)) { // duplicate under lock
                        const uint64_t i = (b1 << BLK_SHIFT) + pos;
                        const uint64_t c = get_cnt(i);
                        if(c < SAT) put(i, hv, c + 1);
                        unlock_block(b1);
                        return;
                    }
                    if(live_[b1] < (BLK * 3) / 4) { // shortcut: one lock
                        // Presence must also cover the secondary block
                        // before placing into the primary: a key placed in
                        // b2 by an earlier two-choice insert must not be
                        // duplicated when b1 later drops below the
                        // threshold. Scanning b2 without its lock is
                        // race-free while b1's lock is held: every
                        // placement path for this key takes b1's lock, so
                        // no concurrent writer can be inserting this key.
                        if(b2 != b1 && block_find(b2, hv, pos)) { // secondary copy
                            const uint64_t i = (b2 << BLK_SHIFT) + pos;
                            const uint64_t c = get_cnt(i);
                            if(c < SAT) put(i, hv, c + 1);
                            unlock_block(b1);
                            return;
                        }
                        const unsigned f = block_free(b1);
                        put((b1 << BLK_SHIFT) + f, hv, 1);
                        bump_live(b1, 1);
                        ++used_;
                        unlock_block(b1);
                        return;
                    }
                    unlock_block(b1);
                    const uint64_t blo = std::min(b1, b2), bhi = std::max(b1, b2);
                    lock_block(blo);
                    if(bhi != blo) lock_block(bhi);
                    bool placed = false;
                    if(block_find(b1, hv, pos)) { // raced insert landed meanwhile
                        const uint64_t i = (b1 << BLK_SHIFT) + pos;
                        const uint64_t c = get_cnt(i);
                        if(c < SAT) put(i, hv, c + 1);
                        placed = true;
                    } else if(bhi != blo && block_find(b2, hv, pos)) { // raced secondary copy
                        const uint64_t i = (b2 << BLK_SHIFT) + pos;
                        const uint64_t c = get_cnt(i);
                        if(c < SAT) put(i, hv, c + 1);
                        placed = true;
                    } else {
                        const unsigned l1 = live_[b1], l2 = live_[b2];
                        uint64_t target = ~uint64_t(0);
                        if(l2 < BLK && (l1 >= BLK || l2 < l1)) target = b2;
                        else if(l1 < BLK) target = b1;
                        if(target != ~uint64_t(0)) {
                            const unsigned f = block_free(target);
                            put((target << BLK_SHIFT) + f, hv, 1);
                            bump_live(target, 1);
                            ++used_;
                            placed = true;
                        }
                    }
                    if(bhi != blo) unlock_block(bhi);
                    unlock_block(blo);
                    if(placed) return;
                    need_grow = true; // both full
                }
            }
            if(need_grow) {
                std::unique_lock<std::shared_mutex> g(grow_mu_);
                if(used_.load() >= max_load_ * double(nslots_)) grow();
                // else another thread grew already; retry placement
            }
        }
    }

    // decrement one occurrence; erases the key when the counter hits zero
    bool dec(uint64_t hv) {
        hv &= HVMASK;
        uint64_t b, pos;
        if(find_slot(hv, b, pos)) {
            const uint64_t i = (b << BLK_SHIFT) + pos;
            const uint64_t c = get_cnt(i);
            if(c > 1) { put(i, hv, c - 1); return true; }
            put(i, 0, 0); bump_live(b, -1); --used_;
            return true;
        }
        return false;
    }
    bool del(uint64_t hv) {return dec(hv);}
    // STL-style surface (mirrors cqfbase_t)
    void insert(uint64_t hv) {add(hv);}
    void insert(const std::pair<uint64_t, uint64_t> &kv) {add_n(kv.first, kv.second);}
    void emplace(uint64_t hv, uint64_t v = 1) {add_n(hv, v);}
    void add_n(uint64_t hv, uint64_t v) {while(v--) add(hv);}
    uint64_t at(uint64_t hv) const {return count(hv);}
    bool erase(uint64_t hv) {
        hv &= HVMASK;
        uint64_t b, pos;
        if(find_slot(hv, b, pos)) {
            put((b << BLK_SHIFT) + pos, 0, 0); bump_live(b, -1); --used_;
            return true;
        }
        return false;
    }
    class counted_ref {
        bcqf_t *f_;
        uint64_t hv_;
    public:
        counted_ref(bcqf_t *f, uint64_t hv): f_(f), hv_(hv) {}
        operator uint64_t() const {return f_->count(hv_);}
        counted_ref &operator=(uint64_t v) {
            uint64_t cur = f_->count(hv_);
            if(v > cur) {f_->add_n(hv_, v - cur);}
            else {while(v < cur--) f_->dec(hv_);}
            return *this;
        }
        counted_ref &operator+=(uint64_t v) {f_->add_n(hv_, v); return *this;}
        counted_ref &operator-=(uint64_t v) {while(v--) f_->dec(hv_); return *this;}
        counted_ref &operator++() {f_->add(hv_); return *this;}
        counted_ref &operator--() {f_->dec(hv_); return *this;}
    };
    counted_ref operator[](uint64_t hv) {return counted_ref(this, hv);}

    template<typename It>
    void insert_sorted(It first, It last) {for(; first != last; ++first) add(*first);}
    template<typename It>
    void insert_sorted_counts(It first, It last) {
        for(; first != last; ++first) add_n(first->first, first->second);
    }
    template<typename It>
    void insert_buffered(It first, It last, size_t threshold = 8192) {
        std::unordered_map<uint64_t, uint64_t> buf;
        buf.reserve(threshold * 2);
        auto flush = [&]() {
            if(buf.empty()) return;
            for(auto &kv: buf) add_n(kv.first, kv.second);
            buf.clear();
        };
        for(; first != last; ++first) {
            ++buf[reduce(hf_(*first))];
            if(buf.size() >= threshold) flush();
        }
        flush();
    }

    void clear() {
        if(!SOA) std::fill(slots_.begin(), slots_.end(), 0);
        else {
            std::fill(hashes_.begin(), hashes_.end(), 0);
            std::fill(cnts_.begin(), cnts_.end(), 0);
        }
        std::fill(live_.begin(), live_.end(), 0);
        for(auto &r: rollup_) r.store(0, std::memory_order_relaxed);
        for(auto &l: locks_) l.store(0, std::memory_order_relaxed);
        used_ = 0;
    }
    void reset() {clear();}
    void free() {
        auto tmp(std::move(slots_));
        auto tmph(std::move(hashes_));
        auto tmpc(std::move(cnts_));
        auto tmp2(std::move(live_));
        auto tmp3(std::move(rollup_));
        auto tmp4(std::move(locks_));
        q_ = 0; nslots_ = nblocks_ = used_ = 0;
    }

    // 2x grow: q+1, reinsert all live (hash, count) pairs
    void grow() {
        if(q_ + 1 > 40 || q_ + 1 > KEY_BITS + BLK_SHIFT)
            throw std::overflow_error("[bcqf] cannot grow: block index would exceed the hash width");
        std::vector<std::pair<uint64_t, uint64_t>> els;
        els.reserve(used_);
        for(uint64_t i = 0; i < nslots_; ++i)
            if(slot_live(i)) els.push_back({get_hv(i), get_cnt(i)});
        init(q_ + 1);
        for(const auto &kv: els) add_n(kv.first, kv.second);
    }

    bcqf_t &operator+=(const bcqf_t &o) {
        if(q_ != o.q_)
            throw std::invalid_argument(std::string("[bcqf] merge requires matching q (") + std::to_string(q_) + " vs " + std::to_string(o.q_) + ")");
        if(o.used_ == 0) return *this;
        if(used_ == 0) {*this = o; return *this;}
        const bcqf_t *small = &o, *large = this;
        if(o.used_ > used_) std::swap(small, large);
        bcqf_t dst(*large);
        for(uint64_t i = 0; i < small->nslots_; ++i)
            if(small->slot_live(i)) dst.add_n(small->get_hv(i), small->get_cnt(i));
        *this = std::move(dst);
        return *this;
    }
    bcqf_t operator+(const bcqf_t &o) const {auto r = *this; r += o; return r;}

    // ---- iteration over (key, count) pairs, block order: slot index
    // ascending within each block, blocks ascending. Forward-only;
    // empty blocks skip via live_, empty superblocks via the rollup.
    class iterator {
        const bcqf_t *f_;
        uint64_t i_;
        void advance() {
            while(f_ && i_ < f_->nslots_) {
                const uint64_t b = i_ >> BLK_SHIFT;
                if((i_ & (BLK - 1)) == 0) { // block boundary: bulk skip
                    if(f_->live_[b] == 0) {
                        if(f_->rollup_[b / SUPER] == 0) {
                            const uint64_t sb = (b / SUPER) * SUPER;
                            const uint64_t bnext = std::min(sb + SUPER, f_->nblocks_);
                            i_ = bnext << BLK_SHIFT; // whole superblock empty
                            continue;
                        }
                        i_ = (b + 1) << BLK_SHIFT; // block empty
                        continue;
                    }
                }
                if(f_->slot_live(i_)) return;
                ++i_;
            }
        }
    public:
        iterator(): f_(nullptr), i_(0) {}
        iterator(const bcqf_t *f, uint64_t i): f_(f), i_(i) {advance();}
        iterator &operator++() {++i_; advance(); return *this;}
        bool operator!=(const iterator &o) const {return f_ != o.f_ || i_ != o.i_;}
        bool operator==(const iterator &o) const {return !(*this != o);}
        std::pair<uint64_t, uint64_t> operator*() const {
            return {f_->get_hv(i_), f_->get_cnt(i_)};
        }
    };
    iterator begin() const {return iterator(this, 0);}
    iterator end() const {return iterator(this, nslots_);}

    // ---- hash-windowed ranges (item 4). Block order is documented at
    // the iterator above: slots ascending within blocks, blocks
    // ascending. NOTE exactness: a windowed key may reside in its
    // SECONDARY block (blk2 is a multiply-mix, not monotone), so exact
    // windowed enumeration scans the whole table in block order with an
    // hv test, bulk-skipping only empty superblocks via the rollup --
    // it does NOT bound the scan to the window's primary-block range.
    // Union/intersection are therefore probe-based (block order is not
    // hv-sorted), O(|window A| + |window B|).
    // Single-pass, forward-only, allocation-free.
    template<typename F>
    void range(uint64_t lo, uint64_t hi, F &&fn) const {
        lo &= HVMASK;
        if(hi > HVMASK + 1) hi = HVMASK + 1; // hi is one-past; 2^KEY_BITS means "to the end"
        if(lo >= hi) return;
        for(uint64_t b = 0; b < nblocks_;) {
            if(rollup_[b / SUPER].load(std::memory_order_relaxed) == 0) { // whole superblock empty
                const uint64_t sb = (b / SUPER) * SUPER;
                b = std::min(sb + SUPER, nblocks_);
                continue;
            }
            if(live_[b] == 0) { ++b; continue; }
            const uint64_t base = b << BLK_SHIFT;
            for(unsigned i = 0; i < BLK; ++i) {
                const uint64_t idx = base + i;
                if(!slot_live(idx)) continue;
                const uint64_t hv = get_hv(idx);
                if(hv >= lo && hv < hi) fn(hv, get_cnt(idx));
            }
            ++b;
        }
    }
    // union over the window: every key in either filter, (hv, ca, cb)
    // with 0 for the absent side
    template<typename F>
    void union_range(const bcqf_t &o, uint64_t lo, uint64_t hi, F &&fn) const {
        range(lo, hi, [&](uint64_t hv, uint64_t ca) {fn(hv, ca, o.count(hv));});
        o.range(lo, hi, [&](uint64_t hv, uint64_t cb) {if(!count(hv)) fn(hv, 0, cb);});
    }
    // intersection over the window: keys live in BOTH, (hv, ca, cb)
    template<typename F>
    void intersection_range(const bcqf_t &o, uint64_t lo, uint64_t hi, F &&fn) const {
        range(lo, hi, [&](uint64_t hv, uint64_t ca) {
            const uint64_t cb = o.count(hv);
            if(cb) fn(hv, ca, cb);
        });
    }
    // ---- freeze into a flat sorted (hash, count)
    // array. Block-ordered enumeration is already MSD-sorted by the top
    // (q - BLK_SHIFT) bits for primary-routed keys (blk1 = top bits), so
    // concatenating per-block sorted runs yields sorted runs with free
    // bucket boundaries; blk2-routed strays (possible only past the
    // primary-only threshold) are collected, sorted, and merged in.
    // Under primary-only placement there are ZERO strays. The output
    // drops all block metadata: the array is the structure.
    std::vector<std::pair<uint64_t, uint64_t>> freeze() const {
        std::vector<std::pair<uint64_t, uint64_t>> out;
        out.reserve(used_.load());
        std::vector<std::pair<uint64_t, uint64_t>> stray;
        std::pair<uint64_t, uint64_t> buf[BLK];
        for(uint64_t b = 0; b < nblocks_; ++b) {
            unsigned n = 0;
            const uint64_t base = b << BLK_SHIFT;
            for(unsigned i = 0; i < BLK; ++i) {
                const uint64_t idx = base + i;
                if(!slot_live(idx)) continue;
                const uint64_t hv = get_hv(idx);
                if(blk1(hv) != b) { // blk2-routed stray: merged in after the walk
                    stray.push_back({hv, get_cnt(idx)});
                    continue;
                }
                buf[n++] = {hv, get_cnt(idx)};
            }
            std::sort(buf, buf + n);
            out.insert(out.end(), buf, buf + n);
        }
        if(!stray.empty()) {
            std::sort(stray.begin(), stray.end());
            std::vector<std::pair<uint64_t, uint64_t>> merged;
            merged.reserve(out.size());
            std::merge(out.begin(), out.end(), stray.begin(), stray.end(), std::back_inserter(merged));
            out.swap(merged);
        }
        return out;
    }
    uint64_t freeze_stray_count() const { // diagnostics: blk2-routed entries
        uint64_t n = 0;
        for(uint64_t b = 0; b < nblocks_; ++b)
            for(unsigned i = 0; i < BLK; ++i) {
                const uint64_t idx = (b << BLK_SHIFT) + i;
                if(slot_live(idx) && blk1(get_hv(idx)) != b) ++n;
            }
        return n;
    }

    // ---- one sequential compaction pass into a compact
    // layout. Enumerates in block order (rollup-skipping empty
    // superblocks), reduces each key into the target's fp space, and
    // accumulates counts -- the overcount semantics of compact fingerprint mode
    // applied in bulk. Post-compact table is a normal instance of the
    // target type: same query code, same fuzz.
    template<size_t FKB, size_t FVB>
    bcqf_t<FKB, FVB, Hash> to_compact() const {
        bcqf_t<FKB, FVB, Hash> dst(q_);
        range(0, HVMASK + 1, [&](uint64_t hv, uint64_t c) {dst.add_n(hv & bcqf_t<FKB, FVB, Hash>::HVMASK, c);});
        return dst;
    }

    uint64_t range_size(uint64_t lo, uint64_t hi) const {
        uint64_t n = 0;
        range(lo, hi, [&](uint64_t, uint64_t) {++n;});
        return n;
    }
    iterator find(uint64_t hv) const {
        hv &= HVMASK;
        uint64_t b, pos;
        if(find_slot(hv, b, pos)) return iterator(this, (b << BLK_SHIFT) + pos);
        return end();
    }

    // ---- serialization (CQF6: logical (hash,cnt) stream; CQF5 files are
    // read by re-inserting their overflow entries) ----
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[3]{MAGIC, q_, uint32_t(VAL_BITS)};
        CW(fp, header, sizeof(header));
        CW(fp, &used_, sizeof(used_));
        std::vector<uint64_t, common::Allocator<uint64_t>> flat(nslots_);
        for(uint64_t i = 0; i < nslots_; ++i) flat[i] = get_hv(i) | (get_cnt(i) << KEY_BITS);
        CW(fp, flat.data(), nslots_ * sizeof(uint64_t)); // logical stream stays u64
        CW(fp, live_.data(), nblocks_ * sizeof(uint16_t));
#undef CW
    }
    void write(const char *path) const {
        gzFile fp(gzopen(path, "wb"));
        if(!fp) throw ZlibError(Z_ERRNO, std::string("Could not open file at '") + path + "' for writing");
        write(fp);
        gzclose(fp);
    }
    void write(const std::string &path) const {write(path.data());}
    void read(gzFile fp) {
#define CR(fp, dst, len) do {if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) throw ZlibError(std::string("[bcqf] Error reading from file"));} while(0)
        uint32_t header[3];
        CR(fp, header, sizeof(header));
        if(header[0] != MAGIC && header[0] != MAGIC5) throw std::runtime_error("[bcqf] bad magic: not a CQF6/CQF5 (block layout) sketch");
        const bool legacy = header[0] == MAGIC5;
        if(header[2] != VAL_BITS)
            throw std::runtime_error(std::string("[bcqf] VAL_BITS mismatch: file has ") + std::to_string(header[2]) + ", this type has " + std::to_string(VAL_BITS));
        if(header[1] < 6 || header[1] > 40 || uint64_t(header[1]) > uint64_t(KEY_BITS) + BLK_SHIFT)
            throw std::runtime_error(std::string("[bcqf] invalid q in file: ") + std::to_string(header[1]));
        if(header[1] != q_) init(header[1]);
        CR(fp, &used_, sizeof(used_));
        std::vector<uint64_t, common::Allocator<uint64_t>> flat(nslots_);
        CR(fp, flat.data(), nslots_ * sizeof(uint64_t));
        for(uint64_t i = 0; i < nslots_; ++i) put(i, flat[i] & HVMASK, flat[i] >> KEY_BITS);
        std::vector<uint16_t, common::Allocator<uint16_t>> rl(nblocks_);
        CR(fp, rl.data(), nblocks_ * sizeof(uint16_t));
        for(uint64_t s = 0; s < rollup_.size(); ++s) rollup_[s].store(0, std::memory_order_relaxed);
        for(uint64_t b = 0; b < nblocks_; ++b) {
            live_[b] = rl[b];
            rollup_[b / SUPER].fetch_add(uint32_t(live_[b]), std::memory_order_relaxed);
        }
        if(legacy) { // CQF5: overflow entries re-insert into the table
            uint64_t osz;
            CR(fp, &osz, sizeof(osz));
            for(uint64_t i = 0; i < osz; ++i) {
                uint64_t k, v;
                CR(fp, &k, sizeof(uint64_t)); CR(fp, &v, sizeof(uint64_t));
                add_n(k & HVMASK, v);
            }
        }
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}

    // ---- invariant checker: throws on violation (active in all builds) ----
    // Block live counts and superblock rollups accurate; every live
    // slot's hash routes to its owning block (b1 or b2); used_ agrees.
    void structure_check() const {
        uint64_t live_total = 0;
        for(uint64_t b = 0; b < nblocks_; ++b) {
            uint64_t cnt = 0;
            for(unsigned i = 0; i < BLK; ++i) {
                const uint64_t idx = (b << BLK_SHIFT) + i;
                const uint64_t c = get_cnt(idx);
                if(c == 0) continue;
                if(c > SAT) throw std::runtime_error("[bcqf] counter above SAT");
                const uint64_t hv = get_hv(idx);
                if(blk1(hv) != b && blk2(hv) != b)
                    throw std::runtime_error(std::string("[bcqf] slot routes outside its block: ") + std::to_string(hv));
                ++cnt;
            }
            if(cnt != live_[b])
                throw std::runtime_error(std::string("[bcqf] live count mismatch in block ") + std::to_string(b));
            if(locks_[(b / SUPER) * LOCK_PAD].load(std::memory_order_relaxed))
                throw std::runtime_error(std::string("[bcqf] lock held in block ") + std::to_string(b));
            live_total += cnt;
        }
        if(used_.load() != live_total)
            throw std::runtime_error("[bcqf] used count mismatch");
        for(uint64_t s = 0; s < rollup_.size(); ++s) {
            uint32_t acc = 0;
            const uint64_t b0 = s * SUPER, b1 = std::min(b0 + SUPER, nblocks_);
            for(uint64_t b = b0; b < b1; ++b) acc += live_[b];
            if(acc != rollup_[s].load(std::memory_order_relaxed))
                throw std::runtime_error(std::string("[bcqf] rollup mismatch in superblock ") + std::to_string(s));
        }
    }
};

// compact fingerprint modes: fp-space instantiations of the
// same engine; callers reduce hv -> fp before add/count. fp collisions
// merge counts (reference-remainder-class overcount, documented in
// compact-mode notes).
using bcqf16_16 = bcqf_t<16, 16>; // fp16 + val16 in u32; fidelity needs n << 2^16
using bcqf24_8 = bcqf_t<24, 8>;   // fp24 + val8 in u32; counts saturate at 255
using bcqf20_12 = bcqf_t<20, 12>;
using bcqf32_8 = bcqf_t<32, 8>;
using bcqf48_8 = bcqf_t<48, 8>;
using bcqf48_16 = bcqf_t<48, 16>;
using cqf20_12 = cqfbase_t<20, 12>;
using cqf20_30 = cqfbase_t<20, 30>;
using cqf32_8 = cqfbase_t<32, 8>;
using cqf64_8 = cqfbase_t<64, 8>;
using cqf64_16 = cqfbase_t<64, 16>;
using cqf64_32 = cqfbase_t<64, 32>;
using cqf64_64 = cqfbase_t<64, 64>;

} // namespace cqf
} // namespace sketch

#endif // SKETCH_CQF_H__
