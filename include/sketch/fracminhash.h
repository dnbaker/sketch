#pragma once
#ifndef SKETCH_FRACMINHASH_H__
#define SKETCH_FRACMINHASH_H__
/*
 * FracMinHash: fractionally-sampled minhash (sourmash-style).
 *
 * Only hashed items with hash value <= max_hash are kept, retaining an
 * expected fraction max_hash / 2^64 of the distinct elements. Estimates:
 *
 *   cardinality    n^ = s * |kept|,            s = 2^64 / max_hash
 *                  Var(n^) = s * n  (Poisson), std err = s / sqrt(|kept|)
 *   containment    C(A in B) = |kept(A) n kept(B)| / |kept(B)|
 *
 * The kept sample is exact set data: merges are set unions, intersections
 * and containments over kept samples are exact, and idempotency/commutativity
 * hold by construction. Sketch memory grows with the kept fraction, so the
 * scaling factor trades memory/accuracy; insert cost is a single hash plus
 * (rarely) a set insert.
 *
 * References:
 *  - Pierce et al., "Large-scale sequence comparisons with sourmash",
 *    F1000Research 2019 (sourmash; FracMinHash).
 *  - Irber et al., "Fast and accurate metagenotyping with sourmash",
 *    bioRxiv 2022 (FracMinHash theory and error bounds).
 */
#include "common.h"
#include "hash.h"

namespace sketch {
inline namespace fmh {

namespace detail {

// Minimal open-addressing (linear probing) set for 64-bit keys.
// UINT64_MAX is reserved as the empty sentinel and is never inserted.
class fmh_kept_set {
    static constexpr uint64_t EMPTY = ~uint64_t(0);
    std::vector<uint64_t, common::Allocator<uint64_t>> slots_;
    size_t size_ = 0;
    uint32_t shift_ = sizeof(uint64_t) * CHAR_BIT - 6; // multiplicative-shift amount: slot = (v*G) >> shift_
    void grow() {
        const auto old(std::move(slots_));
        slots_.assign(old.size() << 1, EMPTY);
        shift_ = sizeof(uint64_t) * CHAR_BIT - ilog2(slots_.size());
        size_ = 0;
        for(const auto v: old) if(v != EMPTY) insert(v);
    }
    size_t slot(uint64_t v) const {
        return size_t(v * 0x9E3779B97F4A7C15ULL) >> shift_;
    }
public:
    fmh_kept_set() {slots_.assign(64, EMPTY);}
    size_t size() const {return size_;}
    size_t capacity() const {return slots_.size();}
    bool insert(uint64_t v) {
        if(2 * (size_ + 1) > slots_.size()) grow();
        const size_t mask = slots_.size() - 1;
        size_t i = slot(v);
        for(;; i = (i + 1) & mask) {
            const auto cur = slots_[i];
            if(cur == v) return false;
            if(cur == EMPTY) {slots_[i] = v; ++size_; return true;}
        }
    }
    bool contains(uint64_t v) const {
        if(slots_.empty()) return false;
        const size_t mask = slots_.size() - 1;
        size_t i = slot(v);
        for(;; i = (i + 1) & mask) {
            const auto cur = slots_[i];
            if(cur == v) return true;
            if(cur == EMPTY) return false;
        }
    }
    template<typename F>
    void for_each(F &&f) const {
        for(const auto v: slots_) if(v != EMPTY) f(v);
    }
    void clear() {
        std::fill(slots_.begin(), slots_.end(), EMPTY);
        size_ = 0;
    }
};

} // namespace detail

template<typename HashStruct=WangHash>
class fracminhashbase_t {
    // Keeps all hashed items <= max_hash_. scale_ = 2^64 / max_hash_ is the
    // cardinality multiplier (kept-fraction = 1 / scale_).
    uint64_t                        max_hash_;
    double                           scale_;
    detail::fmh_kept_set             kept_;
    uint64_t                      min_kept_; // running minimum over kept hashes
    HashStruct                         hf_;
    static constexpr uint32_t MAGIC = 0x464D4831; // "FMH1"
public:
    using final_type = fracminhashbase_t<HashStruct>;
    using HashType = HashStruct;

    explicit fracminhashbase_t(uint64_t max_hash, HashStruct &&hf=HashStruct()):
        max_hash_(max_hash), scale_(std::ldexp(1., 64) / double(max_hash_)),
        min_kept_(~uint64_t(0)), hf_(std::move(hf)) {
        if(max_hash_ == 0)
            throw std::invalid_argument("[fracminhash] max_hash must be nonzero");
    }
    static uint64_t max_hash_for_scale(double scale) {
        const long double v = std::ldexp(1.L, 64) / std::max(static_cast<long double>(scale), 1.L);
        return v >= std::ldexp(1.L, 64) ? ~uint64_t(0): uint64_t(v);
    }
    explicit fracminhashbase_t(double scale, HashStruct &&hf=HashStruct()):
        fracminhashbase_t(max_hash_for_scale(scale), std::move(hf)) {
        if(scale < 1.) throw std::invalid_argument("[fracminhash] scale must be >= 1");
    }
    fracminhashbase_t() = delete;
    // Non-template; see exaloglogbase_t for why.
    fracminhashbase_t(const std::string &path): fracminhashbase_t(uint64_t(~uint64_t(0) >> 8)) {
        read(path);
    }
    fracminhashbase_t(gzFile fp): fracminhashbase_t(uint64_t(~uint64_t(0) >> 8)) {
        read(fp);
    }

    uint64_t max_hash() const {return max_hash_;}
    double scale() const {return scale_;}
    uint64_t size() const {return kept_.size();}   // number of kept samples
    bool empty() const {return kept_.size() == 0;}
    uint64_t min_hash() const {return min_kept_;}
    uint64_t total_added() const {return uint64_t(kept_.size() * scale_ + .5);}
    // Visit every kept hash (e.g. for extraction or custom serialization).
    template<typename F>
    void for_each_kept(F &&f) const {kept_.for_each(std::forward<F>(f));}
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this), kept_.capacity() * sizeof(uint64_t));
    }

    INLINE uint64_t hash(uint64_t val) const {return hf_(val);}
    INLINE void addh(uint64_t element) noexcept {add(hf_(element));}

    // Threshold filter: keep hashes <= max_hash_. (hash == UINT64_MAX is
    // skipped; it is the internal empty sentinel and lands above any
    // max_hash_ < 2^64 - 1 anyway when max_hash_ is not maximal.)
    INLINE void add(uint64_t hashval) noexcept {
        if(hashval > max_hash_ || hashval == ~uint64_t(0)) return;
        if(hashval < min_kept_) min_kept_ = hashval;
        kept_.insert(hashval);
    }

    // Batch threshold filter with SIMD prefiltering: compares a vector of
    // hashes against max_hash_ and inserts only the surviving lanes.
    void add(const uint64_t *vals, size_t n) noexcept {
        size_t i = 0;
#if HAS_AVX_512 && __AVX512F__
        const __m512i lim(_mm512_set1_epi64(max_hash_));
        for(; i + 8 <= n; i += 8) {
            const __m512i hv(_mm512_loadu_si512((const void *)(vals + i)));
            // uint32_t <= : not (hv > lim) == not (lim < hv)
            const __mmask8 gt(_mm512_cmpgt_epu64_mask(hv, lim));
            uint32_t bits(uint32_t(gt) ^ 0xFFu);
            while(bits) {
                const uint32_t off = ctz(bits);
                add(vals[i + off]);
                bits &= bits - 1;
            }
        }
#elif __AVX2__
        const __m256i lim(_mm256_set1_epi64x(max_hash_));
        const __m256i flip(_mm256_set1_epi64x(int64_t(0x8000000000000000ULL)));
        for(; i + 4 <= n; i += 4) {
            const __m256i hv(_mm256_loadu_si256((const __m256i *)(vals + i)));
            // biased uint32_t compare: a <= b uint32_t  <=>  a^s <= b^s signed
            const __m256i hb(_mm256_xor_si256(hv, flip)), lb(_mm256_xor_si256(lim, flip));
            const __m256i gt(_mm256_cmpgt_epi64(hb, lb));
            uint32_t bits(uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(gt))) ^ 0xFu);
            while(bits) {
                const uint32_t off = ctz(bits);
                add(vals[i + off]);
                bits &= bits - 1;
            }
        }
#elif __SSE2__
        const __m128i lim(_mm_set1_epi64x(max_hash_));
        const __m128i flip(_mm_set1_epi64x(int64_t(0x8000000000000000ULL)));
        for(; i + 2 <= n; i += 2) {
            const __m128i hv(_mm_loadu_si128((const __m128i *)(vals + i)));
            const __m128i hb(_mm_xor_si128(hv, flip)), lb(_mm_xor_si128(lim, flip));
            const __m128i gt(_mm_cmpgt_epi64(hb, lb));
            uint32_t bits(uint32_t(_mm_movemask_pd(_mm_castsi128_pd(gt))) ^ 0x3u);
            while(bits) {
                const uint32_t off = ctz(bits);
                add(vals[i + off]);
                bits &= bits - 1;
            }
        }
#endif
        for(; i < n; ++i) add(vals[i]);
    }
    template<typename It>
    void add(It b, It e) noexcept {add(&*b, std::distance(b, e));}

    // n^ = scale * |kept|; Var = scale^2 * n/scale = scale * n.
    double report() const {return scale_ * double(kept_.size());}
    double estimate() const {return report();}
    double creport() const {return report();}
    double cardinality_estimate() const {return report();}
    // Standard error of the cardinality estimate (Poisson): scale/sqrt(|kept|).
    double est_err() const {
        return kept_.size() ? scale_ / std::sqrt(double(kept_.size())): std::numeric_limits<double>::infinity();
    }

    // Exact count of kept samples of o also present here.
    uint64_t intersection_count(const fracminhashbase_t &o) const {
        uint64_t inter = 0;
        if(kept_.size() <= o.kept_.size())
            kept_.for_each([&](uint64_t v) {inter += o.kept_.contains(v);});
        else
            o.kept_.for_each([&](uint64_t v) {inter += kept_.contains(v);});
        return inter;
    }
    // Estimated |this ∩ o| / |o|: the fraction of o's elements contained in
    // this set (sourmash containment).
    double containment_index(const fracminhashbase_t &o) const {
        if(o.kept_.size() == 0) throw std::runtime_error("[fracminhash] containment against an empty sketch");
        return double(intersection_count(o)) / double(o.kept_.size());
    }
    // Jaccard over the kept samples (slight ratio bias O(1/kept)); see also
    // jaccard_index_by_card for the inclusion-exclusion variant.
    double jaccard_index(const fracminhashbase_t &o) const {
        if(kept_.size() == 0 && o.kept_.size() == 0) return 1.;
        const double inter = double(intersection_count(o));
        const double uni = double(kept_.size() + o.kept_.size()) - inter;
        return uni ? inter / uni: 0.;
    }
    double jaccard_index_by_card(const fracminhashbase_t &o) const {
        const double a = report(), b = o.report();
        if(a == 0. && b == 0.) return 1.;
        const double u = union_size(o);
        return u > 0. ? std::max(0., a + b - u) / u: 0.;
    }
    double union_size(const fracminhashbase_t &o) const {
        const auto tmp = *this + o;
        return tmp.report();
    }

    // Union merge.
    fracminhashbase_t &add(const fracminhashbase_t &other) {
        if(other.max_hash_ != max_hash_)
            throw std::invalid_argument("[fracminhash] cannot merge sketches with different max_hash");
        other.kept_.for_each([&](uint64_t v) {this->add(v);});
        return *this;
    }
    fracminhashbase_t &operator+=(const fracminhashbase_t &other) {return add(other);}
    fracminhashbase_t operator+(const fracminhashbase_t &other) const {
        fracminhashbase_t ret(*this);
        ret.add(other);
        return ret;
    }
    bool operator==(const fracminhashbase_t &o) const {
        if(max_hash_ != o.max_hash_ || kept_.size() != o.kept_.size()) return false;
        bool eq = true;
        kept_.for_each([&](uint64_t v) {eq = eq && o.kept_.contains(v);});
        return eq;
    }
    bool operator!=(const fracminhashbase_t &o) const {return !(*this == o);}

    void clear() noexcept {
        kept_.clear();
        min_kept_ = ~uint64_t(0);
    }
    std::string desc_string() const {
        return std::string("FracMinHash, max_hash: ") + std::to_string(max_hash_) + ", scale: " + std::to_string(scale_)
             + ", kept: " + std::to_string(kept_.size());
    }
    std::string to_string() const {return desc_string() + ", estimate: " + std::to_string(report());}

    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[2]{MAGIC, 0};
        CW(fp, header, sizeof(header));
        CW(fp, &max_hash_, sizeof(max_hash_));
        CW(fp, &min_kept_, sizeof(min_kept_));
        const uint64_t n = kept_.size();
        CW(fp, const_cast<uint64_t *>(&n), sizeof(n));
        kept_.for_each([&](uint64_t v) {CW(fp, &v, sizeof(v));});
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
#define CR(fp, dst, len) \
    do {\
        if(static_cast<uint64_t>(gzread(fp, dst, len)) != len) {\
            throw ZlibError(std::string("[E:") + __FILE__ + ':' + std::to_string(__LINE__) + ':' + __PRETTY_FUNCTION__ + "] Error reading from file");\
        }\
    } while(0)
        uint32_t header[2];
        CR(fp, header, sizeof(header));
        if(header[0] != MAGIC) throw std::runtime_error("[fracminhash] bad magic: not a FracMinHash sketch");
        uint64_t mh, mn, n;
        CR(fp, &mh, sizeof(mh));
        CR(fp, &mn, sizeof(mn));
        CR(fp, &n, sizeof(n));
        max_hash_ = mh;
        scale_ = std::ldexp(1., 64) / double(max_hash_);
        clear();
        for(uint64_t i = 0; i < n; ++i) {
            uint64_t v;
            CR(fp, &v, sizeof(v));
            add(v);
        }
        min_kept_ = mn;
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at '") + path + "' for reading");
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {read(path.data());}
};
using fracminhash_t = fracminhashbase_t<>;

} // namespace fmh
} // namespace sketch

#endif // SKETCH_FRACMINHASH_H__
