#pragma once
#ifndef SKETCH_DARTMINHASH_H__
#define SKETCH_DARTMINHASH_H__
/*
 * DartMinHash: dart-throwing minhash sketch (discrete-set specialization of
 * T. Christiani, "DartMinHash: Fast Sketching for Weighted Sets",
 * arXiv:2005.11547, 2020).
 *
 * Each element throws one dart into each of k independent "sequences"
 * (buckets); the dart for bucket j is (rank, id) with
 * rank = W_j(element) in [0, 2^64). Register j keeps the dart of smallest
 * rank thrown so far. Sketches of x and y agree in bucket j exactly when
 * the same element has minimal rank in both, so
 *
 *     Pr[register j matches] = J(x, y)                  (paper eq. (1))
 *     k * J^ ~ Binomial(k, J)   over independent buckets
 *
 * giving an unbiased Jaccard estimator at any sketch size k -- the point
 * of the sketch being that k can be small (64..4096) and estimation is
 * still unbiased, with std sqrt(J(1-J)/k). Cardinality uses the moment
 * estimator n^ = (m - 1) / sum(r_j / 2^64) over the m filled registers
 * (Cohen's size-estimation framework), since E[r_j] = 1/(n+1).
 *
 * Register updates are vectorized: k/LANES independent WangHash lanes per
 * input element (WangHash's shift/add/xor structure maps directly onto
 * 64-bit-lane SIMD), with a biased-sign uint32_t minimum and id blend.
 */
#include "common.h"
#include "hash.h"

namespace sketch {
inline namespace dmh {

namespace detail {

#if HAS_AVX_512 && __AVX512F__
#define DMH_SIMD 1
static constexpr uint32_t DMH_LANES = 8;
static INLINE __m512i dmh_whash(__m512i k) {
    k = _mm512_add_epi64(_mm512_xor_si512(k, _mm512_set1_epi64(-1)), _mm512_slli_epi64(k, 21));
    k = _mm512_xor_si512(k, _mm512_srli_epi64(k, 24));
    k = _mm512_add_epi64(_mm512_add_epi64(k, _mm512_slli_epi64(k, 3)), _mm512_slli_epi64(k, 8));
    k = _mm512_xor_si512(k, _mm512_srli_epi64(k, 14));
    k = _mm512_add_epi64(_mm512_add_epi64(k, _mm512_slli_epi64(k, 2)), _mm512_slli_epi64(k, 4));
    k = _mm512_xor_si512(k, _mm512_srli_epi64(k, 28));
    k = _mm512_add_epi64(k, _mm512_slli_epi64(k, 31));
    return k;
}
static INLINE __mmask8 dmh_lt(__m512i a, __m512i b) { // a < b, uint32_t 64-bit lanes
    return _mm512_cmpgt_epu64_mask(_mm512_xor_si512(a, _mm512_set1_epi64(int64_t(0x8000000000000000ULL))),
                                   _mm512_xor_si512(b, _mm512_set1_epi64(int64_t(0x8000000000000000ULL))));
}
static INLINE __m512i dmh_minu(__m512i a, __m512i b) {
    return _mm512_mask_blend_epi64(dmh_lt(a, b), b, a);
}
static INLINE void dmh_regmin(uint64_t *rank, uint64_t *id, __m512i r2, __m512i id2) {
    const __m512i r1(_mm512_loadu_si512((const void *)rank));
    const __m512i i1(_mm512_loadu_si512((const void *)id));
    const __mmask8 lt(dmh_lt(r2, r1));
    const __mmask8 eq(_mm512_cmpeq_epu64_mask(r1, r2));
    __m512i rk(_mm512_mask_blend_epi64(lt, r1, r2));
    __m512i ik(_mm512_mask_blend_epi64(lt, i1, id2));
    ik = _mm512_mask_blend_epi64(eq, ik, dmh_minu(i1, id2)); // deterministic tie-break
    _mm512_storeu_si512((void *)rank, rk);
    _mm512_storeu_si512((void *)id, ik);
}
#elif __AVX2__
#define DMH_SIMD 1
static constexpr uint32_t DMH_LANES = 4;
static INLINE __m256i dmh_whash(__m256i k) {
    k = _mm256_add_epi64(_mm256_xor_si256(k, _mm256_set1_epi64x(-1)), _mm256_slli_epi64(k, 21));
    k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 24));
    k = _mm256_add_epi64(_mm256_add_epi64(k, _mm256_slli_epi64(k, 3)), _mm256_slli_epi64(k, 8));
    k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 14));
    k = _mm256_add_epi64(_mm256_add_epi64(k, _mm256_slli_epi64(k, 2)), _mm256_slli_epi64(k, 4));
    k = _mm256_xor_si256(k, _mm256_srli_epi64(k, 28));
    k = _mm256_add_epi64(k, _mm256_slli_epi64(k, 31));
    return k;
}
static INLINE __m256i dmh_bias(__m256i x) {return _mm256_xor_si256(x, _mm256_set1_epi64x(int64_t(0x8000000000000000ULL)));}
static INLINE __m256i dmh_lt(__m256i a, __m256i b) { // a < b uint32_t, per lane: all-ones where true
    return _mm256_cmpgt_epi64(dmh_bias(b), dmh_bias(a));
}
static INLINE __m256i dmh_minu(__m256i a, __m256i b) {
    return _mm256_blendv_epi8(b, a, dmh_lt(a, b));
}
static INLINE void dmh_regmin(uint64_t *rank, uint64_t *id, __m256i r2, __m256i id2) {
    const __m256i r1(_mm256_loadu_si256((const __m256i *)rank));
    const __m256i i1(_mm256_loadu_si256((const __m256i *)id));
    const __m256i lt(dmh_lt(r2, r1));
    const __m256i eq(_mm256_cmpeq_epi64(r1, r2));
    const __m256i rk(_mm256_blendv_epi8(r1, r2, lt));
    const __m256i ik(_mm256_blendv_epi8(_mm256_blendv_epi8(i1, id2, lt), dmh_minu(i1, id2), eq));
    _mm256_storeu_si256((__m256i *)rank, rk);
    _mm256_storeu_si256((__m256i *)id, ik);
}
#elif __SSE4_1__
#define DMH_SIMD 1
static constexpr uint32_t DMH_LANES = 2;
static INLINE __m128i dmh_whash(__m128i k) {
    k = _mm_add_epi64(_mm_xor_si128(k, _mm_set1_epi64x(-1)), _mm_slli_epi64(k, 21));
    k = _mm_xor_si128(k, _mm_srli_si64(k, 24));
    k = _mm_add_epi64(_mm_add_epi64(k, _mm_slli_epi64(k, 3)), _mm_slli_epi64(k, 8));
    k = _mm_xor_si128(k, _mm_srli_si64(k, 14));
    k = _mm_add_epi64(_mm_add_epi64(k, _mm_slli_epi64(k, 2)), _mm_slli_epi64(k, 4));
    k = _mm_xor_si128(k, _mm_srli_si64(k, 28));
    k = _mm_add_epi64(k, _mm_slli_epi64(k, 31));
    return k;
}
static INLINE __m128i dmh_bias(__m128i x) {return _mm_xor_si128(x, _mm_set1_epi64x(int64_t(0x8000000000000000ULL)));}
static INLINE __m128i dmh_lt(__m128i a, __m128i b) {
    return _mm_cmpgt_epi64(dmh_bias(b), dmh_bias(a));
}
static INLINE __m128i dmh_minu(__m128i a, __m128i b) {
    return _mm_blendv_epi8(b, a, dmh_lt(a, b)); // SSE4.1; __SSE2__ builds get the scalar path below
}
static INLINE void dmh_regmin(uint64_t *rank, uint64_t *id, __m128i r2, __m128i id2) {
    const __m128i r1(_mm_loadu_si128((const __m128i *)rank));
    const __m128i i1(_mm_loadu_si128((const __m128i *)id));
    const __m128i lt(dmh_lt(r2, r1));
    const __m128i eq(_mm_cmpeq_epi64(r1, r2));
    const __m128i rk(_mm_blendv_epi8(r1, r2, lt));
    const __m128i ik(_mm_blendv_epi8(_mm_blendv_epi8(i1, id2, lt), dmh_minu(i1, id2), eq));
    _mm_storeu_si128((__m128i *)rank, rk);
    _mm_storeu_si128((__m128i *)id, ik);
}
#endif

static constexpr uint64_t DMH_GOLDEN = 0x9E3779B97F4A7C15ULL;

} // namespace detail

template<typename HashStruct=WangHash>
class dartminhashbase_t {
    // k registers: per-bucket minimum (rank, id) dart; rank == UINT64_MAX
    // denotes an empty register (a real dart has rank < UINT64_MAX with
    // probability 1 - 2^-64 per throw; ties are broken by smaller id).
    uint32_t                                          k_;
    uint64_t                                       seed_;
    std::vector<uint64_t, common::Allocator<uint64_t>> ranks_; // padded to LANES
    std::vector<uint64_t, common::Allocator<uint64_t>> ids_;
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_; // per-bucket seeds, padded
    HashStruct                                          hf_;
    static constexpr uint32_t MAGIC = 0x444D4831; // "DMH1"
    static constexpr uint64_t EMPTY = ~uint64_t(0);
public:
    using final_type = dartminhashbase_t<HashStruct>;
    using HashType = HashStruct;

    template<typename... Args>
    explicit dartminhashbase_t(uint32_t k, uint64_t seed=137, Args &&... args):
        k_(k), seed_(seed), hf_(std::forward<Args>(args)...) {
        if(k_ < 2) throw std::invalid_argument("[dartminhash] sketch size k must be >= 2");
        // Pad to at least one full SIMD lane group: the vectorized update
        // writes DMH_LANES registers at a time, and a power-of-2 pad alone
        // is smaller than a lane group for tiny k (heap overflow).
#ifdef DMH_SIMD
        const size_t padded(std::max<size_t>(roundup(k_), detail::DMH_LANES));
#else
        const size_t padded(roundup(k_));
#endif
        ranks_.assign(padded, EMPTY);
        ids_.assign(padded, 0);
        seeds_.resize(padded);
        for(size_t j = 0; j < padded; ++j)
            seeds_[j] = seed_ + uint64_t(j) * detail::DMH_GOLDEN;
    }
    dartminhashbase_t() = delete;
    // Non-template; see exaloglogbase_t for why.
    dartminhashbase_t(const std::string &path): dartminhashbase_t(2, 137) {
        read(path);
    }
    dartminhashbase_t(gzFile fp): dartminhashbase_t(2, 137) {
        read(fp);
    }

    uint32_t size() const {return k_;}      // number of registers (buckets)
    uint64_t seed() const {return seed_;}
    bool is_empty() const {
        for(uint32_t j = 0; j < k_; ++j) if(ranks_[j] != EMPTY) return false;
        return true;
    }
    std::pair<size_t, size_t> est_memory_usage() const {
        return std::make_pair(sizeof(*this), (ranks_.size() + ids_.size() + seeds_.size()) * sizeof(uint64_t));
    }
    const std::vector<uint64_t, common::Allocator<uint64_t>> &ranks() const {return ranks_;}
    const std::vector<uint64_t, common::Allocator<uint64_t>> &ids() const {return ids_;}
    auto &mutable_ranks() {return ranks_;}
    auto &mutable_ids() {return ids_;}

    INLINE uint64_t hash(uint64_t val) const {return hf_(val);}
    INLINE void addh(uint64_t element) noexcept {add(hf_(element));}

    // Throw this element's k darts (one per bucket) and keep the register
    // minima. Vectorized over buckets.
    void add(uint64_t hashval) noexcept {
#ifdef DMH_SIMD
        const uint64_t *sp = seeds_.data();
        uint64_t *rp = ranks_.data(), *ip = ids_.data();
#if HAS_AVX_512 && __AVX512F__
        const __m512i hv(_mm512_set1_epi64(hashval));
        for(uint32_t j = 0; j < k_; j += 8, sp += 8, rp += 8, ip += 8)
            detail::dmh_regmin(rp, ip, detail::dmh_whash(_mm512_xor_si512(hv, _mm512_loadu_si512((const void *)sp))), hv);
#elif __AVX2__
        const __m256i hv(_mm256_set1_epi64x(hashval));
        for(uint32_t j = 0; j < k_; j += 4, sp += 4, rp += 4, ip += 4)
            detail::dmh_regmin(rp, ip, detail::dmh_whash(_mm256_xor_si256(hv, _mm256_loadu_si256((const __m256i *)sp))), hv);
#elif __SSE4_1__
        const __m128i hv(_mm_set1_epi64x(hashval));
        for(uint32_t j = 0; j < k_; j += 2, sp += 2, rp += 2, ip += 2)
            detail::dmh_regmin(rp, ip, detail::dmh_whash(_mm_xor_si128(hv, _mm_loadu_si128((const __m128i *)sp))), hv);
#endif
#else
        for(uint32_t j = 0; j < k_; ++j) {
            const uint64_t r = WangHash::hash(hashval ^ seeds_[j]);
            if(r < ranks_[j]) {ranks_[j] = r; ids_[j] = hashval;}
            else if(r == ranks_[j] && hashval < ids_[j]) ids_[j] = hashval;
        }
#endif
    }
    void add(const uint64_t *vals, size_t n) noexcept {
        for(size_t i = 0; i < n; ++i) add(vals[i]);
    }
    template<typename It>
    void add(It b, It e) noexcept {add(&*b, std::distance(b, e));}

    // Merge: per-bucket minimum dart. Commutative and associative (ties
    // broken by smaller id in both orders).
    dartminhashbase_t &add(const dartminhashbase_t &other) {
        if(other.k_ != k_) throw std::invalid_argument("[dartminhash] cannot merge sketches with different k");
#ifdef DMH_SIMD
        const uint64_t *sp = seeds_.data();
        uint64_t *rp = ranks_.data(), *ip = ids_.data();
#if HAS_AVX_512 && __AVX512F__
        for(uint32_t j = 0; j < k_; j += 8, sp += 8, rp += 8, ip += 8)
            detail::dmh_regmin(rp, ip, _mm512_loadu_si512((const void *)(other.ranks_.data() + j)),
                               _mm512_loadu_si512((const void *)(other.ids_.data() + j)));
#elif __AVX2__
        for(uint32_t j = 0; j < k_; j += 4, sp += 4, rp += 4, ip += 4)
            detail::dmh_regmin(rp, ip, _mm256_loadu_si256((const __m256i *)(other.ranks_.data() + j)),
                               _mm256_loadu_si256((const __m256i *)(other.ids_.data() + j)));
#elif __SSE4_1__
        for(uint32_t j = 0; j < k_; j += 2, sp += 2, rp += 2, ip += 2)
            detail::dmh_regmin(rp, ip, _mm_loadu_si128((const __m128i *)(other.ranks_.data() + j)),
                               _mm_loadu_si128((const __m128i *)(other.ids_.data() + j)));
#endif
#else
        for(uint32_t j = 0; j < k_; ++j) {
            const uint64_t r = other.ranks_[j];
            if(r < ranks_[j]) {ranks_[j] = r; ids_[j] = other.ids_[j];}
            else if(r == ranks_[j] && other.ids_[j] < ids_[j]) ids_[j] = other.ids_[j];
        }
#endif
        return *this;
    }
    dartminhashbase_t &operator+=(const dartminhashbase_t &other) {return add(other);}
    dartminhashbase_t operator+(const dartminhashbase_t &other) const {
        dartminhashbase_t ret(*this);
        ret.add(other);
        return ret;
    }
    bool operator==(const dartminhashbase_t &o) const {
        return k_ == o.k_ && std::equal(ranks_.begin(), ranks_.begin() + k_, o.ranks_.begin())
                          && std::equal(ids_.begin(), ids_.begin() + k_, o.ids_.begin());
    }
    bool operator!=(const dartminhashbase_t &o) const {return !(*this == o);}

    void clear() noexcept {
        std::fill(ranks_.begin(), ranks_.end(), EMPTY);
        std::fill(ids_.begin(), ids_.end(), 0);
    }

    // Number of filled registers.
    uint32_t filled() const {
        uint32_t m = 0;
        for(uint32_t j = 0; j < k_; ++j) m += ranks_[j] != EMPTY;
        return m;
    }
    // Moment cardinality estimator: (m - 1) / sum(r_j / 2^64).
    double report() const {
        const uint32_t m = filled();
        if(m < 2) return double(m);
        long double s = 0.;
        for(uint32_t j = 0; j < k_; ++j)
            if(ranks_[j] != EMPTY) s += static_cast<long double>(ranks_[j]) / std::ldexp(1.L, 64);
        return double((m - 1) / s);
    }
    double estimate() const {return report();}
    double creport() const {return report();}
    double cardinality_estimate() const {return report();}

    // Unbiased Jaccard estimator (paper eq. (1)-(2)); buckets that are
    // empty in both sketches carry no dart information and are excluded.
    double jaccard_index(const dartminhashbase_t &o) const {
        if(k_ != o.k_) throw std::invalid_argument("[dartminhash] cannot compare sketches with different k");
        uint32_t matches = 0, considered = 0;
        for(uint32_t j = 0; j < k_; ++j) {
            const bool ea = ranks_[j] == EMPTY, eb = o.ranks_[j] == EMPTY;
            if(ea && eb) continue;
            ++considered;
            matches += !ea && !eb && ids_[j] == o.ids_[j];
        }
        return considered ? double(matches) / double(considered): 1.;
    }
    double union_size(const dartminhashbase_t &o) const {
        auto tmp = *this + o;
        return tmp.report();
    }
    // |this ∩ o| / |o| via Jaccard and the merged-sketch union estimate.
    double containment_index(const dartminhashbase_t &o) const {
        const double b = o.report();
        if(b == 0.) throw std::runtime_error("[dartminhash] containment against an empty sketch");
        const double u = union_size(o), is = std::max(report() + b - u, 0.);
        return is / b;
    }

    std::string desc_string() const {
        return std::string("DartMinHash, k: ") + std::to_string(k_) + ", filled: " + std::to_string(filled());
    }
    std::string to_string() const {return desc_string() + ", estimate: " + std::to_string(report());}

    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t header[2]{MAGIC, k_};
        CW(fp, header, sizeof(header));
        CW(fp, &seed_, sizeof(seed_));
        CW(fp, ranks_.data(), k_ * sizeof(uint64_t));
        CW(fp, ids_.data(), k_ * sizeof(uint64_t));
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
        if(header[0] != MAGIC) throw std::runtime_error("[dartminhash] bad magic: not a DartMinHash sketch");
        if(header[1] != k_)
            throw std::runtime_error("[dartminhash] sketch was written with a different k");
        CR(fp, &seed_, sizeof(seed_));
        if(seed_ != seeds_[0])
            throw std::runtime_error("[dartminhash] sketch was written with a different seed");
        CR(fp, ranks_.data(), k_ * sizeof(uint64_t));
        CR(fp, ids_.data(), k_ * sizeof(uint64_t));
        // re-materialize per-bucket seeds for the stored seed
        for(size_t j = 0; j < seeds_.size(); ++j)
            seeds_[j] = seed_ + uint64_t(j) * detail::DMH_GOLDEN;
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
using dartminhash_t = dartminhashbase_t<>;

} // namespace dmh
} // namespace sketch

#endif // SKETCH_DARTMINHASH_H__
