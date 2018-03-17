#ifndef HLL_H_
#define HLL_H_
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "sseutil.h"
#include "logutil.h"
#include "util.h"
#include "math.h"
#include "unistd.h"
#include "x86intrin.h"
#include "kthread.h"
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif

#define HAS_AVX_512 (_FEATURE_AVX512F | _FEATURE_AVX512ER | _FEATURE_AVX512PF | _FEATURE_AVX512CD)

#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if defined(NDEBUG)
#  if NDEBUG == 0
#    undef NDEBUG
#  endif
#endif



namespace hll {

/*
 * TODO: calculate distance *directly* without copying to another sketch!
 */

using std::uint64_t;
using std::uint32_t;
using std::uint8_t;
using std::size_t;


namespace detail {
    // Miscellaneous requirements.
    static constexpr double LARGE_RANGE_CORRECTION_THRESHOLD = (1ull << 32) / 30.;
    static constexpr double TWO_POW_32 = 1ull << 32;
    static double small_range_correction_threshold(uint64_t m) {return 2.5 * m;}
static inline double calculate_estimate(uint64_t *counts,
                                        bool use_ertl, uint64_t m, std::uint32_t p, double alpha) {
    double sum = counts[0], value;
    unsigned i;
    if(use_ertl) {
        double z = m * detail::gen_tau(static_cast<double>((m-counts[64 - p +1]))/(double)m);
        for(i = 64-p; i; z += counts[i--], z *= 0.5); // Reuse value variable to avoid an additional allocation.
        z += m * detail::gen_sigma(static_cast<double>(counts[0])/static_cast<double>(m));
        return (m/(2.L*std::log(2.L)))*m / z;
    }
    /* else */
    // Small/large range corrections
    // See Flajolet, et al. HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm
    for(i = 1; i < 64 - p; ++i) sum += counts[i] * (1. / (1ull << i)); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
#if !NDEBUG
    {
        double sum2 = 0.;
        for(int i = 0; i < 64; ++i) sum2 += counts[i] * (1. / (1ull << i)); // 64 - p because we can't have more than that many leading 0s. This is just a speed thing.
        assert(sum2 == sum);
    }
#endif
    if((value = (alpha * m * m / sum)) < detail::small_range_correction_threshold(m)) {
        if(counts[0]) {
            std::fprintf(stderr, "[W:%s:%d]Small value correction. Original estimate %lf. New estimate %lf.\n",
                         __PRETTY_FUNCTION__, __LINE__, value, m * std::log((double)m / counts[0]));
            value = m * std::log((double)(m) / counts[0]);
        }
    } else if(value > detail::LARGE_RANGE_CORRECTION_THRESHOLD) {
        // Reuse sum variable to hold correction.
        sum = -std::pow(2.0L, 32) * std::log1p(-std::ldexp(value, -32));
        if(!std::isnan(sum)) value = sum;
        else std::fprintf(stderr, "[W:%s:%d] Large range correction returned nan. Defaulting to regular calculation.\n", __PRETTY_FUNCTION__, __LINE__);
    }
    return value;
}

template<typename CoreType>
struct parsum_data_t {
    std::atomic<uint64_t> *counts_; // Array decayed to pointer.
    const CoreType          &core_;
    const uint64_t              l_;
    const uint64_t             pb_; // Per-batch
};



union SIMDHolder {

public:

#define DEC_MAX(fn) static constexpr decltype(&fn) max_fn = &fn
#if HAS_AVX_512
    using SType = __m512i;
    DEC_MAX(_mm512_max_epu8);
#elif __AVX2__
    using SType = __m256i;
    DEC_MAX(_mm256_max_epu8);
#elif __SSE2__
    using SType = __m128i;
    DEC_MAX(_mm_max_epu8);
#else
#  error("Need at least SSE2")
#endif
#undef DEC_MAX

    static constexpr size_t nels = sizeof(SType) / sizeof(uint8_t);
    using u8arr = uint8_t[nels];
    SType val;
    u8arr vals;
    void inc_counts(uint64_t *arr) const {
        unroller<0, nels> ur;
        ur(*this, arr);
    }
    template<size_t iternum, size_t niter_left> struct unroller {
        void operator()(const SIMDHolder &ref, uint64_t *arr) const {
            ++arr[ref.vals[iternum]];
            unroller<iternum+1, niter_left-1>()(ref, arr);
        }
    };
    template<size_t iternum> struct unroller<iternum, 0> {
        void operator()(const SIMDHolder &ref, uint64_t *arr) const {}
    };
    static_assert(sizeof(SType) == sizeof(u8arr), "both items in the union must have the same size");
};

template<typename CoreType>
void parsum_helper(void *data_, long index, int tid) {
    parsum_data_t<CoreType> &data(*(parsum_data_t<CoreType> *)data_);
    uint64_t local_counts[64]{0};
    SIMDHolder tmp, *p((SIMDHolder *)&data.core_[index * data.pb_]),
                    *pend((SIMDHolder *)&data.core_[std::min(data.l_, (index+1) * data.pb_)]);
    do {
        tmp = *p++;
        tmp.inc_counts(local_counts);
    } while(p < pend);
    for(uint64_t i = 0; i < 64ull; ++i) data.counts_[i] += local_counts[i];
}

static inline uint64_t finalize(uint64_t h) {
    // Murmurhash3 finalizer, for multiplying hash functions for seedhll_t and hllfilter_t.
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

inline std::set<uint64_t> seeds_from_seed(uint64_t seed, size_t size) {
    LOG_DEBUG("Initializing a vector of seeds of size %zu with a seed-seed of %" PRIu64 "\n", size, seed);
    std::mt19937_64 mt(seed);
    std::set<uint64_t> rset;
    while(rset.size() < size) rset.emplace(mt());
    return rset;
}

} // namespace detail

// Thomas Wang hash
// Original site down, available at https://naml.us/blog/tag/thomas-wang
// This is our core 64-bit hash.
// It has a 1-1 mapping from any one 64-bit integer to another
// and can be inverted with irving_inv_hash.
static INLINE uint64_t wang_hash(uint64_t key) noexcept {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

struct WangHash {
    auto operator()(uint64_t key) const {
        return wang_hash(key);
    }
};
struct MurmurFinHash {
    auto operator()(uint64_t key) const {
        return detail::finalize(key);
    }
};

static INLINE uint64_t roundup64(size_t x) noexcept {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return ++x;
}

#define clztbl(x, arg) do {\
    switch(arg) {\
        case 0:                         x += 4; break;\
        case 1:                         x += 3; break;\
        case 2: case 3:                 x += 2; break;\
        case 4: case 5: case 6: case 7: x += 1; break;\
    }} while(0)

constexpr INLINE int clz_manual( uint32_t x )
{
  int n(0);
  if ((x & 0xFFFF0000) == 0) {n  = 16; x <<= 16;}
  if ((x & 0xFF000000) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF0000000) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (32 - 4));
  return n;
}

// Overload
constexpr INLINE int clz_manual( uint64_t x )
{
  int n(0);
  if ((x & 0xFFFFFFFF00000000ull) == 0) {n  = 32; x <<= 32;}
  if ((x & 0xFFFF000000000000ull) == 0) {n += 16; x <<= 16;}
  if ((x & 0xFF00000000000000ull) == 0) {n +=  8; x <<=  8;}
  if ((x & 0xF000000000000000ull) == 0) {n +=  4; x <<=  4;}
  clztbl(n, x >> (64 - 4));
  return n;
}

// clz wrappers. Apparently, __builtin_clzll is undefined for values of 0.
// However, by modifying our code to set a 1-bit at the end of the shifted
// region, we can guarantee that this does not happen for our use case.

#if __GNUC__ || __clang__
constexpr INLINE unsigned clz(unsigned long long x) {
    return __builtin_clzll(x);
}
constexpr INLINE unsigned clz(unsigned long x) {
    return __builtin_clzl(x);
}
constexpr INLINE unsigned clz(unsigned x) {
    return __builtin_clz(x);
}
constexpr INLINE unsigned ffs(unsigned long long x) {
    return __builtin_ffsll(x);
}
constexpr INLINE unsigned ffs(unsigned long x) {
    return __builtin_ffsl(x);
}
constexpr INLINE unsigned ffs(unsigned x) {
    return __builtin_ffs(x);
}
#else
#pragma message("Using manual clz instead of gcc/clang __builtin_*")
#error("Have not created a manual ffs function. Must be compiled with gcc or clang. (Or a compiler supporting it.)")
#define clz(x) clz_manual(x)
// https://en.wikipedia.org/wiki/Find_first_set#CLZ
// Modified for constexpr, added 64-bit overload.
#endif

static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz failed.");
static_assert(clz(0x0000000000000003ull) == 62, "64-bit clz failed.");
static_assert(clz(0x0000013333000003ull) == 23, "64-bit clz failed.");


constexpr double make_alpha(size_t m) {
    switch(m) {
        case 16: return .673;
        case 32: return .697;
        case 64: return .709;
        default: return 0.7213 / (1 + 1.079/m);
    }
}

#if HAS_AVX_512
using Allocator = sse::AlignedAllocator<uint8_t, sse::Alignment::AVX512>;
#elif __AVX2__
using Allocator = sse::AlignedAllocator<uint8_t, sse::Alignment::AVX>;
#elif __SSE2__
using Allocator = sse::AlignedAllocator<uint8_t, sse::Alignment::SSE>;
#else
using Allocator = std::allocator<uint8_t>;
#endif

// TODO: add a compact, 6-bit version
// For now, I think that it's preferable for thread safety,
// considering there's an intrinsic for the atomic load/store, but there would not
// be for bit-packed versions.


template<typename HashStruct=WangHash>
class hllbase_t {
// HyperLogLog implementation.
// To make it general, the actual point of entry is a 64-bit integer hash function.
// Therefore, you have to perform a hash function to convert various types into a suitable query.
// We could also cut our memory requirements by switching to only using 6 bits per element,
// (up to 64 leading zeros), though the gains would be relatively small
// given how memory-efficient this structure is.

// Attributes
protected:
    uint32_t np_;
    std::vector<uint8_t, Allocator> core_;
    double value_;
    uint32_t is_calculated_:1;
    uint32_t      use_ertl_:1;
    uint32_t     nthreads_:30;
public:
    HashStruct            hf_;

    uint64_t m() const {return static_cast<uint64_t>(1) << np_;}
    double alpha()          const {return make_alpha(m());}
    double relative_error() const {return 1.03896 / std::sqrt(m());}
    // Constructor
    explicit hllbase_t(size_t np, bool use_ertl=true, int nthreads=-1):
        np_(np),
        core_(m(), 0),
        value_(0.), is_calculated_(0), use_ertl_(use_ertl),
        nthreads_(nthreads > 0 ? nthreads: 1) {}
    hllbase_t(const char *path) {
        read(path);
    }
    hllbase_t(const std::string &path): hllbase_t(path.data()) {}
    hllbase_t(gzFile fp): hllbase_t() {
        this->read(fp);
    }
    explicit hllbase_t(): hllbase_t(0, true, -1) {}

    // Call sum to recalculate if you have changed contents.
    void sum() {
        using detail::SIMDHolder;
        uint64_t counts[64]{0};
        SIMDHolder tmp, *p((SIMDHolder *)core_.data()), *pend((SIMDHolder *)&*core_.end());
        do {
            tmp = *p++;
            tmp.inc_counts(counts);
        } while(p < pend);
        value_ = detail::calculate_estimate(counts, use_ertl_, m(), np_, alpha());
#if VERIFY_SIMD
        uint64_t counts2[64]{0};
        for(const auto val: core_) ++counts2[val];
        double val2 = detail::calculate_estimate(counts2, use_ertl_, m(), np_, alpha());
        assert(val2 == value_ || !std::fprintf(stderr, "val2: %lf. val: %lf\n", val2, value_));
        {
            bool allmatch = true;
            for(size_t i(0); i < 64; ++i)
                if(counts[i] != counts2[i])
                    allmatch = false, std::fprintf(stderr, "At pos %zu, counts differ (%i, %i)\n", i, (int)counts[i], (int)counts2[i]);
            assert(allmatch);
        }
#endif
        is_calculated_ = 1;
    }

    // Returns cardinality estimate. Sums if not calculated yet.
    double creport() const {
        if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report."
                                                     " Try the report() function.");
        return value_;
    }
    double report() noexcept {
        if(!is_calculated_) sum();
        return creport();
    }

    // Returns error estimate
    double cest_err() const {
        if(!is_calculated_) throw std::runtime_error("Result must be calculated in order to report.");
        return relative_error() * creport();
    }
    double est_err()  noexcept {
        if(!is_calculated_) sum();
        return cest_err();
    }
    // Returns string representation
    std::string to_string() const {
        std::string params(std::string("p:") + std::to_string(np_) + ";");
        return (params + (is_calculated_ ? std::to_string(creport()) + ", +- " + std::to_string(cest_err())
                                         : desc_string()));
    }
    // Descriptive string.
    std::string desc_string() const {
        char buf[256];
        std::sprintf(buf, "Size: %u. nb: %llu. error: %lf. Is calculated: %s. value: %lf\n",
                     np_, static_cast<long long unsigned int>(m()), relative_error(), is_calculated_ ? "true": "false", value_);
        return buf;
    }

    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval >> q()), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
    }

    INLINE void addh(uint64_t element) {
        element = hf_(element);
        add(element);
    }
    template<typename T, typename Hasher=std::hash<T>>
    INLINE void adds(const T element, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(element))>, uint64_t>, "Must return 64-bit hash");
        add(hasher(element));
    }
#ifdef ENABLE_CLHASH
    template<typename Hasher=clhasher>
    INLINE void adds(const char *s, size_t len, const Hasher &hasher) {
        static_assert(std::is_same_v<std::decay_t<decltype(hasher(s, len))>, uint64_t>, "Must return 64-bit hash");
        add(hasher(s, len));
    }
#endif
    void parsum(int nthreads=-1, size_t pb=4096) {
        if(nthreads < 0) nthreads = std::thread::hardware_concurrency();
        std::atomic<uint64_t> acounts[64];
        std::memset(acounts, 0, sizeof acounts);
        detail::parsum_data_t<decltype(core_)> data{acounts, core_, m(), pb};
        const uint64_t nr(core_.size() / pb + (core_.size() % pb != 0));
        kt_for(nthreads, detail::parsum_helper<decltype(core_)>, &data, nr);
        uint64_t counts[64];
        std::memcpy(counts, acounts, sizeof(counts));
        value_ = detail::calculate_estimate(counts, use_ertl_, m(), np_, alpha());
        is_calculated_ = 1;
    }
    // Reset.
    void clear() {
        // Note: this can be accelerated with SIMD.
        std::fill(std::begin(core_), std::end(core_), 0u);
        value_ = is_calculated_ = 0;
    }
    hllbase_t(hllbase_t&&) = default;
    hllbase_t(const hllbase_t &other):
        np_(other.np_), core_(other.core_), value_(other.value_),
        is_calculated_(other.is_calculated_), use_ertl_(other.use_ertl_),
        nthreads_(other.nthreads_) {}
    hllbase_t& operator=(const hllbase_t &other) {
        // Explicitly define to make sure we don't do unnecessary reallocation.
        if(core_.size() != other.core_.size())
            core_.resize(other.core_.size());
        std::memcpy(core_.data(), other.core_.data(), core_.size());
        np_ = other.np_;
        value_ = other.value_;
        is_calculated_ = other.is_calculated_;
        use_ertl_ = other.use_ertl_;
        nthreads_ = other.nthreads_;
        return *this;
    }
    hllbase_t& operator=(hllbase_t&&) = default;

    hllbase_t &operator+=(const hllbase_t &other) {
        if(other.np_ != np_) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        unsigned i;
#if HAS_AVX_512
        __m512i *els(reinterpret_cast<__m512i *>(core_.data()));
        const __m512i *oels(reinterpret_cast<const __m512i *>(other.core_.data()));
        for(i = 0; i < m() >> 6; ++i) els[i] = _mm512_max_epu8(els[i], oels[i]);
        if(m() < 64) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __AVX2__
        __m256i *els(reinterpret_cast<__m256i *>(core_.data()));
        const __m256i *oels(reinterpret_cast<const __m256i *>(other.core_.data()));
        for(i = 0; i < m() >> 5; ++i) els[i] = _mm256_max_epu8(els[i], oels[i]);
        if(m() < 32) for(;i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#elif __SSE2__
        __m128i *els(reinterpret_cast<__m128i *>(core_.data()));
        const __m128i *oels(reinterpret_cast<const __m128i *>(other.core_.data()));
        for(i = 0; i < m() >> 4; ++i) els[i] = _mm_max_epu8(els[i], oels[i]);
        if(m() < 16) for(; i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#else
        for(i = 0; i < m(); ++i) core_[i] = std::max(core_[i], other.core_[i]);
#endif
        not_ready();
        return *this;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup64(new_size);
        LOG_DEBUG("Resizing to %zu, with np = %zu\n", new_size, (std::size_t)std::log2(new_size));
        clear();
        core_.resize(new_size);
        np_ = (std::size_t)std::log2(new_size);
    }
    bool get_use_ertl() const {return use_ertl_;}
    void set_use_ertl(bool val) {use_ertl_ = val;}
    // Getter for is_calculated_
    bool is_ready() const {return is_calculated_;}
    void not_ready() {is_calculated_ = false;}
    void set_is_ready() {is_calculated_ = true;}
    bool may_contain(uint64_t hashval) const {
        // This returns false positives, but never a false negative.
        return core_[hashval >> q()] >= clz(hashval << np_) + 1;
    }

    bool within_bounds(uint64_t actual_size) const {
        return std::abs(actual_size - creport()) < relative_error() * actual_size;
    }

    bool within_bounds(uint64_t actual_size) {
        return std::abs(actual_size - report()) < est_err();
    }
    const auto &core()    const {return core_;}
    const uint8_t *data() const {return core_.data();}

    uint32_t p() const {return np_;}
    uint32_t q() const {return (sizeof(uint64_t) * CHAR_BIT) - np_;}
    void free() {
        decltype(core_) tmp{};
        std::swap(core_, tmp);
    }
    void write(FILE *fp) const {
        write(fileno(fp));
    }
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t bf[3]{is_calculated_, use_ertl_, nthreads_};
        CW(fp, bf, sizeof(bf));
        CW(fp, &np_, sizeof(np_));
        CW(fp, &value_, sizeof(value_));
        CW(fp, core_.data(), core_.size() * sizeof(core_[0]));
#undef CW
    }
    void write(const char *path, bool write_gz=false) const {
        if(write_gz) {
            gzFile fp(gzopen(path, "wb"));
            if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
            write(fp);
            gzclose(fp);
        } else {
            std::FILE *fp(std::fopen(path, "wb"));
            if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
            write(fileno(fp));
            std::fclose(fp);
        }
    }
    void write(const std::string &path, bool write_gz=false) const {write(path.data(), write_gz);}
    void read(gzFile fp) {
#define CR(fp, dst, len) do {if((uint64_t)gzread(fp, dst, len) != len) throw std::runtime_error("Error reading from file.");} while(0)
        uint32_t bf[3];
        CR(fp, bf, sizeof(bf));
        is_calculated_ = bf[0]; use_ertl_ = bf[1]; nthreads_ = bf[2];
        CR(fp, &np_, sizeof(np_));
        CR(fp, &value_, sizeof(value_));
        core_.resize(m());
        CR(fp, core_.data(), core_.size());
#undef CR
    }
    void read(const char *path) {
        gzFile fp(gzopen(path, "rb"));
        if(fp == nullptr) throw std::runtime_error(std::string("Could not open file at ") + path);
        read(fp);
        gzclose(fp);
    }
    void read(const std::string &path) {
        read(path.data());
    }
    void write(int fileno) const {
        uint32_t bf[3]{is_calculated_, use_ertl_, nthreads_};
        ::write(fileno, bf, sizeof(bf));
        ::write(fileno, &np_, sizeof(np_));
        ::write(fileno, &value_, sizeof(value_));
        ::write(fileno, core_.data(), core_.size() * sizeof(core_[0]));
    }
    void read(int fileno) {
        uint32_t bf[3];
        ::read(fileno, bf, sizeof(bf));
        is_calculated_ = bf[0]; use_ertl_ = bf[1]; nthreads_ = bf[2];
        ::read(fileno, &np_, sizeof(np_));
        ::read(fileno, &value_, sizeof(value_));
        core_.resize(m());
        ::read(fileno, core_.data(), core_.size());
    }
    hllbase_t operator+(const hllbase_t &other) {
        if(other.p() != p())
            throw std::runtime_error(std::string("p (") + std::to_string(p()) + " != other.p (" + std::to_string(other.p()));
        hllbase_t ret(*this);
        ret += other;
        return ret;
    }
    double union_size(const hllbase_t &other) const {
        using detail::SIMDHolder;
        assert(m() == other.m());
        using SType = typename SIMDHolder::SType;
        uint64_t counts[64]{0};
        // We can do this because we use an aligned allocator.
        const SType *p1(reinterpret_cast<const SType *>(data())), *p2(reinterpret_cast<const SType *>(other.data()));
        SIMDHolder tmp;
        do {
            tmp.val = SIMDHolder::max_fn(*p1++, *p2++);
            tmp.inc_counts(counts);
        } while(p1 < reinterpret_cast<const SType *>(&(*core().cend())));
        return detail::calculate_estimate(counts, get_use_ertl(), m(), p(), alpha());
    }
    double jaccard_index(hllbase_t &other) {
        if(!is_ready())             sum();
        if(!other.is_ready()) other.sum();
        return jaccard_index(other);
    }
    double jaccard_index(const hllbase_t &h2) const {
        const double us = union_size(h2);
        double is = creport() + h2.creport() - us;
        if(is <= 0) return 0.;
        is /= us;
        return is;
    }
    size_t size() const {return size_t(m());}
};

using hll_t = hllbase_t<>;

// Returns the size of the set intersection
template<typename HllType>
inline double intersection_size(HllType &first, HllType &other) noexcept {
    if(!first.is_ready()) first.sum();
    if(!other.is_ready()) other.sum();
    return intersection_size((const HllType &)first, (const HllType &)other);
}

template<typename HllType>
inline double jaccard_index(HllType &first, HllType &other) noexcept {
    if(!first.is_ready()) first.sum();
    if(!other.is_ready()) other.sum();
    return jaccard_index((const HllType &)first, (const HllType &)other);
}

template<typename HllType>
inline double jaccard_index(const HllType &h1, const HllType &h2) {
    return h1.jaccard_index(h2);
}

// Returns a HyperLogLog union


template<typename HllType>
static inline double union_size(const HllType &h1, const HllType &h2) {
    return h1.union_size(h2);
}

template<typename HllType>
static inline double intersection_size(const HllType &h1, const HllType &h2) {
    return std::max(0., h1.creport() + h2.creport() - union_size(h1, h2));
}

} // namespace hll

#include "hll_dev.h"

#endif // #ifndef HLL_H_
