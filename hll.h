#ifndef HLL_H_
#define HLL_H_
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <cstring>
#include <vector>
#include "logutil.h"
#include "sseutil.h"
#include "util.h"
#include "math.h"
#include "unistd.h"
#include "x86intrin.h"
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

#ifdef HLL_HEADER_ONLY
#  define _STORAGE_ inline
#else
#  define _STORAGE_
#endif


namespace hll {

/*
 * TODO: calculate distance *directly* without copying to another sketch!
 */

using std::uint64_t;
using std::uint32_t;
using std::uint8_t;
using std::size_t;

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


class hll_t {
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
    uint64_t m() const {return static_cast<uint64_t>(1) << np_;}
    double alpha()          const {return make_alpha(m());}
    double relative_error() const {return 1.03896 / std::sqrt(m());}
    // Constructor
    explicit hll_t(size_t np, bool use_ertl=true, int nthreads=-1):
        np_(np),
        core_(m(), 0),
        value_(0.), is_calculated_(0), use_ertl_(use_ertl),
        nthreads_(nthreads > 0 ? nthreads: 1) {}
    hll_t(const char *path) {
        read(path);
    }
    hll_t(const std::string &path): hll_t(path.data()) {}
    explicit hll_t(): hll_t(0, true, -1) {}

    // Call sum to recalculate if you have changed contents.
    _STORAGE_ void sum();
    _STORAGE_ void parsum(int nthreads=-1, size_t per_batch=1<<18);

    // Returns cardinality estimate. Sums if not calculated yet.
    _STORAGE_ double creport() const;
    double report() noexcept {
        if(!is_calculated_) sum();
        return creport();
    }

    // Returns error estimate
    _STORAGE_ double cest_err() const;
    _STORAGE_ double est_err()  noexcept;

    // Returns string representation
    _STORAGE_ std::string to_string() const;
    // Descriptive string.
    _STORAGE_ std::string desc_string() const;

    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval >> (64u - np_)), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval >> (64u - np_)), lzt(clz(((hashval << 1)|1) << (np_ - 1)) + 1);
        core_[index] = std::max(core_[index], lzt);
#endif
    }

    INLINE void addh(uint64_t element) {
        element = wang_hash(element);
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

    // Reset.
    _STORAGE_ void clear();
    hll_t(hll_t&&) = default;
    hll_t(const hll_t &other):
        np_(other.np_), core_(other.core_), value_(other.value_),
        is_calculated_(other.is_calculated_), use_ertl_(other.use_ertl_),
        nthreads_(other.nthreads_) {}
    hll_t& operator=(const hll_t &other) {
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
    hll_t& operator=(hll_t&&) = default;

    _STORAGE_ hll_t &operator+=(const hll_t &other);
    _STORAGE_ hll_t &operator&=(const hll_t &other);

    // Clears, allows reuse with different np.
    _STORAGE_ void resize(size_t new_size);
    bool get_use_ertl() const {return use_ertl_;}
    void set_use_ertl(bool val) {use_ertl_ = val;}
    // Getter for is_calculated_
    bool is_ready() const {return is_calculated_;}
    void not_ready() {is_calculated_ = false;}
    void set_is_ready() {is_calculated_ = true;}
    bool may_contain(uint64_t hashval) const {
        // This returns false positives, but never a false negative.
        return core_[hashval >> (64u - np_)] >= clz(hashval << np_) + 1;
    }

    bool within_bounds(uint64_t actual_size) const {
        return std::abs(actual_size - creport()) < relative_error() * actual_size;
    }

    bool within_bounds(uint64_t actual_size) {
        return std::abs(actual_size - report()) < est_err();
    }
    const auto &core()    const {return core_;}
    const uint8_t *data() const {return core_.data();}

    auto p() const {return np_;}
    auto q() const {return 64 - np_;}
    _STORAGE_ void free();
    _STORAGE_ void write(FILE *fp);
    _STORAGE_ void write(gzFile fp);
    _STORAGE_ void write(const char *path, bool write_gz=false);
    void write(const std::string &path, bool write_gz=false) {write(path.data(), write_gz);}
    _STORAGE_ void read(FILE *fp);
    _STORAGE_ void read(gzFile fp);
    _STORAGE_ void read(const char *path, bool read_gz=false);
    void read(const std::string &path) {read(path.data());}
#if _POSIX_VERSION
    _STORAGE_ void write(int fileno);
    _STORAGE_ void read(int fileno);
#endif

    size_t size() const {return size_t(m());}
};


// Returns the size of a symmetric set difference.
double operator^(hll_t &first, hll_t &other);
// Returns the set intersection of two sketches.
hll_t operator&(hll_t &first, hll_t &other);
// Returns the size of the set intersection
double intersection_size(hll_t &first, hll_t &other) noexcept;
double jaccard_index(hll_t &first, hll_t &other) noexcept;
double jaccard_index(const hll_t &first, const hll_t &other);
// Returns a HyperLogLog union
hll_t operator+(const hll_t &one, const hll_t &other);

namespace detail {
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
    if((value = (alpha * m * m / sum)) < detail::small_range_correction_threshold(m)) {
        if(counts[0]) {
            LOG_DEBUG("Small value correction. Original estimate %lf. New estimate %lf.\n",
                       value, m * std::log((double)m / counts[0]));
            value = m * std::log((double)(m) / counts[0]);
        }
    } else if(value > detail::LARGE_RANGE_CORRECTION_THRESHOLD) {
        // Reuse sum variable to hold correction.
        sum = -std::pow(2.0L, 32) * std::log1p(-std::ldexp(value, -32));
        if(std::isnan(sum)) {
            LOG_WARNING("Large range correction returned nan. Defaulting to regular calculation.\n");
        } else value = sum;
    }
    return value;
}

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


} // namespace detail

static inline double union_size(const hll_t &h1, const hll_t &h2) {
    using detail::SIMDHolder;
    assert(h1.m() == h2.m());
    using SType = typename SIMDHolder::SType;
    uint64_t counts[64]{0};
    // We can do this because we use an aligned allocator.
    const SType *p1(reinterpret_cast<const SType *>(h1.data())), *p2(reinterpret_cast<const SType *>(h2.data()));
    SIMDHolder tmp;
    do {
        tmp.val = SIMDHolder::max_fn(*p1++, *p2++);
        tmp.inc_counts(counts);
    } while(p1 < reinterpret_cast<const SType *>(&(*h1.core().cend())));
    return detail::calculate_estimate(counts, h1.get_use_ertl(), h1.m(), h1.p(), h1.alpha());
}

static inline double intersection_size(const hll_t &h1, const hll_t &h2) {
    return std::max(0., h1.creport() + h2.creport() - union_size(h1, h2));
}

} // namespace hll

#ifdef ENABLE_HLL_DEVELOP
#pragma message("hll develop enabled (-DENABLE_HLL_DEVELOP)")
#include "hll_dev.h"
#endif

#ifdef HLL_HEADER_ONLY
#  include "hll.cpp"
#endif

#endif // #ifndef HLL_H_
