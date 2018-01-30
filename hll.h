#ifndef HLL_H_
#define HLL_H_
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>
#include "logutil.h"
#include "sseutil.h"
#include "util.h"
#include "math.h"
#include "unistd.h"

#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#include "x86intrin.h"
#define HAS_AVX_512 _FEATURE_AVX512F

namespace hll {

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
// For our hash function, there is only 1 64-bit integer value which causes this problem.
// I'd expect that this is acceptable. And on Haswell+, this value is the correct value.
#if __GNUC__ || __clang__
#ifdef AVOID_CLZ_UNDEF
#define DEF_CHECK(fn) x ? fn(x) : sizeof(x) * CHAR_BIT
#else
#define DEF_CHECK(fn) fn(x)
#endif

constexpr INLINE unsigned clz(unsigned long long x) {
    return DEF_CHECK(__builtin_clzll);
}
constexpr INLINE unsigned clz(unsigned long x) {
    return DEF_CHECK(__builtin_clzl);
}
constexpr INLINE unsigned clz(unsigned x) {
    return DEF_CHECK(__builtin_clz);
}
constexpr INLINE unsigned ctz(unsigned long long x) {
    return DEF_CHECK(__builtin_ctzll);
}
constexpr INLINE unsigned ctz(unsigned long x) {
    return DEF_CHECK(__builtin_ctzl);
}
constexpr INLINE unsigned ctz(unsigned x) {
    return DEF_CHECK(__builtin_ctz);
}
#undef DEF_CHECK
#else
#pragma message("Using manual clz instead of gcc/clang __builtin_*")
#error("Have not created a manual ctz function. Must be compiled with gcc or clang.")
#define clz(x) clz_manual(x)
// https://en.wikipedia.org/wiki/Find_first_set#CLZ
// Modified for constexpr, added 64-bit overload.
#endif

static_assert(clz(0x0000FFFFFFFFFFFFull) == 16, "64-bit clz hand-rolled failed.");
static_assert(clz(0x000000000FFFFFFFull) == 36, "64-bit clz hand-rolled failed.");
static_assert(clz(0x0000000000000FFFull) == 52, "64-bit clz hand-rolled failed.");
static_assert(clz(0x0000000000000003ull) == 62, "64-bit clz hand-rolled failed.");
static_assert(clz(0x0000013333000003ull) == 23, "64-bit clz hand-rolled failed.");


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
    void sum();
    void parsum(int nthreads=-1, size_t per_batch=1<<18);

    // Returns cardinality estimate. Sums if not calculated yet.
    double creport() const;
    double report() noexcept {
        if(!is_calculated_) sum();
        return creport();
    }

    // Returns error estimate
    double cest_err() const;
    double est_err()  noexcept;

    // Returns string representation
    std::string to_string() const;
    // Descriptive string.
    std::string desc_string() const;

    INLINE void add(uint64_t hashval) {
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval >> (64u - np_)), lzt(clz(hashval << np_) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval >> (64u - np_)), lzt(clz(hashval << np_) + 1);
        if(core_[index] < lzt) core_[index] = lzt;
#endif
    }

    INLINE void addh(uint64_t element) {add(wang_hash(element));}

    // Reset.
    void clear();
    hll_t(const hll_t&) = default;
    hll_t(hll_t&&) = default;
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

    hll_t const &operator+=(const hll_t &other);
    hll_t const &operator&=(const hll_t &other);

    // Clears, allows reuse with different np.
    void resize(size_t new_size);
    // Getter for is_calculated_
    bool get_use_ertl() const {return use_ertl_;}
    void set_use_ertl(bool val) {use_ertl_ = val;}
    bool is_ready() const {return is_calculated_;}
    void not_ready() {is_calculated_ = false;}
    void set_is_ready() {is_calculated_ = true;}

    bool within_bounds(uint64_t actual_size) const {
        return std::abs(actual_size - creport()) < relative_error() * actual_size;
    }

    bool within_bounds(uint64_t actual_size) {
        return std::abs(actual_size - report()) < est_err();
    }
    const auto &data() const {return core_;}

    auto p() const {return np_;}
    void free();
    void write(FILE *fp);
    void write(const char *path);
    void write(const std::string &path) {write(path.data());}
    void read(FILE *fp);
    void read(const char *path);
    void read(const std::string &path) {read(path.data());}
#if _POSIX_VERSION
    void write(int fileno);
    void read(int fileno);
#endif

    size_t size() const {return size_t(1) << np_;}
};

class hlldub_t: public hll_t {
    // hlldub_t inserts each value twice (forward and reverse)
    // and simply halves cardinality estimates.
public:
    template<typename... Args>
    hlldub_t(Args &&...args): hll_t(std::forward<Args>(args)...) {}
    INLINE void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(hashval >> p()) + 1);
            core_[index] < lzt;
            __sync_bool_compare_and_swap(core_.data() + index, core_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(hashval >> p()) + 1);
        if(core_[index] < lzt) core_[index] = lzt;
#endif
    }
    double report() {
        sum();
        return hll_t::report() * 0.5;
    }
    double creport() {
        return hll_t::creport() * 0.5;
    }

    INLINE void addh(uint64_t element) {add(wang_hash(element));}

};

class dhll_t: public hll_t {
    // dhll_t is a bidirectional hll sketch which does not currently support set operations
    // It is based on the idea that the properties of a hll sketch work for both leading and trailing zeros and uses them as independent samples.
    std::vector<uint8_t, Allocator> dcore_;
public:
    template<typename... Args>
    dhll_t(Args &&...args): hll_t(std::forward<Args>(args)...),
                            dcore_(1ull << hll_t::p()) {
    }
    void sum();
    void add(uint64_t hashval) {
        hll_t::add(hashval);
#ifndef NOT_THREADSAFE
        for(const uint32_t index(hashval & ((m()) - 1)), lzt(ctz(hashval >> p()) + 1);
            dcore_[index] < lzt;
            __sync_bool_compare_and_swap(dcore_.data() + index, dcore_[index], lzt));
#else
        const uint32_t index(hashval & (m() - 1)), lzt(ctz(hashval >> p()) + 1);
        if(dcore_[index] < lzt) dcore_[index] = lzt;
#endif
    }
    void addh(uint64_t element) {
        add(wang_hash(element));
    }
};

// Returns the size of a symmetric set difference.
double operator^(hll_t &first, hll_t &other);
// Returns the set intersection of two sketches.
hll_t operator&(hll_t &first, hll_t &other);
// Returns the size of the set intersection
double intersection_size(hll_t &first, hll_t &other) noexcept;
double intersection_size(hll_t &first, hll_t &other, hll_t &scratch_space) noexcept;
double intersection_size(const hll_t &first, const hll_t &other);
double intersection_size(const hll_t &first, const hll_t &other,
                         hll_t &scratch_space);
double jaccard_index(hll_t &first, hll_t &other) noexcept;
double jaccard_index(const hll_t &first, const hll_t &other);
double jaccard_index(hll_t &first, hll_t &other, hll_t &scratch) noexcept;
double jaccard_index(const hll_t &first, const hll_t &other, hll_t &scratch);
// Returns a HyperLogLog union
hll_t operator+(const hll_t &one, const hll_t &other);


} // namespace hll
#ifdef HLL_HEADER_ONLY
#  include "hll.cpp"
#endif

#endif // #ifndef HLL_H_
