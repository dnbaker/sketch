#ifndef CRUEL_BLOOM_H__
#define CRUEL_BLOOM_H__
#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cinttypes>
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
#include "unistd.h"
#include "x86intrin.h"
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif

#ifndef _VEC_H__ // In case vec.h has been used but with blaze before.
#  define NO_SLEEF
#  define NO_BLAZE
#  include "vec.h" // Import vec.h, but disable blaze and sleef.
#endif

#define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)


#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH 1
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if defined(NDEBUG)
#  if NDEBUG == 0
#    undef NDEBUG
#  endif
#endif



namespace bf {

/*
 * TODO: calculate distance *directly* without copying to another sketch!
 */

using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::size_t;

template<typename T>
static inline bool is_pow2(T val) {
    return val && (val & (val - 1)) == 0;
}

namespace hash {


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
  return key = key + (key << 31);
}

struct WangHash {
    using Space = vec::SIMDTypes<uint64_t>;
    using Type = typename vec::SIMDTypes<uint64_t>::Type;
    using VType = typename vec::SIMDTypes<uint64_t>::VType;
    auto operator()(uint64_t key) const {
        return wang_hash(key);
    }
    INLINE Type operator()(Type element) const {
        VType key = Space::add(Space::slli(element, 21), ~element); // key = (~key) + (key << 21);
        key = Space::srli(key.simd_, 24) ^ key.simd_; //key ^ (key >> 24)
        key = Space::add(Space::add(Space::slli(key.simd_, 3), Space::slli(key.simd_, 8)), key.simd_); // (key + (key << 3)) + (key << 8);
        key = key.simd_ ^ Space::srli(key.simd_, 14);  // key ^ (key >> 14);
        key = Space::add(Space::add(Space::slli(key.simd_, 2), Space::slli(key.simd_, 4)), key.simd_); // (key + (key << 2)) + (key << 4); // key * 21
        key = key.simd_ ^ Space::srli(key.simd_, 28); // key ^ (key >> 28);
        key = Space::add(Space::slli(key.simd_, 31), key.simd_);    // key + (key << 31);
        return key.simd_;
    }
    INLINE Type &apply_inplace(Type &element) const {
        element = Space::add(Space::slli(element, 21), ~element); // key = (~key) + (key << 21);
        element = Space::srli(element, 24) ^ element; //element ^ (element >> 24)
        element = Space::add(Space::add(Space::slli(element, 3), Space::slli(element, 8)), element); // (element + (element << 3)) + (element << 8);
        element = element ^ Space::srli(element, 14);  // element ^ (element >> 14);
        element = Space::add(Space::add(Space::slli(element, 2), Space::slli(element, 4)), element); // (element + (element << 2)) + (element << 4); // element * 21
        element = element ^ Space::srli(element, 28); // element ^ (element >> 28);
        element = Space::add(Space::slli(element, 31), element);    // element + (element << 31);
        return element;
    }
};

struct MurFinHash {
    using Space = vec::SIMDTypes<uint64_t>;
    using Type = typename vec::SIMDTypes<uint64_t>::Type;
    using VType = typename vec::SIMDTypes<uint64_t>::VType;
    INLINE uint64_t operator()(uint64_t key) const {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccd;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53;
        key ^= key >> 33;
        return key;
    }
    INLINE Type operator()(Type key) const {
        return this->operator()(*((VType *)&key));
    }
    INLINE Type operator()(VType key) const {
#if (HAS_AVX_512)
        static const Type mul1 = Space::set1(0xff51afd7ed558ccd);
        static const Type mul2 = Space::set1(0xc4ceb9fe1a85ec53);
#endif

#if !NDEBUG
        auto save = key.arr_[0];
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xff51afd7ed558ccd;});
#  else
        key = Space::mullo(key.simd_, mul1); // h *= 0xff51afd7ed558ccd;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
#if (HAS_AVX_512) == 0
        key.for_each([](uint64_t &x) {x *= 0xc4ceb9fe1a85ec53;});
#  else
        key = Space::mullo(key.simd_, mul2); // h *= 0xc4ceb9fe1a85ec53;
#endif
        key = Space::srli(key.simd_, 33) ^ key.simd_;  // h ^= h >> 33;
        assert(this->operator()(save) == key.arr_[0]);
        return key.simd_;
    }
};

} // namespace hash

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

namespace lut {
static constexpr uint8_t nbitsperhash [] {
/*
# Auto-generated using:
def ru(x):
    x -= 1
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 6; x |= x >> 16; x |= x >> 32
    x += 1
    return x
print("    %s" % ", ".join(map(str, map(ru, range(64)))))
*/
    0, 1, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
};
static const uint8_t nhashesper64bitword [] {
/*
# Auto-generated using:
def ru(x):
    x -= 1
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 6; x |= x >> 16; x |= x >> 32
    x += 1
    return x
print("    0xFFu, %s" % ", ".join(str(64 // ru(x)) for x in range(1, 63)))

*/
    0xFFu, 64, 32, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};
} // namespace lut

template<typename ValueType>
#if HAS_AVX_512
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX512>;
#elif __AVX2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::AVX>;
#elif __SSE2__
using Allocator = sse::AlignedAllocator<ValueType, sse::Alignment::SSE>;
#else
using Allocator = std::allocator<ValueType, ss::Alignment::Normal>;
#endif

// TODO: add a compact, 6-bit version
// For now, I think that it's preferable for thread safety,
// considering there's an intrinsic for the atomic load/store, but there would not
// be for bit-packed versions.



template<typename HashStruct=hash::WangHash>
class bfbase_t {
// HyperLogLog implementation.
// To make it general, the actual point of entry is a 64-bit integer hash function.
// Therefore, you have to perform a hash function to convert various types into a suitable query.
// We could also cut our memory requirements by switching to only using 6 bits per element,
// (up to 64 leading zeros), though the gains would be relatively small
// given how memory-efficient this structure is.

// Attributes
protected:
    uint8_t                                       np_;
    uint8_t                                       nh_;
    uint8_t                                     perh_; // Number of hashes per 64-bit hash
    uint8_t                                    qmark_; // No idea what to do with this yet.
    std::vector<uint64_t, Allocator<uint64_t>>  core_;
    std::vector<uint64_t, Allocator<uint64_t>> seeds_;
    uint64_t                                seedseed_;
public:
    static constexpr unsigned OFFSET = 6; // log2(CHAR_BIT * 8) == log2(64) == 6
    using HashType = HashStruct;
    const HashStruct        hf_;

    using VectorSpace = vec::SIMDTypes<uint64_t>;
    using VType       = typename vec::SIMDTypes<uint64_t>::VType;

    uint64_t m() const {return core_.size() << OFFSET;}
    uint64_t p() const {return np_ + OFFSET;}
    auto nhashes() const {return nh_;}
    uint64_t mask() const {
        return m() - UINT64_C(1);
    }
    bool is_empty() const {
        return np_ == OFFSET;
    }

    // Constructor
    explicit bfbase_t(size_t l2sz, unsigned nhashes, uint64_t seedval):
        np_(l2sz - OFFSET), nh_(nhashes), seedseed_(seedval), hf_{}
    {
        //if(l2sz < OFFSET) throw std::runtime_error("Need at least a power of size 6\n");
        resize(1ull << l2sz);
    }
    explicit bfbase_t(): bfbase_t(OFFSET, 1) {}
    void seed() {
        std::mt19937_64 mt(seedseed_);
        auto nperhash64 = lut::nhashesper64bitword[p()];
        assert(is_pow2(nperhash64));
        while(seeds_.size() * nperhash64 < nh_)
            if(auto val = mt(); std::find(seeds_.cbegin(), seeds_.cend(), val) == seeds_.cend())
                seeds_.emplace_back(val);
    }

    template<typename IndType>
    INLINE void set1(IndType ind) {
        ind &= mask();
        core_[ind >> OFFSET] |= (1 << (ind & (OFFSET - 1)));
        assert(is_set(ind));
    }

    auto is_set(uint64_t ind) const {
        ind &= mask();
        return core_[ind >> OFFSET] & (1 << (ind & (OFFSET - 1)));
    }

    INLINE bool all_set(const uint64_t &hv, unsigned n, unsigned shift) const {
        if(!is_set(hv)) return false;
        for(unsigned i(0); i < n; ++i)
            if(!is_set(hv >> (++i * shift)))
                return false;
        return true;
    }


    INLINE void sub_set1(const uint64_t &hv, unsigned n, unsigned shift) {
        set1(hv);
        for(unsigned subhind = 0; subhind < n; set1((hv >> (++subhind * shift))));
    }
    INLINE void addh(uint64_t element) {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()], npersimd = VectorSpace::COUNT * npw;
        const auto shift = lut::nbitsperhash[p()];
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        while(nleft > npersimd) {
            VType v(hf_(VectorSpace::set1(element) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {sub_set1(val, npw, shift);});
            nleft -= npersimd;
        }
        const uint64_t *sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            const auto todo = std::min(npw, nleft);
            sub_set1(hf_(element ^ *sptr++), todo, shift);
            nleft -= todo;
        }
    }
    INLINE void addh(const std::string &element) {
#ifdef ENABLE_CLHASH
        if constexpr(std::is_same<HashStruct, clhasher>::value) {
            addh(hf_(element));
        } else {
#endif
            addh(std::hash<std::string>{}(element));
#ifdef ENABLE_CLHASH
        }
#endif
    }
    // Reset.
    void clear() {
        VType v1 = VectorSpace::set1(0);
        for(VType *p1((VType *)&core_[0]), *p2((VType *)&core_[core_.size()]); p1 < p2; *p1++ = v1);
    }
    bfbase_t(bfbase_t&&) = default;
    bfbase_t(const bfbase_t &other) = default;
    bfbase_t& operator=(const bfbase_t &other) {
        // Explicitly define to make sure we don't do unnecessary reallocation.
        core_ = other.core_; np_ = other.np_; nh_ = other.nh_; seedseed_ = other.seedseed_; return *this;
    }
    bfbase_t& operator=(bfbase_t&&) = default;
    bfbase_t clone() const {return bfbase_t(np_, nh_, seedseed_);}
    bool same_params(const bfbase_t &other) const {
        return std::tie(np_, nh_, seedseed_) == std::tie(other.np_, other.nh_, other.seedseed_);
    }

    bfbase_t &operator+=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        VType *els((VType *)core_.data());
        const VType *oels((const VType *)other.core_.data());
        for(unsigned i = 0; i < (core_.size() / VectorSpace::COUNT / CHAR_BIT); ++i)
            els[i].simd_ = VectorSpace::or_fn(els[i].simd_, oels[i].simd_);
        return *this;
    }

    bfbase_t &operator&=(const bfbase_t &other) {
        if(!same_params(other)) {
            char buf[256];
            sprintf(buf, "For operator +=: np_ (%u) != other.np_ (%u)\n", np_, other.np_);
            throw std::runtime_error(buf);
        }
        unsigned i;
        VType *els((VType *)core_.data());
        const VType *oels((const VType *)other.core_.data());
        for(unsigned i = 0; i < (core_.size() / VectorSpace::COUNT / CHAR_BIT); ++i)
            els[i].simd_ = VectorSpace::and_fn(els[i].simd_, oels[i].simd_);
        return *this;
    }

    // Clears, allows reuse with different np.
    void resize(size_t new_size) {
        new_size = roundup64(new_size);
        clear();
        core_.resize(new_size >> OFFSET);
        clear();
        np_ = (std::size_t)std::log2(new_size) - OFFSET;
        seed();
        assert(np_ < 64); // To handle underflow
    }
    // Getter for is_calculated_
    bool may_contain(uint64_t val) const {
        bool ret = true;
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()], npersimd = VectorSpace::COUNT * npw;
        const auto shift = lut::nbitsperhash[p()];
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        const uint64_t *sptr;
        while(nleft > npersimd) {
            VType v(hf_(VectorSpace::set1(val) ^ (*seedptr++).simd_));
            v.for_each([&](const uint64_t &val) {ret &= all_set(val, npw, shift);});
            if(!ret) goto f;
            nleft -= npersimd;
        }
        sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            if((ret &= all_set(hf_(val ^ *sptr++), std::min(npw, nleft), shift)) == 0) goto f;
            nleft -= std::min(npw, nleft);
        }
        f:
        return ret;
    }
    void may_contain(const std::vector<uint64_t> vals, std::vector<uint64_t> &ret) const {
        return may_contain(vals.data(), vals.size(), ret);
    }
    void may_contain(const uint64_t *vals, size_t nvals, std::vector<uint64_t> &ret) const {
        // TODO: descend farther in batching, doing each subhash together for cache efficiency.
        ret.clear();
        ret.resize(nvals >> 6 + ((nvals & 0x63u) != 0), UINT64_C(-1));
        unsigned nleft = nh_, npw = lut::nhashesper64bitword[p()], npersimd = VectorSpace::COUNT * npw;
        const auto shift = lut::nbitsperhash[p()];
        const VType *seedptr = reinterpret_cast<const VType *>(&seeds_[0]);
        VType seed, v;
        while(nleft > npersimd) {
            seed.simd_ = (*seedptr++).simd_;
            for(unsigned  i(0); i < nvals; ++i) {
                bool is_present = true;
                v.simd_ = hf_(VectorSpace::set1(vals[i]) ^ seed.simd_);
                v.for_each([&](const uint64_t &val) {
                    ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(val, npw, shift)) << (i & 63u));
                });
            }
            nleft -= npersimd;
        }
        const uint64_t *sptr = reinterpret_cast<const uint64_t *>(seedptr);
        while(nleft) {
            uint64_t hv, seed = *sptr++;
            for(unsigned i(0); i < nvals; ++i) {
                hv = hf_(vals[i] ^ seed);
                ret[i >> 6] &= UINT64_C(-1) ^ (static_cast<uint64_t>(!all_set(hv, std::min(npw, nleft), shift)) << (i & 63u));
            }
            nleft -= std::min(npw, nleft);
        }
    }

    const auto &core()    const {return core_;}
    const uint8_t *data() const {return core_.data();}

    void free() {
        decltype(core_) tmp{};
        std::swap(core_, tmp);
    }
#if 0
    void write(FILE *fp) const {
        write(fileno(fp));
    }
    void write(gzFile fp) const {
#define CW(fp, src, len) do {if(gzwrite(fp, src, len) == 0) throw std::runtime_error("Error writing to file.");} while(0)
        uint32_t bf[]{is_calculated_, clamp_, estim_, jestim_, nthreads_};
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
        uint32_t bf[5];
        CR(fp, bf, sizeof(bf));
        is_calculated_ = bf[0];
        clamp_  = bf[1];
        estim_  = (EstimationMethod)bf[2];
        jestim_ = (JointEstimationMethod)bf[3];
        nthreads_ = bf[4];
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
        uint32_t bf[]{is_calculated_, clamp_, estim_, jestim_, nthreads_};
        ::write(fileno, bf, sizeof(bf));
        ::write(fileno, &np_, sizeof(np_));
        ::write(fileno, &value_, sizeof(value_));
        ::write(fileno, core_.data(), core_.size() * sizeof(core_[0]));
    }
    void read(int fileno) {
        uint32_t bf[5];
        ::read(fileno, bf, sizeof(bf));
        is_calculated_ = bf[0];
        clamp_         = bf[1];
        estim_         = (EstimationMethod)bf[2];
        jestim_        = (JointEstimationMethod)bf[3];
        nthreads_      = bf[4];
        ::read(fileno, &np_, sizeof(np_));
        ::read(fileno, &value_, sizeof(value_));
        core_.resize(m());
        ::read(fileno, core_.data(), core_.size());
    }
#endif
    bfbase_t operator+(const bfbase_t &other) const {
        if(!same_params(other))
            throw std::runtime_error("Different parameters.");
        bfbase_t ret(*this);
        ret += other;
        return ret;
    }
    double union_size(const bfbase_t &other) const {
        throw std::runtime_error("Need to write.");
        return 0.;
    }
    double jaccard_index(const bfbase_t &h2) const {
        throw std::runtime_error("Not finished");
        return std::max(0., 1.);
    }
    size_t size() const {return size_t(m());}
};

using bf_t = bfbase_t<>;

// Returns the size of the set intersection
template<typename BloomType>
inline double intersection_size(BloomType &first, BloomType &other) noexcept {
    first.csum(), other.csum();
    return intersection_size((const BloomType &)first, (const BloomType &)other);
}

template<typename BloomType>
inline double jaccard_index(const BloomType &h1, const BloomType &h2) {
    return h1.jaccard_index(h2);
}
template<typename BloomType>
inline double jaccard_index(BloomType &h1, BloomType &h2) {
    return h1.jaccard_index(h2);
}

template<typename BloomType>
static inline double intersection_size(const BloomType &h1, const BloomType &h2) {
    throw std::runtime_error("NotImplementedError");
    return 0.;
}

} // namespace bf

#endif // #ifndef CRUEL_BLOOM_H__
