#include "./common.h"
#include "aesctr/wy.h"

namespace sketch {

using namespace common;

template<size_t fpsize, size_t ctrsize=64-fpsize, typename Hasher=hash::WangHash, typename Policy=policy::SizeDivPolicy<uint64_t>, typename RNG=wy::WyHash<uint64_t>, typename Allocator=common::Allocator<uint64_t>>
class HeavyKeeper {

    static_assert(fpsize + ctrsize > 0 && 64 % (fpsize + ctrsize) == 0, "fpsize and ctrsize must be evenly divisible into our 64-bit words and at least one must be nonzero.");

    // Members
    Policy pol_;
    size_t nh_;
    std::vector<uint64_t, Allocator> data_;
    Hasher hasher_;
    RNG rng_;
    double b_;
public:
    static constexpr size_t VAL_PER_REGISTER = 64 / (fpsize + ctrsize);
    // Constructor
    template<typename...Args>
    HeavyKeeper(size_t requested_size, size_t subtables, float pdec=1.08, Args &&...args): pol_(requested_size), nh_(subtables), data_((requested_size * subtables) / VAL_PER_REGISTER), hasher_(std::forward<Args>(args)...), b_(pdec) {
        assert(subtables);
        std::fprintf(stderr, "fpsize: %zu. ctrsize: %zu. requested size: %zu. actual size: %zu. Overflow check? %d. nhashes: %zu\n", fpsize, ctrsize, requested_size, pol_.nelem(), 1, nh_);
    }


    // utilities
    template<typename T, typename=std::enable_if_t<!std::is_same<T, uint64_t>::value>>
    uint64_t hash(const T &x) const {return hasher_(x);}
    void seed(uint64_t x) const {rng_.seed(x);}

    static constexpr uint64_t sig_size = fpsize + ctrsize;
    static constexpr uint64_t fingerprint_mask = ((1ull << fpsize) - 1) << ctrsize;
    static constexpr uint64_t count_mask       = (1ull << ctrsize) - 1;

    std::pair<uint64_t, uint64_t> decode(uint64_t x) const {
        return std::make_pair(x & count_mask, (x & fingerprint_mask) >> ctrsize);
    }
    uint64_t encode(uint64_t count, uint64_t fp) const {
        assert(fp < (1ull << fpsize));
        assert(count < (1ull << ctrsize));
        return count | (fp << ctrsize);
    }

    uint64_t from_index(size_t i, size_t subidx) const {
        auto dataptr = data_.data() + subidx * pol_.nelem();
        auto pos = pol_.mod(i);
        uint64_t value = dataptr[pos / VAL_PER_REGISTER];
        CONST_IF(VAL_PER_REGISTER > 1) {
            static constexpr uint64_t mask = VAL_PER_REGISTER > 1 ? (1ull << (64 / VAL_PER_REGISTER)) - 1: uint64_t(-1);
            value = (value >> ((pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER))) & mask;
        }
        return value;
    }

    void store(size_t pos, size_t subidx, uint64_t fp, uint64_t count) {
        auto dataptr = data_.data() + subidx * pol_.nelem();
        CONST_IF(VAL_PER_REGISTER == 1) {
            dataptr[pos] = encode(count, fp);
            return;
        }
        uint64_t to_insert = encode(count, fp);
        to_insert <<= (pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER);
        dataptr[pos] = (dataptr[pos] & ~(((1ull << (64 / VAL_PER_REGISTER)) - 1) << (pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER))) // zero out
            | to_insert; // add new
    }

    static constexpr uint64_t max_fp() {return ((1ull << fpsize) - 1);}
    template<typename T>
    auto addh(const T &x) {return this->add(hash(x));}
    auto add(uint64_t x) {
        uint64_t maxv = 0;
        for(size_t i = 0;;) {
            size_t pos = pol_.mod(x), newfp = pol_.div_.div(x) & max_fp();
#if __cplusplus >= 201703L
            auto [count, fp] = decode(from_index(pos, i));
#else
            auto vals = decode(from_index(pos, i));
            auto count = vals.first, fp = vals.second;
#endif
            if(count == 0) {
                //std::fprintf(stderr, "first entry for x = %zu\n", size_t(x));
                store(pos, i, newfp, 1);
                maxv = std::max(uint64_t(1), maxv);
            }
            else if(fp == newfp) {
                //std::fprintf(stderr, "pos/sig matched for entry for x = %zu\n", size_t(x));
                store(pos, i, newfp, ++count);
                maxv = std::max(uint64_t(1), uint64_t(count));
            } else {
                //std::fprintf(stderr, "pos/sig didn't match for %zu at count %zu\n", size_t(x), count);
                if(rng_() < (std::numeric_limits<uint64_t>::max()) * std::pow(b_, -ssize_t(count))) {
                    if(--count == 0)
                        store(pos, i, newfp, 1);
                    else
                        store(pos, i, fp, count);
                    maxv = std::max(maxv, uint64_t(count));
                }
            }
            if(++i == nh_) return maxv;
            wy::wyhash64_stateless(&x);
        }
    }
    template<typename T>
    uint64_t query(const T &x) const {
        return queryh(hash(x));
    }
    uint64_t queryh(uint64_t x) const {
        uint64_t ret = 0;
        size_t pos = pol_.mod(x), newfp = pol_.div_.div(x) & ((1ull << fpsize) - 1);
        for(size_t i = 0;;) {
            auto [count, fp] = decode(from_index(pos, i));
            if(fp == newfp) {
                ret = std::max(ret, count);
            }
            if(++i == nh_) return ret;
            wy::wyhash64_stateless(&x);
        }
    }
};


} // namespace sketch
