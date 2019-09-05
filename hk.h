#include "./common.h"
#include "aesctr/wy.h"

namespace sketch {

static constexpr uint64_t bitmask(size_t n) {
    uint64_t x = uint64_t(1) << (n - 1);
    x ^= x - 1;
    return x;
}

using namespace common;

namespace hk {

template<size_t fpsize, size_t ctrsize=64-fpsize, typename Hasher=hash::WangHash, typename Policy=policy::SizeDivPolicy<uint64_t>, typename RNG=wy::WyHash<uint64_t>, typename Allocator=common::Allocator<uint64_t>>
class HeavyKeeper {

    static_assert(fpsize + ctrsize > 0 && 64 % (fpsize + ctrsize) == 0, "fpsize and ctrsize must be evenly divisible into our 64-bit words and at least one must be nonzero.");

    // Members
    Policy pol_;
    size_t nh_;
    std::vector<uint64_t, Allocator> data_;
    Hasher hasher_;
    RNG rng_;
    std::uniform_real_distribution<double> gen;
    double b_;
public:
    static constexpr size_t VAL_PER_REGISTER = 64 / (fpsize + ctrsize);
    // Constructor
    template<typename...Args>
    HeavyKeeper(size_t requested_size, size_t subtables, float pdec=1.08, Args &&...args):
        pol_(requested_size), nh_(subtables),
        data_((requested_size * subtables + (VAL_PER_REGISTER - 1)) / VAL_PER_REGISTER),
        hasher_(std::forward<Args>(args)...),
        b_(pdec)
    {
        assert(subtables);
#if VERBOSE_AF
        std::fprintf(stderr, "fpsize: %zu. ctrsize: %zu. requested size: %zu. actual size: %zu. Overflow check? %d. nhashes: %zu\n", fpsize, ctrsize, requested_size, pol_.nelem(), 1, nh_);
#endif
    }

    HeavyKeeper(const HeavyKeeper &o) = default;
    HeavyKeeper(HeavyKeeper &&o)      = default;
    HeavyKeeper& operator=(HeavyKeeper &&o)      = default;
    HeavyKeeper& operator=(const HeavyKeeper &o) = default;


    // utilities
    template<typename T, typename=std::enable_if_t<!std::is_same<T, uint64_t>::value>>
    uint64_t hash(const T &x) const {return hasher_(x);}
    uint64_t hash(uint64_t x) const {return hasher_(x);}
    uint64_t hash(uint32_t x) const {return hasher_(x);}
    uint64_t hash(int64_t x) const {return hasher_(uint64_t(x));}
    uint64_t hash(int32_t x) const {return hasher_(uint32_t(x));}
    void seed(uint64_t x) const {rng_.seed(x);}

    void clear() {
        std::memset(data_.data(), 0, sizeof(data_[0]) * data_.size());
    }

    static constexpr uint64_t sig_size = fpsize + ctrsize;
    static constexpr uint64_t sig_mask         = bitmask(sig_size);
    static constexpr uint64_t sig_mask_at_pos(size_t i) {
        return sig_mask << (i % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER);
    }
    static_assert(sig_size <= 64, "For sig_mask to be greater than 64, we'd need to use a larger value type.");
    static constexpr uint64_t fingerprint_mask = bitmask(fpsize) << ctrsize;
    static constexpr uint64_t count_mask       = bitmask(ctrsize);

    std::pair<uint64_t, uint64_t> decode(uint64_t x) const {
        return std::make_pair(x & count_mask, (x & fingerprint_mask) >> ctrsize);
    }
    uint64_t encode(uint64_t count, uint64_t fp) const {
        assert(fp < (1ull << fpsize));
        assert(count < (1ull << ctrsize));
        return count | (fp << ctrsize);
    }

    uint64_t from_index(size_t i, size_t subidx) const {
        auto dataptr = data_.data() + (subidx * pol_.nelem() / VAL_PER_REGISTER);
        assert(dataptr < &data_[data_.size()] || !std::fprintf(stderr, "subidx: %zu. nelem: %zu. data size: %zu\n", subidx, pol_.nelem(), data_.size()));
        auto pos = pol_.mod(i);
        uint64_t value = dataptr[pos / VAL_PER_REGISTER];
        CONST_IF(VAL_PER_REGISTER > 1) {
            value = (value >> ((pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER))) & sig_mask;
        }
        return value;
    }

    void store(size_t pos, size_t subidx, uint64_t fp, uint64_t count) {
        assert(subidx < nh_);
        auto dataptr = data_.data() + (subidx * pol_.nelem() / VAL_PER_REGISTER);
        assert(dataptr < data_.data() + data_.size());
        uint64_t to_insert = encode(count, fp);
        //std::fprintf(stderr, "Position is %zu vs pol nelem %zu. pos: %zu\n", dataptr - data_.data(), pol_.nelem(), pos);
        CONST_IF(VAL_PER_REGISTER == 1) {
            dataptr[pos] = to_insert;
            return;
        }
        size_t shift = ((pos % VAL_PER_REGISTER) * (64 / VAL_PER_REGISTER));
        to_insert <<= shift;
        auto &r = dataptr[pos / VAL_PER_REGISTER];
        r &= ~(bitmask(64 / VAL_PER_REGISTER) << shift);
        r |= to_insert;
    }
    bool random_sample(size_t count) {
        return count && gen(rng_) <= std::pow(b_, -ssize_t(count));
        //return count != 0 && rng_() <= uint64_t(std::ldexp(std::pow(b_, -ssize_t(count)), 64));
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
            assert(encode(count, fp) == from_index(pos, i));
            //std::fprintf(stderr, "%zu/%zu\n", size_t(encode(count, fp)), from_index(pos, i));
            if(count == 0) {
                //std::fprintf(stderr, "first entry for x = %zu\n", size_t(x));
                store(pos, i, newfp, 1);
                maxv = std::max(uint64_t(1), maxv);
            } else if(fp == newfp) {
                //std::fprintf(stderr, "pos/sig matched for entry for x = %zu\n", size_t(x));
                store(pos, i, newfp, count == count_mask ? count: count);
                maxv = std::max(uint64_t(1), uint64_t(count));
            } else {
                //std::fprintf(stderr, "pos/sig didn't match for %zu at count %zu\n", size_t(x), count);
                if(random_sample(count)) {
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
            auto p = decode(from_index(pos, i));
            auto count = p.first, fp = p.second;
            if(fp == newfp) {
                ret = std::max(ret, count);
            }
            if(++i == nh_) return ret;
            wy::wyhash64_stateless(&x);
        }
    }
    auto est_count(uint64_t x) {
        return query(x);
    }
    template<typename T>
    uint64_t est_count(const T &x) const {
        return query(x);
    }
    HeavyKeeper &operator|=(const HeavyKeeper &o) {
        uint64_t ret = 0;
        size_t data_index = 0;
        for(size_t subidx = 0; subidx < nh_; ++subidx) {
            for(size_t i = 0; i < o.size(); ++i) {
                uint64_t lv = data_[i], rv = o.data_[i];
                for(size_t j = 0; j < VAL_PER_REGISTER; ++j, ++data_index) {
                    auto mask = sig_mask_at_pos(j);
                    uint64_t llv = lv & mask, lrv = rv & mask;
                    auto llp = decode(llv), lrp = decode(lrv);
                    auto lc = llp.first, lpf = llp.second,
                         rc = lrp.first, rf = lrp.seccond;
                    if(lpf == rf) { // Easily predicted branch first
                        auto newc = lc + rc;
                        if(VAL_PER_REGISTER == 1) {
                            data_[data_index] = (lpf << ctrsize) | (lc + rc);
                            assert(newc >= lc && newc >= rc);
                        } else {
                            store(data_index, subidx, lpf, newc);
                        }
                    } else {
                        auto newc = std::max(lc, rc), newsub = std::min(lc, rc);
                        static constexpr bool slow_way = false;
                        if(slow_way) {
                            while(newsub--)
                                if(random_sample(newc))
                                    --newc;
                        } else newc -= newsub; // Not rigorous, but an exact formula would be effort/perhaps costly.
                        if(newc == 0)
                            store(data_index, subidx, 0, 0);
                        else
                            store(data_index, subidx, newc == lc ? lpf: rf, newc);
                    }
                }
            }
        }
        return *this;
    }
    HeavyKeeper &operator+=(const HeavyKeeper &o) {return *this |= o;}
    HeavyKeeper operator+(const HeavyKeeper &x) const {
        HeavyKeeper cpy(*this);
        cpy += x;
        return cpy;
    }
};

} // namespace hk

using namespace hk;



} // namespace sketch
