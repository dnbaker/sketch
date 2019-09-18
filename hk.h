#include "./common.h"
#include "aesctr/wy.h"
#include "tsg.h"
#if !NDEBUG
#include <unordered_set>
#endif

namespace sketch {

static constexpr uint64_t bitmask(size_t n) {
    uint64_t x = uint64_t(1) << (n - 1);
    x ^= x - 1;
    return x;
}

inline namespace hk {

namespace detail {
static constexpr inline bool is_valid_decay_base(double x) {
    return x >= 1.;
}
}

template<size_t fpsize, size_t ctrsize=64-fpsize, typename Hasher=hash::WangHash, typename Policy=policy::SizeDivPolicy<uint64_t>, typename RNG=wy::WyHash<uint64_t>, typename Allocator=common::Allocator<uint64_t>>
class HeavyKeeper {

    static_assert(fpsize > 0 && ctrsize > 0 && 64 % (fpsize + ctrsize) == 0, "fpsize and ctrsize must be evenly divisible into our 64-bit words and at least one must be nonzero.");

    struct decoded_register: std::pair<uint64_t, uint64_t> {
        uint64_t &count() {return this->first;}
        uint64_t &fp() {return this->second;}
    };

    // Members
    Policy pol_;
    const size_t nh_;
    std::vector<uint64_t, Allocator> data_;
    Hasher hasher_;
    const double b_;
public:
    static constexpr size_t VAL_PER_REGISTER = 64 / (fpsize + ctrsize);
    // Constructor
    template<typename...Args>
    HeavyKeeper(size_t requested_size, size_t subtables, double pdec=1.08, Args &&...args):
        pol_(requested_size), nh_(subtables),
        data_((requested_size * subtables + (VAL_PER_REGISTER - 1)) / VAL_PER_REGISTER),
        hasher_(std::forward<Args>(args)...),
        b_(pdec)
    {
        assert(subtables);
        if(!detail::is_valid_decay_base(pdec))
            throw UnsatisfiedPreconditionError(std::string("pdec is not valid (>= 1.). Value: ") + std::to_string(pdec));
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

    decoded_register decode(uint64_t x) const {
        decoded_register ret;
        ret.count() = x & count_mask;
        ret.fp()    = (x & fingerprint_mask) >> ctrsize;
        return ret;
    }
    uint64_t encode(uint64_t count, uint64_t fp) const {
        assert(fp < (1ull << fpsize));
        assert(count < (1ull << ctrsize));
        return count | (fp << ctrsize);
    }
    uint64_t encode(decoded_register reg) const {
        return encode(reg.count(), reg.fp());
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
        static thread_local std::uniform_real_distribution<double> gen;
        static thread_local tsg::ThreadSeededGen<RNG> rng;
        auto limit =  std::pow(b_, -ssize_t(count));
        auto val = gen(rng);
#if VERBOSE_AF
        std::fprintf(stderr, "limit: %f. count: %zu. val: %f, pass: %s\n", limit, count, val, val <= limit ? "true": "false");
#endif
        return count && val <= limit;
    }
    static constexpr uint64_t max_fp() {return ((1ull << fpsize) - 1);}
    template<typename T2>
    void divmod(uint64_t x, T2 &pos, T2 &fp) const {
        pos = pol_.mod(x);
        fp = pol_.div_.div(x) & max_fp();
    }
    void display_encoded_value(uint64_t x) {
        auto dec = decode(x);
        std::fprintf(stderr, "dec.count() = %zu. dec.fp() (hash) = %zu\n", dec.count(), dec.fp());
    }
    uint64_t addh(uint64_t x) {
        x = hash(x);
        uint64_t maxv = 0;
        unsigned i = 0;
        FOREVER {
            size_t pos, newfp;
            divmod(x, pos, newfp);
            decoded_register vals = decode(from_index(pos, i));
            auto count = vals.count();
            auto fp = vals.fp();
            assert(encode(count, fp) == from_index(pos, i));
            assert(decode(encode(count, fp)).first == count);
            assert(decode(encode(count, fp)).second == fp);
#if VERBOSE_AF
            std::fprintf(stderr, "Encoded value from previous register: ");
            display_encoded_value(encode(count, fp));
            std::fprintf(stderr, "pos: %" PRIu64 ", fp: %" PRIu64 "\n", pos, fp);
#endif
            if(count == 0) {
#if VERBOSE_AF
                //Empty bucket -- simply insert
                std::fprintf(stderr, "first entry for x = %zu\n", size_t(x));
#endif
                store(pos, i, newfp, 1);
                assert(decode(from_index(pos, i)).second == newfp);
                assert(decode(from_index(pos, i)).fp() == newfp);
                assert(decode(from_index(pos, i)).first == 1);
                assert(decode(from_index(pos, i)).count() == 1);
                maxv += maxv == 0;
            } else if(fp == newfp) {
                count += count < count_mask;
                store(pos, i, newfp, count);
                assert(decode(from_index(pos, i)).second == newfp);
                assert(decode(from_index(pos, i)).fp() == newfp);
                assert(decode(from_index(pos, i)).first == count);
                assert(decode(from_index(pos, i)).count() == count);
                maxv = std::max(maxv, uint64_t(count));
            } else {
                if(random_sample(count)) {
                    if(--count == 0) {
                        store(pos, i, newfp, 1);
                        assert(decode(from_index(pos, i)).second == newfp);
                        assert(decode(from_index(pos, i)).fp() == newfp);
                        assert(decode(from_index(pos, i)).first == 1);
                        assert(decode(from_index(pos, i)).count() == 1);
                        maxv = std::max(maxv, size_t(1));
                    } else {
                        store(pos, i, fp, count);
                        assert(decode(from_index(pos, i)).second == fp);
                        assert(decode(from_index(pos, i)).fp() == fp);
                        assert(decode(from_index(pos, i)).first == count);
                    }
                }
                assert(decode(from_index(pos, i)).first != 0);
            }
            if(++i == nh_) break;
            wy::wyhash64_stateless(&x);
        }
        return maxv;
    }
    uint64_t query(uint64_t x) const {
        x = hash(x);
        uint64_t ret = 0;
        unsigned i = 0;
        FOREVER {
            size_t pos, newfp;
            divmod(x, pos, newfp);
            auto p = decode(from_index(pos, i));
            auto count = p.count(), fp = p.fp();
            if(fp == newfp) ret = std::max(ret, count);
            if(++i == nh_) break;
            wy::wyhash64_stateless(&x);
        }
        return ret;
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

} // namespace sketch
