#pragma once
#ifndef UPDATE_H__
#include "common.h"

namespace sketch {
namespace update {
struct Increment {
    // Saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) const {
        if(static_cast<IntType>(ref) < maxval)
            ref = static_cast<IntType>(ref) + 1;
        //ref += (ref < maxval);
    }
    template<typename T, typename Container, typename IntType>
    void operator()(std::vector<T> &ref, Container &con, IntType nbits) const {
            int64_t count = con[ref[0]];
            ++count;
            if(range_check<typename std::decay_t<decltype(*(std::declval<Container>().cbegin()))>>(nbits, count) == 0) {
                for(const auto el: ref)
                    con[el] = count;
            }
    }
    template<typename... Args>
    Increment(Args &&... args) {}
    static uint64_t est_count(uint64_t val) {
        return val;
    }
    template<typename T1, typename T2>
    static auto combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        return RetType(i) + RetType(j);
    }
};
struct PowerOfTwo {
    common::DefaultRNGType rng_;
    uint64_t  gen_;
    uint8_t nbits_;
    // Also saturates
    template<typename T, typename IntType>
    void operator()(T &ref, IntType maxval) {
#if !NDEBUG
        std::fprintf(stderr, "maxval: %zu. ref: %zu\n", size_t(maxval), size_t(ref));
#endif
        if(static_cast<IntType>(ref) == 0) ref = 1;
        else {
            if(ref >= maxval) return;
            if(HEDLEY_UNLIKELY(nbits_ < ref)) gen_ = rng_(), nbits_ = 64;
            const IntType oldref = ref;
            ref = oldref + ((gen_ & (UINT64_C(-1) >> (64 - oldref))) == 0);
            gen_ >>= oldref, nbits_ -= oldref;
        }
    }
    template<typename T, typename Container, typename IntType, typename Alloc>
    void operator()(std::vector<T, Alloc> &ref, Container &con, IntType nbits) {
        uint64_t val = con[ref[0]];
        if(val == 0) {
            for(const auto el: ref)
                con[el] = 1;
        } else {
            if(HEDLEY_UNLIKELY(nbits_ < val)) gen_ = rng_(), nbits_ = 64;
            auto oldval = val;
            if((gen_ & (UINT64_C(-1) >> (64 - val))) == 0) {
                ++val;
                if(range_check(nbits, val) == 0)
                    for(const auto el: ref)
                        con[el] = val;
            }
            gen_ >>= oldval;
            nbits_ -= oldval;
        }
    }
    template<typename T1, typename T2>
    static auto combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        RetType i_(i), j_(j);
        return std::max(i_, j_) + (i == j);
    }
    PowerOfTwo(uint64_t seed=0): rng_(seed), gen_(rng_()), nbits_(64) {}
    static constexpr uint64_t est_count(uint64_t val) {
        return val ? uint64_t(1) << (val - 1): 0;
    }
};
struct CountSketch {
    // Saturates
    template<typename T, typename IntType, typename IntType2>
    void operator()(T &ref, IntType maxval, IntType2 hash) const {
        ref = int64_t(ref) + (hash&1 ? 1: -1);
    }
    template<typename T, typename Container, typename IntType>
    ssize_t operator()(std::vector<T> &ref, std::vector<T> &hashes, Container &con, IntType nbits) const {
        using IDX = std::decay_t<decltype(con[0])>;
        IDX newval;
        std::vector<IDX> s;
        assert(ref.size() == hashes.size());
        for(size_t i(0); i < ref.size(); ++i) {
            newval = con[ref[i]] + (hashes[i]&1 ? 1: -1);
            s.push_back(newval);
            if(range_check<IDX>(nbits, newval) == 0)
                con[ref[i]] = newval;
        }
        if(s.size()) {
            common::sort::insertion_sort(s.begin(), s.end());
            return (s[s.size()>>1] + s[(s.size()-1)>>1]) >> 1;
        }
        return 0;
    }
    template<typename... Args>
    static void Increment(Args &&... args) {}
    uint64_t est_count(uint64_t val) const {
        return val;
    }
    template<typename T1, typename T2>
    static uint64_t combine(const T1 &i, const T2 &j) {
        using RetType = std::common_type_t<T1, T2>;
        std::fprintf(stderr, "[%s:%d:%s] I'm not sure this is actually right; this is essentially a placeholder.\n", __FILE__, __LINE__, __PRETTY_FUNCTION__);
        return RetType(i) + RetType(j);
    }
    template<typename... Args>
    CountSketch(Args &&... args) {}
};

} // update
} // sketch
#define UPDATE_H__
#endif /* UPDATE_H__ */
