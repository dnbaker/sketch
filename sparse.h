#pragma once
#include "hll.h"

namespace sketch {

namespace sparse {

// For HLL

struct SparseEncoding {
    static constexpr uint32_t get_index(uint32_t val);
    static constexpr uint8_t get_value(uint32_t val);
    static constexpr uint32_t encode_value(uint32_t index, uint8_t val) ;
};

struct SparseHLL32: public SparseEncoding {
    static constexpr uint32_t get_index(uint32_t val) {
        return val >> 8;
    }
    static constexpr uint8_t get_value(uint32_t val) {return static_cast<uint8_t>(val);} // Implicitly truncated, but we add a cast for clarity
    static constexpr uint32_t encode_value(uint32_t index, uint8_t val) {
        return (index << 8) | val;
    }
};

template<typename HashStruct=hash::WangHash>
class SparseHLL {
    int p_;
    std::vector<uint32_t> vals_;
    std::unique_ptr<std::array<uint32_t, 64>> sum_;
    double v_ = -1.;
public:
    SparseHLL(const hll::hllbase_t<HashStruct> &hll): p_(hll.p()) {
        for(size_t i = 0; i < hll.core().size(); ++i) {
            if(hll.core()[i]) {
                vals_.emplace_back(SparseHLL32::encode_value(i, hll.core()[i]));
            }
        }
    }
    void sort() {
        common::sort::default_sort(vals_.begin(), vals_.end(), [](auto x, auto y) {return SparseHLL32::get_index(x) < SparseHLL32::get_index(y);});
    }
    bool is_sorted() const {
        for(size_t i = 0; i < vals_.size() - 1; ++i) {
            if(SparseHLL32::get_index(vals_[i]) >= SparseHLL32::get_index(vals_[i + 1])) return false;
        }
        return true;
    }
    unsigned q() const {return 64 - p_;}
    SparseHLL(int p): p_(p) {}
    std::array<uint32_t, 64> sum() const {
        std::array<uint32_t, 64> ret{0};
        for(const auto v: vals_)
            ++ret[SparseHLL32::get_value(v)];
        ret[0] += (1ull << p_) - vals_.size();
        return ret;
    }
    std::array<uint32_t, 64> &count_sum() {
        if(!sum_) {
            sum_ = new std::array<uint32_t, 64>;
            *sum_ = sum();
        }
        return *sum_;
    }
    double report() {
        if(v_ >= 0) return v_;
        auto &csum = count_sum();
        return v_ = hll::detail::ertl_ml_estimate(csum, p_, q());
    }
    template<typename I>
    void fill_from_pairs(I it1, I it2) {
        while(it1 != it2) {
            auto v = *it1++;
            vals_.push_back(SparseHLL32::encode_value(v.first, v.second));
        }
    }
    template<typename I>
    void fill_from_packed(I it1, I it2) {
        while(it1 != it2) {
            auto v = *it1++;
            vals_.push_back(v);
        }
    }
    void clear() {
        v_ = -1.;
        vals_.clear();
    }
    double report() const {
        if(v_ >= 0) return v_;
        auto lsum = sum();
        return hll::detail::ertl_ml_estimate(lsum, p_, q());
    }
    std::array<double, 3> query(const hll::hllbase_t<HashStruct> &hll, const std::array<uint32_t, 64> *a=nullptr) const {
        std::array<uint32_t, 64> *tmp = a ? nullptr: reinterpret_cast<std::array<uint32_t, 64> *>(__builtin_alloca(sizeof(*a)));
        if(!a) {
            *tmp = hll::detail::sum_counts(hll.core());
        }
        const std::array<uint32_t, 64> &osum = a ? *a: *tmp;
        assert(is_sorted());
        auto it = vals_.begin();
        std::array<uint32_t, 64> &lsum = *sum_;
        std::array<uint32_t, 64> usum = osum;
        auto hcore = hll.core();
        for(const auto c: vals_) {
            const auto ind = SparseHLL32::get_index(c);
            const auto val = SparseHLL32::get_value(c);
            const auto oval = hcore[ind];
            if(hcore[ind] < val) {
                --usum[oval];
                ++usum[val];
            }
        }
        std::array<double, 3> ret;
        double myrep = this->report(), orep = hll.creport(), us = hll::detail::ertl_ml_estimate(usum, p_, q());
        double is = myrep + orep - us;
        ret[0] = myrep - is;
        ret[1] = orep - is;
        ret[2] = is;
        std::fprintf(stderr, "is: %lf. my size %lf o size %lf\n", is, myrep, orep);
        return ret;
    }
    double jaccard_index(const hll::hllbase_t<HashStruct> &hll, const std::array<uint32_t, 64> *a=nullptr) const {
        auto lq = query(hll, a);
        return lq[2] / (lq[0] + lq[1] + lq[2]);
    }
    double containment_index(const hll::hllbase_t<HashStruct> &hll, const std::array<uint32_t, 64> *a=nullptr) const {
        auto lq = query(hll, a);
        return lq[2] / (lq[0] + lq[2]);
    }

};

template<typename Container, typename HashStruct>
inline std::array<double, 3> pair_query(const Container &con, const hll::hllbase_t<HashStruct> &hll, const std::array<uint32_t, 64> *a=nullptr) {
    std::array<uint32_t, 64> *tmp = a ? nullptr: reinterpret_cast<std::array<uint32_t, 64> *>(__builtin_alloca(sizeof(*a)));
    if(!a) {
        *tmp = hll::detail::sum_counts(hll.core());
    }
    const std::array<uint32_t, 64> &osum = a ? *a: *tmp;
    std::array<uint32_t, 64> usum = osum;
    std::array<uint32_t, 64> lsum{0};
    const auto p = hll.p();
    lsum[0] = (1ul << hll.p()) - con.size();
    auto hcore = hll.core();
    for(const auto &pair: con) {
        const auto oval = hcore[pair.second];
        if(hcore[pair.first] < pair.second)
            --usum[oval], ++usum[pair.second];
        ++lsum[pair.second];
    }
    double myrep = hll::detail::ertl_ml_estimate(lsum, p, 64 - p),
            orep = hll.creport(),
              us = hll::detail::ertl_ml_estimate(usum, p, 64 - p),
              is = myrep + orep - us;
    return std::array<double, 3>{myrep - is, orep - is, is};
}

} // sparse

} // sketch
