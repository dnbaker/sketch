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
    static constexpr size_t SHIFT = 6;
    static constexpr size_t MASK = (1 << SHIFT) - 1;
    static constexpr size_t max_p() {return 32 - SHIFT;}
    static constexpr uint32_t get_index(uint32_t val) {
        return val >> SHIFT;
    }
    static constexpr uint8_t get_value(uint32_t val) {return static_cast<uint8_t>(val & MASK);} // Implicitly truncated, but we add a cast for clarity
    static constexpr uint32_t encode_value(uint32_t index, uint8_t val) {
        return (index << SHIFT) | val;
    }
};

template<typename HashStruct=hash::WangHash>
class SparseHLL {
    int p_;
    std::vector<uint32_t> vals_;
    std::unique_ptr<std::array<uint32_t, 64>> sum_;
    double v_ = -1.;
public:
    bool operator==(const SparseHLL &o) const {
        return vals_ == o.vals_;
    }
    SparseHLL(SparseHLL &&o) = default;
    SparseHLL &operator=(SparseHLL &&o) = default;
    SparseHLL &operator=(const SparseHLL &o) {
        p_ = o.p_; vals_ = o.vals_; v_ = o.v_;
        if(o.sum_) {
            if(!sum_)
                sum_.reset(new std::array<uint32_t, 64>);
            *sum_ = o.sum_;
        } else if(sum_) sum_.reset(nullptr);
    }
    SparseHLL(const SparseHLL &o): p_(o.p_), vals_(o.vals_), v_(o.v_) {
#if VERBOSE_AF
        std::fprintf(stderr, "Copy constructor\n");
#endif
        if(o.sum_) {
            sum_.reset(new std::array<uint32_t, 64>);
            *sum_ = *o.sum_;
        }
    }
    SparseHLL(const hll::hllbase_t<HashStruct> &hll): SparseHLL(hll.p()) {
#if VERBOSE_AF
        std::fprintf(stderr, "HLL constructor\n");
#endif
        for(size_t i = 0; i < hll.core().size(); ++i) {
            if(hll.core()[i]) {
                vals_.emplace_back(SparseHLL32::encode_value(i, hll.core()[i]));
            }
        }
    }
    SparseHLL(uint32_t p): p_(p) {
        if(p > SparseHLL32::max_p()) throw std::runtime_error(std::string("p exceeds maximum ") + std::to_string(SparseHLL32::max_p()));
#if VERBOSE_AF
        std::fprintf(stderr, "p constructor p: %u\n", p_);
#endif
    }
    void sort() {
        common::sort::default_sort(vals_.begin(), vals_.end(), [](auto x, auto y) {return SparseHLL32::get_index(x) < SparseHLL32::get_index(y);});
    }
    bool is_sorted() const {
        for(size_t i = 0; i < (vals_.size() ? vals_.size() - 1: 0); ++i) {
            if(SparseHLL32::get_index(vals_[i]) >= SparseHLL32::get_index(vals_[i + 1])) return false;
        }
        return true;
    }
    unsigned q() const {return 64 - p_;}
    std::array<uint32_t, 64> sum() const {
        assert(is_sorted());
        std::array<uint32_t, 64> ret{0};
        for(const auto v: vals_)
            ++ret[SparseHLL32::get_value(v)];
        ret[0] = (1ull << p_) - vals_.size();
        return ret;
    }
    std::array<uint32_t, 64> &count_sum() {
        if(!sum_) {
            assert(is_sorted());
            sum_.reset(new std::array<uint32_t, 64>);
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
        std::array<uint32_t, 64> *tmp = a ? nullptr: reinterpret_cast<std::array<uint32_t, 64> *>(__builtin_alloca(sizeof(*a))),
                                 *tmp2 = sum_ ? sum_.get(): reinterpret_cast<std::array<uint32_t, 64> *>(__builtin_alloca(sizeof(*tmp2)));
        if(!a) {
            *tmp = hll::detail::sum_counts(hll.core());
            assert(std::accumulate(tmp->begin(), tmp->end(), uint32_t(0), std::plus<>()) == size_t(1) << p_);
        }
        const std::array<uint32_t, 64> &osum = a ? *a: *tmp;
        assert(is_sorted());
        if(!sum_) *tmp2 = sum();
        std::array<uint32_t, 64> &lsum = sum_ ? *sum_: *tmp2;
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
        assert(std::accumulate(osum.begin(), osum.end(), 0u, std::plus<>()) == size_t(1) << p_);
        assert(std::accumulate(lsum.begin(), lsum.end(), 0u, std::plus<>()) == size_t(1) << p_);
        assert(std::accumulate(usum.begin(), usum.end(), 0u, std::plus<>()) == size_t(1) << p_);
        std::array<double, 3> ret;
        double myrep = this->report(), orep = hll.creport(), us = hll::detail::ertl_ml_estimate(usum, p_, q());
        double is = myrep + orep - us;
        ret[0] = std::max(myrep - is, 0.);
        ret[1] = std::max(orep - is, 0.);
        ret[2] = std::max(is, 0.);
#if VERBOSE_AF
        std::fprintf(stderr, "is: %lf. my size %lf o size %lf\n", ret[0], ret[1], ret[2]);
#endif
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
    template<typename Allocator>
    SparseHLL(const std::vector<uint32_t, Allocator> &a) {
        assert(vals_.size() == 0);
        const size_t nelem = a.size(), nb = sizeof(uint32_t) * nelem;
        auto buf = static_cast<uint32_t *>(std::malloc(nb));
        if(!buf) throw std::bad_alloc();
        std::memcpy(buf, a.data(), nb);
        if(nelem > 64)
            sort::default_sort(buf, buf + nelem);
        else
            sort::insertion_sort(buf, buf + nelem);
        assert(std::is_sorted(buf, buf + nelem));
        size_t ind = 0;
        while(ind < nelem) {
            if(ind < nelem - 1) {
                while(ind < nelem - 1 && ((buf[ind]>>6) == (buf[ind + 1]>>6))) {
#if VERBOSE_AF
                    std::fprintf(stderr, "same index: %u\n", SparseHLL32::get_index(buf[ind]));
#endif
                    ++ind;
                }
#if VERBOSE_AF
                std::fprintf(stderr, "saving for index: %u, has value %u and fully encoded value %u\n", SparseHLL32::get_index(buf[ind]), unsigned(SparseHLL32::get_value(buf[ind])), buf[ind]);
#endif
                assert(vals_.empty() || buf[ind] > vals_.back() || std::fprintf(stderr, "ind is %u\n", unsigned(ind)) == 0);
                vals_.push_back(buf[ind++]);
            } else {
#if VERBOSE_AF
                std::fprintf(stderr, "ind is nelem - 1, and saving for index: %u, has value %u and fully encoded value %u\n", SparseHLL32::get_index(buf[ind]), unsigned(SparseHLL32::get_value(buf[ind])), buf[ind]);
#endif
                vals_.push_back(buf[ind++]);
            }
        }
#if !NDEBUG
        std::set<uint32_t> vs(vals_.begin(), vals_.end());
        assert(vs.size() == vals_.size());
        assert(std::equal(vs.begin(), vs.end(), vals_.begin()));
        vs.clear();
        for(const auto v: vals_) vs.insert(SparseHLL32::get_index(v));
        assert(vs.size() == vals_.size());
#endif
        assert(is_sorted());
        std::free(buf);
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
#if !NDEBUG
    for(const auto &sum: {usum, osum, lsum}) {
        assert(std::accumulate(sum.begin(), sum.end(), size_t(0)) == size_t(1) << p);
    }
#endif
    double myrep = hll::detail::ertl_ml_estimate(lsum, p, 64 - p),
            orep = hll.creport(),
              us = hll::detail::ertl_ml_estimate(usum, p, 64 - p),
              is = myrep + orep - us;
    return std::array<double, 3>{std::max(myrep - is, 0.), std::max(orep - is, 0.), std::max(is, 0.)};
}

template<typename Container, typename HashStruct>
inline std::array<double, 3> pair_query(const Container &con, const Container &c2, const int p) {
    std::array<uint32_t, 64> lsum{0}, rsum{0}, usum{0};
    lsum[0] = (1ul << p) - con.size();
    rsum[0] = (1ul << p) - c2.size();
    auto il = con.begin(), ir = c2.begin(), el = con.end(), er = c2.end();
    while(il != el && ir != er) {
        if(il->first > ir->first) {
            ++rsum[ir->second];
            ++usum[ir->second];
            ++ir;
        } else if(ir->first > il->first) {
            ++rsum[il->second];
            ++usum[il->second];
            ++il;
        } else {
            ++lsum[il->second];
            ++rsum[ir->second];
            ++usum[std::max(il->second, ir->second)];
            ++ir; ++il;
        }
    }
    while(il != el) {
        ++usum[il->second];
        ++lsum[il->second];
    }
    while(ir != er) {
        ++usum[ir->second];
        ++rsum[ir->second];
    }
    double myrep = hll::detail::ertl_ml_estimate(lsum, p, 64 - p),
            orep = hll::detail::ertl_ml_estimate(rsum, p, 64 - p),
              us = hll::detail::ertl_ml_estimate(usum, p, 64 - p),
              is = myrep + orep - us;
    return std::array<double, 3>{std::max(myrep - is, 0.), std::max(orep - is, 0.), std::max(is, 0.)};
}
template<typename Allocator>
void flatten(std::vector<uint32_t, Allocator> &a) {
    size_t nfilled = 0;
    const size_t nelem = a.size();
    if(nelem > 64)
        sort::default_sort(a.data(), a.data() + nelem);
    else
        sort::insertion_sort(a.data(), a.data() + nelem);
    assert(std::is_sorted(a.data(), a.data() + nelem));
    size_t ind = 0;
    while(ind < nelem) {
        if(ind != nelem - 1) {
            while(ind < nelem - 1 && ((a[ind]>>6) == (a[ind + 1]>>6))) {
                ++ind;
            }
            a[nfilled++] = a[ind++];
        } else {
            a[nfilled++] = a[ind++];
        }
    }
    a.resize(nfilled);
    assert(std::is_sorted(a.begin(), a.end()));
}
template<typename Allocator, typename Functor>
void flattened_for_each(std::vector<uint32_t, Allocator> &a, const Functor &func) {
    const size_t nelem = a.size();
    if(nelem > 64)
        sort::default_sort(a.data(), a.data() + nelem);
    else
        sort::insertion_sort(a.data(), a.data() + nelem);
    assert(std::is_sorted(a.data(), a.data() + nelem));
    size_t ind = 0;
    while(ind < nelem) {
        if(ind != nelem - 1) {
            while(ind != nelem - 1 && ((a[ind]>>6) == (a[ind + 1]>>6))) {
                ++ind;
            }
            func(a[ind++]);
        } else {
            func(a[ind++]);
        }
    }
}

template<typename Allocator, typename HashStruct>
inline std::array<double, 3> flatten_and_query(std::vector<uint32_t, Allocator> &con, const hll::hllbase_t<HashStruct> &hll, const std::array<uint32_t, 64> *a=nullptr, bool is_already_flattened=false) {
    std::array<uint32_t, 64> *tmp = a ? nullptr: reinterpret_cast<std::array<uint32_t, 64> *>(__builtin_alloca(sizeof(*a)));
    if(!a) {
        *tmp = hll::detail::sum_counts(hll.core());
    }
    const std::array<uint32_t, 64> &osum = a ? *a: *tmp;
    std::array<uint32_t, 64> usum = osum;
    std::array<uint32_t, 64> lsum{0};
    const auto p = hll.p();
    if(!is_already_flattened) flatten(con);
    assert(con.size() <= 1ul << p);
    lsum[0] = (1ul << p) - con.size();
    auto hcore = hll.core();
    assert(std::is_sorted(con.begin(), con.end()));
#if VERBOSE_AF
    std::fprintf(stderr, "FLATTENED IS THE END. size now: %zu\n", con.size());
#endif
    for(const auto v: con) {
        const auto first = SparseHLL32::get_index(v);
        const auto second = SparseHLL32::get_value(v);
        assert(first < hcore.size());
        const auto oval = hcore[first];
        if(hcore[first] < second)
            --usum[oval], ++usum[second];
        ++lsum[second];
    }
#define __sum_check(x) \
    assert(std::accumulate(std::begin(x), std::end(x), size_t(0)) == size_t(1) << p);
    __sum_check(osum);
    __sum_check(lsum);
    __sum_check(usum);
#undef __sum_check
    double myrep = hll::detail::ertl_ml_estimate(lsum, p, 64 - p),
            orep = hll::detail::ertl_ml_estimate(osum, p, 64 - p),
              us = hll::detail::ertl_ml_estimate(usum, p, 64 - p),
              is = myrep + orep - us;
    return std::array<double, 3>{std::max(myrep - is, 0.), std::max(orep - is, 0.), std::max(is, 0.)};
}

} // sparse

} // sketch
