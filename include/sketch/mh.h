#pragma once
#include <mutex>
//#include <queue>
#include "sketch/hll.h" // For common.h and clz functions
#include "sketch/fixed_vector.h"
#include <unordered_map>
#include "sketch/isz.h"
#include <queue>
#include "flat_hash_map/flat_hash_map.hpp"
#include "sketch/hash.h"
#include "vec/vec.h"


/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  CountingRangeMinHash (bottom-k, but with weights)
 *                  ShrivastavaHash
 */

namespace sketch {

#ifndef VEC_DISABLED__
using Space = vec::SIMDTypes<uint64_t>;
#endif

using hash::WangHash;
inline namespace minhash {

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

template<typename T, typename Cmp=std::greater<T>>
class AbstractMinHash {
protected:
    uint64_t ss_;
    AbstractMinHash(uint64_t sketch_size): ss_(sketch_size) {}
    auto &sketch();
    const auto &sketch() const;
    auto nvals() const {return ss_;}
    std::vector<T, common::Allocator<T>> sketch2vec() const {
        std::vector<T, common::Allocator<T>> ret;
        auto &s = sketch();
        ret.reserve(s.size());
        for(const auto el: this->sketch()) ret.emplace_back(el);
        return ret;
    }
public:
    uint64_t sketch_size() const {
        return ss_;
    }
};




template<typename T, typename Cmp=std::greater<T>, typename Hasher=WangHash>
class KMinHash: public AbstractMinHash<T, Cmp> {
    std::vector<uint64_t, common::Allocator<uint64_t>> seeds_;
    std::vector<T, common::Allocator<uint64_t>> hashes_;
    Hasher hf_;
    // Uses k hash functions with seeds.
    // TODO: reuse code from hll/bf for vectorized hashing.
public:
    KMinHash(size_t nkeys, size_t sketch_size, uint64_t seedseed=137, Hasher &&hf=Hasher()):
        AbstractMinHash<T, Cmp>(sketch_size),
        hashes_(nkeys),
        hf_(std::move(hf))
    {
        DefaultRNGType gen(seedseed);
        seeds_.reserve(nseeds());
        while(seeds_.size() < nseeds()) seeds_.emplace_back(gen());
        throw NotImplementedError();
    }
    size_t nseeds() const {return this->nvals() * sizeof(uint64_t) / sizeof(T);}
    size_t nhashes_per_seed() const {return sizeof(uint64_t) / sizeof(T);}
#ifndef VEC_DISABLED__
    size_t nhashes_per_vector() const {return sizeof(uint64_t) / sizeof(Space::Type);}
#endif
    void addh(T val) {
        throw NotImplementedError();
        // Hash stuff.
        // Then use a vectorized minimization comparison instruction
    }
    SET_SKETCH(hashes_)
};

template<typename T, typename Allocator> struct FinalRMinHash; // Forward definition
template<typename HashStruct=WangHash, typename VT=uint64_t, bool select_bottom=true> struct BottomKHasher;
/*
The sketch is the set of minimizers.

*/
template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=WangHash,
         typename Allocator=common::Allocator<T>
        >
struct RangeMinHash: public AbstractMinHash<T, Cmp> {
protected:
    Hasher hf_;
    Cmp cmp_;
    std::set<T, Cmp> minimizers_; // using std::greater<T> so that we can erase from begin()

public:
    using final_type = FinalRMinHash<T, Allocator>;
    using Compare = Cmp;
    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()):
        AbstractMinHash<T, Cmp>(sketch_size), hf_(std::move(hf)), cmp_(std::move(cmp))
    {
    }
    void show() {
        std::fprintf(stderr, "%zu mins\n", minimizers_.size());
        for(const auto x: minimizers_) {
            std::fprintf(stderr, "%zu\n", size_t(x));
        }
    }
    RangeMinHash(std::string) {throw NotImplementedError("");}
    double cardinality_estimate() const {
        const double result = (std::numeric_limits<T>::max()) / this->max_element() * minimizers_.size();
        return result;
    }
    RangeMinHash(gzFile fp) {
        if(!fp) throw std::runtime_error("Null file handle!");
        this->read(fp);
    }
    DBSKETCH_READ_STRING_MACROS
    DBSKETCH_WRITE_STRING_MACROS
    ssize_t read(gzFile fp) {
        ssize_t ret = gzread(fp, this, sizeof(*this));
        T v;
        for(ssize_t read; (read = gzread(fp, &v, sizeof(v))) == sizeof(v);minimizers_.insert(v), ret += read);
        return ret;
    }
    RangeMinHash &operator+=(const RangeMinHash &o) {
        minimizers_.insert(o.begin(), o.end());
        while(minimizers_.size() > this->ss_) {
            minimizers_.erase(minimizers_.begin());
        }
        return *this;
    }
    RangeMinHash operator+(const RangeMinHash &o) const {
        RangeMinHash ret(*this);
        ret += o;
        return ret;
    }
    ssize_t write(gzFile fp) const {
        if(!fp) throw std::runtime_error("Null file handle!");
        char tmp[sizeof(*this)];
        std::memcpy(tmp, this, sizeof(*this));
        // Hacky, non-standard-layout-type alternative to offsetof
        std::memset(tmp + (reinterpret_cast<const char *>(&minimizers_) - reinterpret_cast<const char *>(this)), 0, sizeof(minimizers_));
        ssize_t ret = gzwrite(fp, tmp, sizeof(tmp));
        for(const auto v: minimizers_)
            ret += gzwrite(fp, &v, sizeof(v));
        return ret;
    }
    auto rbegin() const {return minimizers_.rbegin();}
    auto rbegin() {return minimizers_.rbegin();}
    T max_element() const {
#if 0
        for(const auto e: *this)
            assert(*begin() >= e);
#endif
        return *begin();
    }
    T min_element() const {
        return *rbegin();
    }
    INLINE void addh(T val) {
        add(hf_(val));
    }
    INLINE void add(T val) {
        if(minimizers_.size() == this->ss_) {
            if(cmp_(max_element(), val)) {
                minimizers_.insert(val);
                if(minimizers_.size() > this->ss_)
                    minimizers_.erase(minimizers_.begin());
            }
        } else minimizers_.insert(val);
    }
    template<typename T2>
    INLINE void addh(T2 val) {
        val = hf_(val);
        CONST_IF(std::is_same<T2, T>::value) {
            add(val);
        } else {
            static_assert(sizeof(val) % sizeof(T) == 0, "val must be the same size or greater than inserted element.");
            T *ptr = reinterpret_cast<T *>(&val);
            for(unsigned i = 0; i < sizeof(val) / sizeof(T); add(ptr[i++]));
        }
    }
    auto begin() {return minimizers_.begin();}
    auto begin() const {return minimizers_.begin();}
    auto end() {return minimizers_.end();}
    auto end() const {return minimizers_.end();}
    template<typename C2>
    size_t intersection_size(const C2 &o) const {
        return isz::intersection_size(o, *this, cmp_);
    }
    double jaccard_index(const RangeMinHash &o) const {
        //assert(o.size() == minimizers_.size());
        auto lit = minimizers_.rbegin(), rit = o.minimizers_.rbegin(), lend = minimizers_.rend(), rend = o.minimizers_.rend();
        const size_t n = minimizers_.size();
        size_t nused = 0, shared = 0;
        while(nused++ < n) {
            if(*lit == *rit) {
                ++shared;
                if(++lit == lend) break;
                if(++rit == rend) break;
            } else {
                if(!cmp_(*lit, *rit)) {
                    if(++lit == lend) break;
                } else {
                    if(++rit == rend) break;
                }
            }
        }
        return double(shared) / n;
    }
    template<typename Container>
    Container to_container() const {
        if(this->ss_ != size()) // If the sketch isn't full, add max to the end until it is.
            minimizers_.resize(this->ss_, std::numeric_limits<T>::max());
        return Container(std::rbegin(minimizers_), std::rend(minimizers_.end()));
    }
    void clear() {
        decltype(minimizers_)().swap(minimizers_);
    }
    void free() {clear();}
    final_type cfinalize() const {
        std::vector<T> reta(minimizers_.begin(), minimizers_.end());
        return final_type(std::move(reta));
    }
    final_type finalize() & {
        return cfinalize();
    }
    final_type finalize() const & {
        return cfinalize();
    }
    final_type finalize() && {
        return static_cast<const RangeMinHash &&>(*this).finalize();
    }
    final_type finalize() const && {
        std::vector<T> reta(minimizers_.begin(), minimizers_.end());
        if(reta.size() < this->ss_) {
            reta.insert(reta.end(), this->ss_ - reta.size(), std::numeric_limits<uint64_t>::max());
        }
        final_type ret(std::move(reta));
        return ret;
    }
    std::vector<T> mh2vec() const {return to_container<std::vector<T>>();}
    size_t size() const {return minimizers_.size();}
    using key_compare = typename decltype(minimizers_)::key_compare;
    SET_SKETCH(minimizers_)
};

namespace weight {


struct EqualWeight {
    // Do you even weight bro?
    template<typename T>
    constexpr double operator()(T &x) const {return 1.;}
};

template<typename ScorerType>
struct DictWeight {
    const ScorerType &m_;
    DictWeight(const ScorerType &m): m_(m) {}
    template<typename T>
    double operator()(T &x) const {
        return m_.score(x);
    }
};
}

template<typename T, typename Allocator=std::allocator<T>>
struct FinalRMinHash {
    static_assert(std::is_unsigned<T>::value, "must be unsigned btw");

    using allocator = Allocator;
    std::vector<T, Allocator> first;
    uint64_t lsum_;
    using container_type = decltype(first);
    size_t intersection_size(const FinalRMinHash &o) const {
        return isz::intersection_size(first, o.first);
    }
    double jaccard_index(const FinalRMinHash &o) const {
        auto lit = begin(), rit = o.begin(), lend = end(), rend = o.end();
        const size_t n = size();
        size_t nused = 0, shared = 0;
        while(nused++ < n) {
            if(*lit == *rit) {
                ++shared;
                if(++lit == lend) break;
                if(++rit == rend) break;
            } else {
                if(*lit < *rit) {
                    if(++lit == lend) break;
                } else {
                    if(++rit == rend) break;
                }
            }
        }
        return double(shared) / n;
        //double is = intersection_size(o);
        //return is / ((size() << 1) - is);
    }
    FinalRMinHash &operator+=(const FinalRMinHash &o) {
        std::vector<T, Allocator> newfirst; newfirst.reserve(o.size());
        if(this->size() != o.size()) throw std::runtime_error("Non-matching parameters for FinalRMinHash comparison");
        auto i1 = this->begin(), i2 = o.begin();
        while(newfirst.size() < first.size()) {
            if(*i2 < *i1) {
                newfirst.push_back(*i2++);
            } else if(*i1 < *i2) {
                newfirst.push_back(*i1++);
            } else newfirst.push_back(*i1), ++i1, ++i2;
        }
        std::swap(newfirst, first);
        return *this;
    }
    FinalRMinHash operator+(const FinalRMinHash &o) const {
        auto tmp = *this;
        tmp += o;
        return tmp;
    }
    /*
    double union_size(const FinalRMinHash &o) const {
        std::vector<T> total(o.begin(), o.end());
        total.insert(total.end(), begin(), end());
        std::sort(total.begin(), total.end());
        total.resize(std::min(o.size(), size()));
        const size_t maxv = total.back();
        return (double(std::numeric_limits<T>::max()) / maxv) * this->size();
        PREC_REQ(this->size() == o.size(), "Non-matching parameters for FinalRMinHash comparison");
        size_t n_in_sketch = 0;
        auto i1 = this->rbegin(), i2 = o.rbegin();
        while(n_in_sketch < first.size() - 1) {
            // Easier to branch-predict:  http://www.vldb.org/pvldb/vol8/p293-inoue.pdf
            if(*i1 == *i2) {
                ++i1, ++i2;
            } else if(*i1 < *i2) {
                ++i1;
            } else {
                ++i2;
            }
            ++n_in_sketch;
        }
        // TODO: test after refactoring
        assert(i1 < this->rend());
        const size_t mv = std::min(*i1, *i2);
        const double est = double(std::numeric_limits<T>::max()) / mv * this->size();
        std::fprintf(stderr, "mv: %zu. est: %g. Expected maxv %zu\n", size_t(mv), est, maxv);
        return est;
    }
    */
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        // KMV (kth-minimum value) estimate
        return (static_cast<double>(std::numeric_limits<T>::max()) / double(this->max_element()) * first.size());
    }
    void sum() const {
        lsum_ = std::accumulate(this->first.begin(), this->first.end(), size_t(0));
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalRMinHash &o, const WeightFn &fn=WeightFn()) const {
        const size_t lsz = size();
        const size_t rsz = o.size();
        double num = 0, denom = 0;
        size_t i1 = 0, i2 = 0;
        size_t nused = 0;
        while(nused++ < lsz) {
            if(first[i1] < o.first[i2]) {
                denom += fn(first[i1]);
                if(++i1 == lsz) break;
            } else if(o.first[i2] < first[i1]) {
                denom += fn(o.first[i2]);
                assert(i2 != rsz);
                if(++i2 == rsz) break;
            } else {
                const auto v1 = fn(first[i1]), v2 = fn(o.first[i2]);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                assert(i2 != rsz);
                ++i1; ++i2;
                if(i1 == lsz || i2 == rsz) break;
            }
        }
        return num / denom;
    }
    typename container_type::const_iterator begin() {
        return first.cbegin();
    }
    typename container_type::const_iterator end() {
        return first.cend();
    }
    typename container_type::const_iterator begin() const {
        return first.cbegin();
    }
    typename container_type::const_iterator end() const {
        return first.cend();
    }
    typename container_type::const_reverse_iterator rbegin() {
        return first.crbegin();
    }
    typename container_type::const_reverse_iterator rend() {
        return first.crend();
    }
    typename container_type::const_reverse_iterator rbegin() const {
        return first.crbegin();
    }
    typename container_type::const_reverse_iterator rend() const {
        return first.crend();
    }
    void free() {
        decltype(first) tmp; std::swap(tmp, first);
    }
    template<typename It>
    FinalRMinHash(It start, It end) {
        std::copy(start, end, std::back_inserter(first));
        sort();
    }
    template<typename Alloc>
    FinalRMinHash(const std::vector<T, Alloc> &ofirst): FinalRMinHash(ofirst.begin(), ofirst.end()) {}
    template<typename Hasher, bool is_bottom>
    FinalRMinHash(const BottomKHasher<Hasher, T, is_bottom> &bk): FinalRMinHash(bk.mpq_.getq().begin(), bk.mpq_.getq().end()) {}
    FinalRMinHash(FinalRMinHash &&o) = default;
    ssize_t read(gzFile fp) {
        uint64_t sz;
        if(gzread(fp, &sz, sizeof(sz)) != sizeof(sz)) throw ZlibError("Failed to read");
        first.resize(sz);
        ssize_t ret = sizeof(sz), nb = first.size() * sizeof(first[0]);
        if(gzread(fp, first.data(), nb) != nb) throw ZlibError("Failed to read");
        ret += nb;
        sort();
        return ret;
    }
    ssize_t write(gzFile fp) const {
        uint64_t sz = this->size();
        ssize_t ret = gzwrite(fp, &sz, sizeof(sz));
        ret += gzwrite(fp, first.data(), first.size() * sizeof(first[0]));
        return ret;
    }
    DBSKETCH_READ_STRING_MACROS
    DBSKETCH_WRITE_STRING_MACROS
    auto max_element() const {
        const auto ret = first.back();
        assert(std::accumulate(first.begin(), first.end(), true, [&](bool t, auto v) {return t && ret >= v;}));
        return ret;
    }
    template<typename Hasher, bool ismax>
    FinalRMinHash(BottomKHasher<Hasher, T, ismax> &&prefinal): FinalRMinHash(std::move(prefinal.mpq_.getq())) {
        prefinal.clear();
    }
    template<typename Hasher, typename Cmp>
    FinalRMinHash(RangeMinHash<T, Cmp, Hasher> &&prefinal): FinalRMinHash(std::move(prefinal.finalize())) {
        prefinal.clear();
    }
    FinalRMinHash(const std::string &s): FinalRMinHash(s.data()) {}
    FinalRMinHash(gzFile fp) {read(fp);}
    FinalRMinHash(const char *infname) {read(infname);}
    FinalRMinHash(const FinalRMinHash &o): first(o.first) {sort();}
    size_t size() const {return first.size();}
protected:
    FinalRMinHash() {}
    FinalRMinHash &operator=(const FinalRMinHash &o) = default;
    void sort() {
        std::sort(this->first.begin(), this->first.end());
    }
};


template<typename T, typename CountType> struct FinalCRMinHash; // Forward


template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=WangHash,
         typename CountType=uint32_t
        >
class CountingRangeMinHash: public AbstractMinHash<T, Cmp> {
    static_assert(std::is_arithmetic<CountType>::value, "CountType must be arithmetic");
    struct VType {
        T first;
        mutable CountType second;
        inline bool operator<(const VType &b) const {
            return Cmp()(this->first , b.first);
        }
        inline bool operator>(const VType &b) const {
            return !Cmp()(this->first , b.first);
        }
        inline bool operator==(const VType &b) const {
            return this->first == b.first;
        }
        VType(T v, CountType c): first(v), second(c) {}
        VType(const VType &o): first(o.first), second(o.second) {}
        VType &operator=(const VType &o) {
            this->first = o.first;
            this->second = o.second;
        }
        VType(gzFile fp) {if(gzread(fp, this, sizeof(*this)) != sizeof(*this)) throw ZlibError("Failed to read");}
    };
    Hasher hf_;
    Cmp cmp_;
    mutable CountType cached_sum_sq_ = 0, cached_sum_ = 0;
    std::set<VType> minimizers_; // using std::greater<T> so that we can erase from begin()
public:
    const auto &min() const {return minimizers_;}
    using size_type = CountType;
    using final_type = FinalCRMinHash<T, CountType>;
    auto size() const {return minimizers_.size();}
    auto begin() {return minimizers_.begin();}
    auto begin() const {return minimizers_.begin();}
    auto rbegin() {return minimizers_.rbegin();}
    auto rbegin() const {return minimizers_.rbegin();}
    auto end() {return minimizers_.end();}
    auto end() const {return minimizers_.end();}
    auto rend() {return minimizers_.rend();}
    auto rend() const {return minimizers_.rend();}
    void free() {
        std::set<VType> tmp;
        std::swap(tmp, minimizers_);
    }
    CountingRangeMinHash(size_t n, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()): AbstractMinHash<T, Cmp>(n), hf_(std::move(hf)), cmp_(std::move(cmp)) {}
    CountingRangeMinHash(std::string s): CountingRangeMinHash(0) {throw NotImplementedError("");}
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        return double(std::numeric_limits<T>::max()) / std::max_element(minimizers_.begin(), minimizers_.end(), [](auto x, auto y) {return x.first < y.first;})->first * minimizers_.size();
    }
    INLINE void add(T val) {
        if(minimizers_.size() == this->ss_) {
            if(cmp_(begin()->first, val)) {
                auto it = minimizers_.find(VType(val, 0));
                if(it == minimizers_.end()) {
                    minimizers_.erase(begin());
                    minimizers_.insert(VType(val, CountType(1)));
                } else ++it->second;
            }
        } else minimizers_.insert(VType(val, CountType(1)));
    }
    INLINE void addh(T val) {
        val = hf_(val);
        this->add(val);
    }
    auto max_element() const {
        return minimizers_.begin()->first;
    }
    auto sum_sq() {
        if(cached_sum_sq_) goto end;
        cached_sum_sq_ = std::accumulate(this->begin(), this->end(), 0., [](auto s, const VType &x) {return s + x.second * x.second;});
        end:
        return cached_sum_sq_;
    }
    auto sum_sq() const {return cached_sum_sq_;}
    auto sum() {
        if(cached_sum_) goto end;
        cached_sum_ = std::accumulate(std::next(this->begin()), this->end(), this->begin()->second, [](auto s, const VType &x) {return s + x.second;});
        end:
        return cached_sum_;
    }
    auto sum() const {return cached_sum_;}
    double histogram_intersection(const CountingRangeMinHash &o) const {
        assert(o.size() == size());
        size_t denom = 0, num = 0;
        auto i1 = minimizers_.rbegin(), i2 = o.minimizers_.rbegin();
        auto e1 = minimizers_.rend(), e2 = o.minimizers_.rend();
        size_t nused = 0;
        const size_t sz = size();
        for(;nused++ < sz;) {
            if(!cmp_(i1->first, i2->first)) {
                denom += (i1++)->second;
                if(i1 == e1) break;
            } else if(!cmp_(i2->first, i1->first)) {
                denom += (i2++)->second;
                if(i2 == e2) break;
            } else {
                const auto v1 = i1->second, v2 = i2->second;
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                ++i1; ++i2;
                if(i1 == e1) break;
                if(i2 == e2) break;
            }
        }
        //while(i1 != e1) denom += i1++->second;
        //while(i2 != e2) denom += i2++->second;
        return static_cast<double>(num) / denom;
    }
    double containment_index(const CountingRangeMinHash &o) const {
        assert(o.size() == size());
        size_t denom = 0, num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin(),
             e1 = minimizers_.end(),   e2 = o.minimizers_.end();
        const size_t nelem = size();
        size_t nused = 0;
        while(nused++ < nelem) {
            if(cmp_(i1->first, i2->first)) {
                denom += i1->second;
                if(++i1 == e1) break;
            } else if(cmp_(i2->first, i1->first)) {
                if(++i2 == e2) break;
            } else {
                const auto v1 = i1->second, v2 = i2->second;
                denom += v1;
                num += std::min(v1, v2);
                if(++i1 == e1) break;
                if(++i2 == e2) break;
            }
        }
        return static_cast<double>(num) / denom;
    }
    DBSKETCH_WRITE_STRING_MACROS
    DBSKETCH_READ_STRING_MACROS
    ssize_t write(gzFile fp) const {
        uint64_t n = minimizers_.size();
        if(gzwrite(fp, &n, sizeof(n)) != sizeof(n)) throw ZlibError("Failed to write");
        for(const auto &pair: minimizers_) {
            if(gzwrite(fp, std::addressof(pair), sizeof(pair)) != sizeof(pair))
                ZlibError("Failed to write");
        }
        return sizeof(VType) * minimizers_.size();
    }
    ssize_t read(gzFile fp) {
        uint64_t n;
        if(gzread(fp, &n, sizeof(n)) != sizeof(n)) throw ZlibError("Failed to read");
        for(size_t i = n; i--; minimizers_.insert(VType(fp)));
        return sizeof(n) + sizeof(VType) * n;
    }

    void clear() {
        decltype(minimizers_) tmp;
        std::swap(tmp, minimizers_);
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const CountingRangeMinHash &o, const WeightFn &fn) const {
        assert(o.size() == size());
        double denom = 0, num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
        const auto e1 = minimizers_.cend(), e2 = o.minimizers_.cend();
        const size_t nelem = size();
        for(size_t nused = 0;nused++ < nelem;) {
            auto &lhf = i1->first, &rhf = i2->first;
            if(cmp_(lhf, rhf)) {
                denom += (i1->second) * fn(lhf);
                if(++i1 == e1) break;
            } else if(cmp_(rhf, lhf)) {
                denom += (i2->second) * fn(rhf);
                if(++i2 == e2) break;
            } else {
                assert(rhf == lhf);
                const auto v1 = i1->second * fn(lhf), v2 = i2->second * fn(rhf);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                if(++i1 == e1 || ++i2 == e2) break;
            }
        }
        while(i1 < e1) denom += i1->second * fn(i1->first), ++i1;
        while(i2 < e2) denom += i2->second * fn(i2->first), ++i2;
        return static_cast<double>(num) / denom;
    }
    final_type finalize() & {
        return cfinalize();
    }
    final_type finalize() const & {
        return cfinalize();
    }
    final_type cfinalize() const & {
        return FinalCRMinHash<T, CountType>(*this);
    }
    final_type finalize() && {
        return static_cast<const CountingRangeMinHash &&>(*this).finalize();
    }
    final_type finalize() const && {
        auto ret(FinalCRMinHash<T, CountType>(std::move(*this)));
        return ret;
    }
    template<typename Func>
    void for_each(const Func &func) const {
        for(const auto &i: minimizers_) {
            func(i);
        }
    }
    template<typename Func>
    void for_each(const Func &func) {
        for(auto &i: minimizers_) {
            func(i);
        }
    }
    void print() const {
        for_each([](auto &p) {std::fprintf(stderr, "key %s with value %zu\n", std::to_string(p.first).data(), size_t(p.second));});
    }
    size_t union_size(const CountingRangeMinHash &o) const {
        size_t denom = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
        auto e1 = minimizers_.end(), e2 = o.minimizers_.end();
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                denom += i1->second;
                if(++i1 == e1) break;
            } else if(cmp_(i2->first, i1->first)) {
                denom += i2->second;
                if(++i2 == e2) break;
            } else {
                denom += std::max(i1->second, i2->second);
                ++i1; ++i2;
                if( (i1 == e1) || (i2 == e2)) break;
            }
        }
        return denom;
    }
    size_t intersection_size(const CountingRangeMinHash &o) const {
        size_t num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
        auto e1 = minimizers_.end(), e2 = o.minimizers_.end();
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                if(++i1 == e1) break;
            } else if(cmp_(i2->first, i1->first)) {
                if(++i2 == e2) break;
            } else {
                num += std::min(i1->second, i2->second);
                if(++i1 == e1) break;
                if(++i2 == e2) break;
            }
        }
        return num;
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        return histogram_intersection(o);
    }
};



template<typename T, typename CountType=uint32_t>
struct FinalCRMinHash: public FinalRMinHash<T> {
    using super = FinalRMinHash<T>;
    std::vector<CountType> second;
    uint64_t count_sum_;
    double count_sum_l2norm_;
    using count_type = CountType;
    using key_type = T;
    FinalCRMinHash(gzFile fp) {
        this->read(fp);
    }
    FinalCRMinHash(const std::string &path): FinalCRMinHash(path.data()) {}
    FinalCRMinHash(const char *path) {
        this->read(path);
    }
    void free() {
        super::free();
        std::vector<CountType> tmp;
        std::swap(tmp, second);
    }
    size_t countsum() const {return std::accumulate(second.begin(), second.end(), size_t(0), [](auto sz, auto sz2) {return sz += sz2;});}
    size_t countsumsq() const {
#if !NDEBUG
        size_t lsum = 0;
        for(const auto v: this->second) lsum += size_t(v) * v;

        size_t asum = std::accumulate(second.begin(), second.end(), size_t(0), [](auto sz, auto sz2) {return sz += size_t(sz2) * sz2;});
        assert(asum == lsum);
        return asum;
#else
        return std::accumulate(second.begin(), second.end(), size_t(0), [](auto sz, auto sz2) {return sz += size_t(sz2) * sz2;});
#endif
    }
    double cosine_distance(const FinalCRMinHash &o) const {
        throw NotImplementedError("This has not been implemented correctly.");
        const size_t lsz = this->size(), rsz = o.size();
        size_t num = 0;
        size_t nused = 0;
        for(size_t i1 = 0, i2 = 0;nused++ < lsz;) {
            if(this->first[i1] < o.first[i2]) {
                if(++i1 == lsz) break;
            } else if(o.first[i2] < this->first[i1]) {
                if(++i2 == rsz) break;
            } else {
                const auto v1 = o.second[i2], v2 = second[i1];
                num += v1 * v2;
                if(++i1 == lsz || ++i2 == rsz) break;
            }
        }
        return dot(o) / count_sum_l2norm_ / o.count_sum_l2norm_;
    }
    double dot(const FinalCRMinHash &o) const {
        const size_t lsz = this->size(), rsz = o.size();
        size_t num = 0;
        size_t nused = 0;
        for(size_t i1 = 0, i2 = 0;nused++ < lsz;) {
            if(this->first[i1] < o.first[i2]) {
                if(++i1 == lsz) break;
            } else if(o.first[i2] < this->first[i1]) {
                if(++i2 == rsz) break;
            } else {
                const auto v1 = o.second[i2], v2 = second[i1];
                num += v1 * v2;
                if(++i1 == lsz || ++i2 == rsz) break;
            }
        }
        return static_cast<double>(num);
    }
    double histogram_intersection(const FinalCRMinHash &o) const {
        const size_t lsz = this->size(), rsz = o.size();
        assert(std::accumulate(this->second.begin(), this->second.end(), size_t(0)) == this->count_sum_);
        assert(std::accumulate(o.second.begin(), o.second.end(), size_t(0)) == o.count_sum_);
        assert(std::sqrt(std::accumulate(this->second.begin(), this->second.end(), 0., [](auto psum, auto newv) {return psum + newv * newv;})) == this->count_sum_l2norm_);
        assert(std::sqrt(std::accumulate(o.second.begin(), o.second.end(), 0., [](auto psum, auto newv) {return psum + newv * newv;})) == o.count_sum_l2norm_);
        assert(count_sum_ > 0);
        assert(o.count_sum_ > 0);
        size_t num = 0;
        size_t i1 = 0, i2 = 0;
        size_t nused = 0;
        size_t denom = 0;
        while(nused++ < lsz) {
            auto &lhs = this->first[i1];
            auto &rhs = o.first[i2];
            if(lhs < rhs) {
                denom += this->second[i1++];
                if(i1 == lsz) break;
            } else if(rhs < lhs) {
                denom += this->second[i2++];
                if(i2 == lsz) break;
            } else {
                assert(!(lhs < rhs));
                assert(!(rhs < lhs));
                assert(lhs == rhs);
                auto lhv = this->second[i1++], rhv = o.second[i2++];
                num += std::min(lhv, rhv);
                if(i1 == lsz || i2 == rsz) break;
            }
        }
        return static_cast<double>(num) / denom;
    }
    DBSKETCH_READ_STRING_MACROS
    DBSKETCH_WRITE_STRING_MACROS
    ssize_t read(gzFile fp) {
        uint64_t nelem;
        ssize_t ret = gzread(fp, &nelem, sizeof(nelem));
        if(ret != sizeof(nelem)) throw ZlibError("Failed to read");
        this->first.resize(nelem);
        if((ret = gzread(fp, &count_sum_, sizeof(count_sum_))) != sizeof(count_sum_)) throw ZlibError("Failed to read");
        if((ret = gzread(fp, &count_sum_l2norm_, sizeof(count_sum_l2norm_))) != sizeof(count_sum_l2norm_)) throw ZlibError("Failed to read");
        if((ret = gzread(fp, this->first.data(), sizeof(this->first[0]) * nelem) != ssize_t(sizeof(this->first[0]) * nelem))) throw ZlibError("Failed to read");
        ret += gzread(fp, this->first.data(), sizeof(this->first[0]) * nelem);
        this->second.resize(nelem);
        ret += gzread(fp, this->second.data(), sizeof(this->second[0]) * nelem);
        assert(std::set<uint64_t>(this->first.begin(), this->first.end()).size() == this->first.size());
        prepare();
        return ret;
    }
    double union_size(const FinalCRMinHash &o) const {
        PREC_REQ(this->size() == o.size(), "mismatched parameters");
        return super::union_size(o);
    }
    ssize_t write(gzFile fp) const {
        assert(std::set<uint64_t>(this->first.begin(), this->first.end()).size() == this->first.size());
        uint64_t nelem = second.size();
        ssize_t ret = gzwrite(fp, &nelem, sizeof(nelem));
        ret += gzwrite(fp, &count_sum_, sizeof(count_sum_));
        ret += gzwrite(fp, &count_sum_l2norm_, sizeof(count_sum_l2norm_));
        ret += gzwrite(fp, this->first.data(), sizeof(this->first[0]) * nelem);
        ret += gzwrite(fp, this->second.data(), sizeof(this->second[0]) * nelem);
        return ret;
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalCRMinHash &o, const WeightFn &fn=WeightFn()) const {
        const size_t lsz = this->size();
        const size_t rsz = o.size();
        assert(rsz == o.second.size());
        assert(rsz == o.first.size());
        double denom = 0, num = 0;
        size_t i1 = 0, i2 = 0;
        size_t nused = 0;
        while(nused++ < lsz) {
            auto &lhs = this->first[i1];
            auto &rhs = o.first[i2];
            if(lhs < rhs) {
                denom += second[i1] * fn(lhs);
                if(++i1 == lsz) break;
            } else if(rhs < lhs) {
                denom += o.second[i1] * fn(rhs);
                if(++i2 == rsz) break;
            } else {
                auto tmpnum = std::min(second[i1], o.second[i2]);
                auto tmpden = std::max(second[i1], o.second[i2]);
                denom += fn(lhs) * tmpden;
                num   += fn(lhs) * tmpnum;
                ++i2, ++i1;
                if(i2 == rsz || i1 == lsz) break;
            }
            assert(i2 < o.second.size());
        }
        return num / denom;
    }
    void prepare(size_t ss=0) {
        const size_t fs = this->first.size();
        ss = ss ? ss: fs;
        fixed::vector<std::pair<key_type, count_type>> tmp(fs);
        for(size_t i = 0; i < fs; ++i)
            tmp[i] = std::make_pair(this->first[i], second[i]);
        auto tmpcmp = [](auto x, auto y) {return x.first < y.first;};
        common::sort::default_sort(tmp.begin(), tmp.end(), tmpcmp);
        for(size_t i = 0; i < this->first.size(); ++i)
            this->first[i] = tmp[i].first, this->second[i] = tmp[i].second;
        assert(std::is_sorted(this->first.begin(), this->first.end()));
        PREC_REQ(this->first.size() == this->second.size(), "Counts and hashes must have equal size");
        const std::ptrdiff_t diff = ss - this->first.size();
        if(diff > 0) {
            this->first.insert(this->first.end(), diff, std::numeric_limits<T>::max());
            this->second.insert(this->second.end(), diff, 0);
        } else if(diff < 0) {
            const auto fe = this->first.end();
            const auto se = this->second.end();
            assert(fe + diff < fe);
            assert(se + diff < se);
            this->first.erase(fe + diff, fe);
            this->second.erase(se + diff, se);
        }
        assert(this->first.size() == this->second.size());
        assert(this->first.size() == ss);
        count_sum_ = countsum();
        count_sum_l2norm_ = std::sqrt(countsumsq());
        POST_REQ(this->first.size() == this->second.size(), "Counts and hashes must have equal size");
    }
    template<typename Valloc, typename=std::enable_if_t<!std::is_same<Valloc, typename super::allocator>::value>>
    FinalCRMinHash(std::vector<T, Valloc> &&first, std::vector<CountType> &&second, size_t ss=0) {
        //this->first = std::move(first);
        this->first.assign(first.begin(), first.end());
        this->second = std::move(second);
        std::swap(first, std::vector<T, Valloc>());
        prepare(ss);
    }
    FinalCRMinHash(std::vector<T, typename super::allocator> &&first, std::vector<CountType> &&second, size_t ss=0) {
#if VERBOSE_AF
        std::fprintf(stderr, "first size: %zu.\n", first.size());
        std::fprintf(stderr, "second size: %zu.\n", second.size());
#endif
        this->first = std::move(first);
        this->second = std::move(second);
        prepare(ss);
    }
    template<typename Valloc>
    FinalCRMinHash(std::pair<std::vector<T, Valloc>, std::vector<CountType>> &&args, size_t ss=0):
        FinalCRMinHash(std::move(args.first), std::move(args.second), ss)
    {
    }
    template<typename Hasher, typename Cmp>
    static std::pair<std::vector<T>, std::vector<CountType>> crmh2vecs(const CountingRangeMinHash<T, Cmp, Hasher, CountType> &prefinal) {
        std::vector<T> tmp;
        std::vector<CountType> tmp2;
        tmp.reserve(prefinal.size()), tmp2.reserve(prefinal.size());
        for(const auto &pair: prefinal)
            tmp.push_back(pair.first), tmp2.push_back(pair.second);
        return std::make_pair(std::move(tmp), std::move(tmp2));
    }
    template<typename Hasher, typename Cmp>
    FinalCRMinHash(const CountingRangeMinHash<T, Cmp, Hasher, CountType> &prefinal): FinalCRMinHash(crmh2vecs(prefinal), prefinal.sketch_size()) {
        assert(this->first.size() == prefinal.sketch_size());
    }
    template<typename Hasher, typename Cmp>
    FinalCRMinHash(CountingRangeMinHash<T, Cmp, Hasher, CountType> &&prefinal): FinalCRMinHash(static_cast<const CountingRangeMinHash<T, Cmp, Hasher, CountType> &>(prefinal)) {
        prefinal.clear();
    }
    double jaccard_index(const FinalCRMinHash &o) const {
        return tf_idf(o);
    }
    double containment_index(const FinalCRMinHash &o) const {
        const size_t lsz = this->size();
        const size_t rsz = o.size();
        assert(rsz == o.second.size());
        assert(rsz == o.first.size());
        double denom = 0, num = 0;
        size_t i1 = 0, i2 = 0;
        size_t nused = 0;
        while(nused++ < lsz) {
            auto &lhs = this->first[i1];
            auto &rhs = o.first[i2];
            if(lhs < rhs) {
                denom += second[i1] * fn(lhs);
                if(++i1 == lsz) break;
            } else if(rhs < lhs) {
                if(++i2 == rsz) break;
            } else {
                auto tmpnum = std::min(second[i1], o.second[i2]);
                denom += fn(lhs) * second[i1];
                num   += fn(lhs) * tmpnum;
                ++i2, ++i1;
                if(i2 == rsz || i1 == lsz) break;
            }
            assert(i2 < o.second.size());
        }
        return double(num) / denom;
    }
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        return FinalRMinHash<T>::cardinality_estimate(mode);
    }
};



template<typename T>
double jaccard_index(const T &a, const T &b) {
    return a.jaccard_index(b);
}


template<bool weighted, typename Signature=std::uint32_t, typename IndexType=std::uint32_t,
         typename FT=float, bool sparsecache=false>
struct ShrivastavaHash {
    const IndexType nd_;
    const IndexType nh_;
    const uint64_t seedseed_;
    std::unique_ptr<uint64_t[]> seeds_;
    FT mv_;
    std::unique_ptr<FT[]> maxvals_;
    schism::Schismatic<IndexType> div_;
    std::vector<Signature> mintimes;

    INLINE static uint64_t preseed2final(uint64_t preseed) {
        uint64_t val = wy::wyhash64_stateless(&preseed);
        return val;
    }
public:
    ShrivastavaHash(size_t ndim, size_t nhashes, size_t seedseed=0): nd_(ndim), nh_(nhashes), seedseed_(seedseed), seeds_(new uint64_t[nh_]), div_(ndim) {
        CONST_IF(weighted) { // Defaults to weight of 1 until a weight is provided.
            mv_ = 1.;
        }
        uint64_t cseed = seedseed_;
        for(size_t i = 0; i < nh_; ++i) {
            seeds_[i] = wy::wyhash64_stateless(&cseed);
        }
        CONST_IF(sparsecache) {
            mintimes.resize(size_t(nh_) * nd_, std::numeric_limits<Signature>::max());
            OMP_PFOR
            for(size_t i = 0; i < nh_; ++i) {
                std::fprintf(stderr, "Starting caching for hash %zu/%u\n", i, nh_);
                unsigned left_to_find = nd_;
                uint64_t searchseed = seeds_[i];
                for(size_t cind = 0;;++cind) {
                    uint64_t val = preseed2final(searchseed + cind);
                    const size_t index = i * nd_ + div_.mod(val);
                    auto &mt(mintimes[index]);
                    if(mt == std::numeric_limits<Signature>::max()) {
                        mt = cind;
                        if(--left_to_find == 0u) break;
                    }
                }
            }
        }
    }
    void set_threshold(FT v) {
        mv_ = 1. / v;
    }
    static constexpr bool is_weighted() {return weighted;}
    template<typename OFT>
    void set_threshold(const OFT *p) {
        assert(p);
        maxvals_.reset(new FT[nd_]);
        std::transform(p, p + nd_, maxvals_.get(), [](auto x) {return static_cast<FT>(1. / x);});
        mv_ = 0.;
    }
    FT get_threshold(size_t ind) const {
        assert(!weighted || mv_ || maxvals_.get());
        return mv_ ? mv_: maxvals_[ind];
    }
    template<typename T>
    Signature compute_hash_index(const T &x, IndexType idx) const {
        uint64_t searchseed = seeds_[idx];
        for(Signature sig = 0; ;++sig) {
            uint64_t val = preseed2final(searchseed + sig);
            auto dm = div_.divmod(val);
            auto div = dm.quot, rem = dm.rem;
            CONST_IF(!weighted) {
                if(x[rem]) return sig;
            } else CONST_IF(sizeof(IndexType) == 4) {
                static constexpr FT finv = 1. / (1ull << 32);
                FT rv = (val >> 32) * finv;
                if(rv < get_threshold(rem) * x[rem]) return sig;
            } else {
                FT rv;
                if(nd_ < 0xFFFFFFFFull) {
                    static constexpr FT finv = 1. / (1ull << 32);
                    rv = (div & 0xFFFFFFFFull) * finv;
                } else {
                    static constexpr FT finv52 = 1. / (1ull << 52);
                    uint64_t nv = preseed2final(div);
                    rv = (nv >> 12) * finv52;
                }
                if(rv < get_threshold(rem) * x[rem])
                    return sig;
            }
        }
    }
    template<typename T>
    void hash(const T &x, Signature *ret, std::false_type) const {
        CONST_IF(!sparsecache) throw std::runtime_error("Cannot use sparsecache if not enabled");
        std::unique_ptr<FT[]> hm(new FT[this->nd_]());
        for(const auto &pair: x) hm[pair.index()] = pair.value();

        std::fill(ret, ret + this->nh_, std::numeric_limits<Signature>::max());
        CONST_IF(weighted) {
            for(unsigned i = 0; i < nh_; ++i) {
                for(auto &pair: x) {
                    ret[i] = std::min(mintimes[pair.index() * nh_ + i], ret[i]);
                }
                Signature time = ret[i];
                for(;;++time) {
                    uint64_t val = preseed2final(seeds_[i] + time);
                    auto dm = this->div_.divmod(val);
                    if(!hm[dm.rem]) continue;
                    auto div = dm.quot, rem = dm.rem;
                    FT rv;
                    CONST_IF(sizeof(IndexType) == 4) {
                        static constexpr FT finv = 1. / (1ull << 32);
                        //assert(hm.find(rem) != hm.end());
                        rv = (val >> 32) * finv;
                    } else {
                        static constexpr FT finv52 = 1. / (1ull << 52);
                        uint64_t nv = preseed2final(div);
                        rv = (nv >> 12) * finv52;
                    }
                    if(rv < this->get_threshold(rem) * hm[dm.rem]) break;
                }
                ret[i] = time;
            }
        } else {
            for(const auto &pair: x) {
                const size_t ind = pair.index();
                // Elementwise minimum between feature coordinates
                assert(mintimes.size() == this->nh_ * this->nd_);
#if !defined(VEC_DISABLED__) && (__AVX2__ || __SSE2__)
                static constexpr size_t MUL = sizeof(typename Space::Type) / sizeof(Signature);
                unsigned int i;
                auto retp = reinterpret_cast<typename Space::Type *>(ret);
                auto srcp = reinterpret_cast<const typename Space::Type *>(&mintimes[ind * this->nh_]);

                SK_UNROLL_8
                for(i = 0; i < this->nh_ / MUL; ++i) {
                    Space::storeu(retp + i, Space::min(Space::loadu(retp + i), Space::loadu(srcp + i)));
                }
                for(i *= MUL; i < this->nh_; ++i)
                    ret[i] = std::min(ret[i], mintimes[ind * this->nh_ + i]);
#else
                SK_UNROLL_8
                for(unsigned i = 0;i < this->nh_; ++i) {
                    ret[i] = std::min(ret[i], mintimes[ind * this->nh_ + i]);
                }
#endif
            }
        }
    }
    template<typename T>
    void hash(const T &x, Signature *ret, std::true_type) const {
        for(size_t i = 0; i < nh_; ++i) {
            ret[i] = compute_hash_index(x, i);
        }
    }
    template<typename T>
    void hash(T &x, Signature *ret) const {
        hash(x, ret, std::integral_constant<bool, std::is_arithmetic<std::decay_t<decltype(*x.begin())> >::value>());
    }
    template<typename T>
    std::vector<Signature> hash(const T &x) const {
        std::vector<Signature> ret(nh_);
        hash(x, ret.data());
        return ret;
    }
};

template<bool weighted, typename Signature=std::uint32_t, typename IndexType=std::uint32_t,
         typename FT=float>
struct SparseShrivastavaHash: public ShrivastavaHash<weighted, Signature, IndexType, FT, true> {
    template<typename...Args> SparseShrivastavaHash(Args&&...args):
        ShrivastavaHash<weighted, Signature, IndexType, FT, true>(std::forward<Args>(args)...) {}
};

template<typename HashStruct, typename VT, bool select_bottom>
struct BottomKHasher {
    using final_type = FinalRMinHash<VT, Allocator<VT>>;
    using heap_cmp = std::conditional_t<select_bottom,
                                        std::less<void>, std::greater<void>>;
    struct mpq: std::priority_queue<VT, std::vector<VT, Allocator<VT>>, heap_cmp> {
        auto &getq() {return this->c;}
        const auto &getq() const {return this->c;}
        template<typename X, typename Y>
        bool cmp(const X &x, const Y &y) const {
            return this->comp(x, y);
        }
    };

    size_t k_;
    HashStruct hs_;
    mpq mpq_;
    ska::flat_hash_set<VT> set_;

    void clear() {
        set_.clear();
        mpq_.getq().clear();
    }
    void reset() {clear();}

    double cardinality_estimate() const {
        if(select_bottom) {
            return double(std::numeric_limits<VT>::max()) / this->mpq_.top() * mpq_.size();
        } else {
            return double(std::numeric_limits<VT>::max()) / (std::numeric_limits<VT>::max() - this->mpq_.top()) * mpq_.size();
        }
    }

    BottomKHasher(size_t k, HashStruct &&hs=HashStruct()): k_(k), hs_(std::move(hs)) {}
    void addh(uint64_t v) {add(hs_(v));}
    void add(uint64_t hv) {
        if(set_.find(hv) != set_.end()) return;
        if(mpq_.size() < k_) {
            mpq_.push(hv);
            set_.insert(hv);
        } else if(mpq_.cmp(hv, mpq_.top())) {
            set_.erase(mpq_.top());
            mpq_.pop(), mpq_.push(hv);
            set_.insert(hv);
        }
    }
    final_type finalize() const & {
        return final_type(mpq_.getq());
    }
    final_type finalize() && {
        return final_type(std::move(mpq_.getq()));
    }
    void write(std::string path) const {
        this->finalize().write(path);
    }
    void write(gzFile fp) const {
        this->finalize().write(fp);
    }
    ssize_t read(std::string s) {
        FinalRMinHash<VT> ret(s.data());
        k_ = ret.first.size();
        set_.reserve(k_);
        for(const auto v: ret.first) {
            set_.insert(v);
            mpq_.push(v);
        }
        return sizeof(ret.first[0]) * ret.first.size() + sizeof(ret);
    }
    ssize_t read(gzFile fp) {
        FinalRMinHash<VT> ret(fp);
        k_ = ret.first.size();
        set_.reserve(k_);
        for(const auto v: ret.first) {
            set_.insert(v);
            mpq_.push(v);
        }
        return sizeof(ret.first[0]) * ret.first.size() + sizeof(ret);
    }
};


} // inline namespace minhash
namespace mh = minhash;
} // namespace sketch

#undef ONLY_SKETCH_UNION
