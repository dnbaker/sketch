#pragma once
#include <mutex>
//#include <queue>
#include "hll.h" // For common.h and clz functions
#include "fixed_vector.h"
#include <unordered_map>
#include "isz.h"


/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  HyperMinHash
 */

namespace sketch {
inline namespace minhash {

#define SET_SKETCH(x) const auto &sketch() const {return x;} auto &sketch() {return x;}

namespace detail {



static constexpr double HMH_C = 0.169919487159739093975315012348;

template<typename FType, typename=typename std::enable_if<std::is_floating_point<FType>::value>::type>
inline FType beta(FType v) {
    FType ret = -0.370393911 * v;
    v = std::log1p(v);
    FType poly = v * v;
    ret += 0.070471823 * v;
    ret += 0.17393686 * poly;
    poly *= v;
    ret += 0.16339839 * poly;
    poly *= v;
    ret -= 0.09237745 * poly;
    poly *= v;
    ret += 0.03738027 * poly;
    poly *= v;
    ret += -0.005384159 * poly;
    poly *= v;
    ret += 0.00042419 * poly;
    return ret;
}

} // namespace detail

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




template<typename T, typename Cmp=std::greater<T>, typename Hasher=common::WangHash>
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
    size_t nhashes_per_vector() const {return sizeof(uint64_t) / sizeof(Space::Type);}
    void addh(T val) {
        throw NotImplementedError();
        // Hash stuff.
        // Then use a vectorized minimization comparison instruction
    }
    SET_SKETCH(hashes_)
};

template<typename T, typename Allocator> struct FinalRMinHash; // Forward definition
/*
The sketch is the set of minimizers.

*/
template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash,
         typename Allocator=common::Allocator<T>
        >
struct RangeMinHash: public AbstractMinHash<T, Cmp> {
protected:
    Hasher hf_;
    Cmp cmp_;
    ///using HeapType = std::priority_queue<T, std::vector<T, Allocator<T>>>;
    std::set<T, Cmp> minimizers_; // using std::greater<T> so that we can erase from begin()

public:
    using final_type = FinalRMinHash<T, Allocator>;
    using Compare = Cmp;
    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()):
        AbstractMinHash<T, Cmp>(sketch_size), hf_(std::move(hf)), cmp_(std::move(cmp))
    {
    }
    RangeMinHash(std::string) {throw NotImplementedError("");}
    double cardinality_estimate() const {
        return double(std::numeric_limits<T>::max()) / this->max_element() * minimizers_.size();
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
        while(minimizers_.size() > this->ss_)
            minimizers_.erase(minimizers_.begin());
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
        val = hf_(val);
        this->add(val);
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
        return common::intersection_size(o, *this, cmp_);
    }
    double jaccard_index(const RangeMinHash &o) const {
        assert(o.size() == minimizers_.size());
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
        if(reta.size() < this->ss_)
            reta.insert(reta.end(), this->ss_ - reta.size(), std::numeric_limits<uint64_t>::max());
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
        return common::intersection_size(first, o.first);
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
    void print_all_cards() const {
        for(const auto x: {HARMONIC_MEAN, ARITHMETIC_MEAN, MEDIAN}) {
            std::fprintf(stderr, "cardest with x = %d is %lf\n", int(x), cardinality_estimate(x));
        }
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
    double union_size(const FinalRMinHash &o) const {
        PREC_REQ(this->size() == o.size(), "Non-matching parameters for FinalRMinHash comparison");
        size_t n_in_sketch = 0;
        auto i1 = this->rbegin(), i2 = o.rbegin();
        T mv;
        while(n_in_sketch < first.size() - 1) {
            // Easier to branch-predict:  http://www.vldb.org/pvldb/vol8/p293-inoue.pdf
            if(*i1 != *i2) ++i1, ++i2;
            else {
                const int c = *i1 < *i2;
                i2 += !c; i1 += c;
            }
            ++n_in_sketch;
        }
        mv = *i1 < *i2 ? *i1: *i2;
        // TODO: test after refactoring
        assert(i1 < this->rend());
        return double(std::numeric_limits<T>::max()) / (mv) * this->size();
    }
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        // KMV estimate
        double sum = (std::numeric_limits<T>::max() / double(this->max_element()) * first.size());
        return sum;
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
    template<typename Alloc>
    FinalRMinHash(const std::vector<T, Alloc> &ofirst): first(ofirst.size()) {std::copy(ofirst.begin(), ofirst.end(), first.begin()); sort();}
    template<typename It>
    FinalRMinHash(It start, It end): first(std::distance(start, end)) {
        std::copy(start, end, first.begin());
        sort();
    }
    template<typename Alloc, typename=std::enable_if_t<std::is_same<Alloc, allocator>::value>>
    FinalRMinHash(std::vector<T, Alloc> &&ofirst): first(std::move(ofirst)) {
        sort();
    }
    FinalRMinHash(FinalRMinHash &&o): first(std::move(o.first)) {sort();}
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
    template<typename Hasher, typename Cmp>
    FinalRMinHash(RangeMinHash<T, Cmp, Hasher> &&prefinal): FinalRMinHash(std::move(prefinal.finalize())) {
        prefinal.clear();
        sort();
    }
    FinalRMinHash(const std::string &s): FinalRMinHash(s.data()) {}
    FinalRMinHash(gzFile fp) {read(fp);}
    FinalRMinHash(const char *infname) {read(infname);}
    size_t size() const {return first.size();}
protected:
    FinalRMinHash() {}
    FinalRMinHash(const FinalRMinHash &o) = default;
    FinalRMinHash &operator=(const FinalRMinHash &o) = default;
    void sort() {
        common::sort::default_sort(this->first.begin(), this->first.end());
    }
};

template<typename T, typename CountType> struct FinalCRMinHash; // Forward


template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash,
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
                if( (++i1 == e1) | (++i2 == e2)) break;
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
        for(const auto v: this->second) lsum += v * v;
        
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
        size_t num = 0, denom = 0;
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


template<typename T=uint64_t, typename Hasher=WangHash>
class HyperMinHash {
    uint64_t seeds_ [2] __attribute__ ((aligned (sizeof(uint64_t) * 2)));
    DefaultCompactVectorType core_;
    uint16_t p_, r_;
    Hasher hf_;
#ifndef NOT_THREADSAFE
    std::mutex mutex_; // I should be able to replace most of these with atomics.
#endif
public:
    static constexpr uint32_t q() {return uint32_t(std::ceil(ilog2(sizeof(T) * CHAR_BIT)));} // To hold popcount for a 64-bit integer.
    enum ComparePolicy {
        Manual,
        U8,
        U16,
        U32,
        U64,
    };
    void swap(HyperMinHash &o) {
        std::swap_ranges(reinterpret_cast<uint8_t *>(this), reinterpret_cast<uint8_t *>(this) + sizeof(*this), reinterpret_cast<uint8_t *>(&o));
    }
    auto &core() {return core_;}
    const auto &core() const {return core_;}
    uint64_t mask() const {
        return (uint64_t(1) << r_) - 1;
    }
    auto max_mhval() const {return mask();}
    template<typename... Args>
    HyperMinHash(unsigned p, unsigned r, Args &&...args):
        seeds_{0xB0BAF377C001D00DuLL, 0x430c0277b68144b5uLL}, // Fully arbitrary seeds
        core_(r + q(), 1ull << p),
        p_(p), r_(r),
        hf_(std::forward<Args>(args)...)
    {
        common::detail::zero_memory(core_); // Second parameter is a dummy for interface compatibility with STL
#if VERBOSE_AF
        std::fprintf(stderr, "p: %u. r: %u\n", p, r);
        print_params();
#endif
    }
private:
    HyperMinHash() {}
public:
    void write(gzFile fp) const {
        gzwrite(fp, this, sizeof(*this));
        gzwrite(fp, core_.get(), core_.bytes());
    }
    void write(const char *path) const {
        gzFile fp = gzopen(path, "wb");
        if(fp == nullptr) throw ZlibError(Z_ERRNO, std::string("Could not open file for writingn at ") + path);
        this->write(fp);
        gzclose(fp);
    }
    void read(gzFile fp) const {
        this->clear();
        HyperMinHash tmp;
        gzread(fp, &tmp, sizeof(tmp));
        r_ = tmp.r_;
        p_ = tmp.p_;
        core_ = DefaultCompactVectorType(tmp.r_ + tmp.q(), 1ull << tmp.p_);
        seeds_as_sse() = tmp.seeds_as_sse();
        gzread(fp, core_.get(), core_.bytes());
        std::memset(&tmp, 0, sizeof(tmp));
    }
    void clear() {
        std::memset(core_.get(), 0, core_.bytes());
    }
    HyperMinHash(const HyperMinHash &a): core_(a.r() + q(), 1ull << a.p()), p_(a.p_), r_(a.r_), hf_(a.hf_) {
        seeds_as_sse() = a.seeds_as_sse();
        assert(a.core_.bytes() == core_.bytes());
        std::memcpy(core_.get(), a.core_.get(), core_.bytes());
    }
    void print_all(std::FILE *fp=stderr) {
        for(size_t i = 0; i < core_.size(); ++i) {
            size_t v = core_[i];
            std::fprintf(stderr, "Index %zu has value %d for lzc and %d for remainder, with full value = %zu\n", i, int(get_lzc(v)), int(get_mhr(v)), size_t(core_[i]));
        }
    }
    auto minimizer_size() const {
        return r_ + q();
    }
    ComparePolicy simd_policy() const {
        switch(minimizer_size()) {
            case 8:  return ComparePolicy::U8;
            case 16: return ComparePolicy::U16;
            case 32: return ComparePolicy::U32;
            case 64: return ComparePolicy::U64;
            default: return ComparePolicy::Manual;
        }
    }
    // Encoding and decoding table entries
    auto get_lzc(uint64_t entry) const {
        return entry >> r_;
    }
    auto get_mhr(uint64_t entry) const {
        return entry & max_mhval();
    }
    template<typename I1, typename I2,
             typename=typename std::enable_if<std::is_integral<I1>::value && std::is_integral<I1>::value>::type>
    auto encode_register(I1 lzc, I2 min) const {
        // We expect that min has already been masked so as to eliminate unnecessary operations
        assert(min <= max_mhval());
        return (uint64_t(lzc) << r_) | min;
    }
    std::array<uint32_t, 64> sum_counts() const {
        using hll::detail::SIMDHolder;
        // TODO: this
        // Note: we have whip out more complicated vectorized maxes for
        // widths of 16, 32, or 64
        std::array<uint32_t, 64> ret{0};
        if(core_.bytes() >= sizeof(SIMDHolder)) {
            switch(simd_policy()) {
#define CASE_U(cse, func, msk)\
                case cse: {\
                    const Space::Type mask = Space::set1(UINT64_C(msk));\
                    for(const SIMDHolder *ptr = reinterpret_cast<const SIMDHolder *>(core_.get()), *eptr = reinterpret_cast<const SIMDHolder *>(core_.get() + core_.bytes());\
                        ptr != eptr; ++ptr) {\
                        auto tmp = *ptr;\
                        tmp = Space::and_fn(Space::srli(*reinterpret_cast<VType *>(&tmp), r_), mask);\
                        tmp.func(ret);\
                    }\
                    break;\
                }
                CASE_U(U8,  inc_counts,   0x3f3f3f3f3f3f3f3f)
                CASE_U(U16, inc_counts16, 0x003f003f003f003f)
                CASE_U(U32, inc_counts32, 0x0000003f0000003f)
                CASE_U(U64, inc_counts64, 0x000000000000003f)
#undef CASE_U
                case Manual: default: goto manual_core;
            }
        } else {
            manual_core:
            for(const auto i: core_) {
                uint8_t lzc = get_lzc(i);
                if(__builtin_expect(lzc > 64, 0)) {std::fprintf(stderr, "Value for %d should not be %dn", int(i), int(get_lzc(i))); std::exit(1);}\
                ++ret[lzc];
            }
        }
        return ret;
    }
#undef MANUAL_CORE
    double estimate_hll_portion(double relerr=1e-2) const {
        return hll::detail::ertl_ml_estimate(this->sum_counts(), p(), q(), relerr);
    }
    double report(double relerr=1e-2) const {
        const auto csum = this->sum_counts();
#if VERBOSE_AF
        std::fprintf(stderr, "Performing estimate. Counts values: ");
        for(const auto v: csum)
            std::fprintf(stderr, "%zu,", v);
        std::fprintf(stderr, "\n");
#endif
        double est = hll::detail::ertl_ml_estimate(csum, p(), 64 - p(), relerr);
        if(est > 0. && est < static_cast<double>(core_.size() << 10)) {
#if VERBOSE_AF
            std::fprintf(stderr, "report ertl_ml_estimate %lf\n", est);
#endif
            return est;
        }
        const double mhinv = std::ldexp(1, -int(r_));
        std::fprintf(stderr, "mhinv: %lf. Manual: %lf\n", mhinv, 1./(1<<r_));
        double sum = 0.;
        for(const auto v: core_) {
            sum += std::ldexp(1. + std::ldexp(get_mhr(v), -int64_t(r_)), -int64_t(get_lzc(v)));
#if VERBOSE_AF
            std::fprintf(stderr, "sum: %lf\n", sum);
#endif
        }
        if(__builtin_expect(!sum, 0)) sum = std::numeric_limits<double>::infinity();
        else                          sum = static_cast<double>(std::pow(core_.size(), 2)) / sum;
        return sum;
    }
    auto p() const {return p_;}
    auto r() const {return r_;}
    void set_seeds(uint64_t seed1, uint64_t seed2) {
        seeds_[0] = seed1; seeds_[1] = seed2;
    }
    void set_seeds(__m128i s) {seeds_as_sse() = s;}
    int print_params(std::FILE *fp=stderr) const {
        return std::fprintf(fp, "p: %u. q: %u. r: %u.\n", p(), q(), r());
    }
    const __m128i &seeds_as_sse() const {return *reinterpret_cast<const __m128i *>(seeds_);}
    __m128i &seeds_as_sse() {return *reinterpret_cast<__m128i *>(seeds_);}

    INLINE void addh(uint64_t element) {
        add(hf_(_mm_set1_epi64x(element) ^ seeds_as_sse()));
    }
    template<typename ET>
    INLINE void addh(ET element) {
        element.for_each([&](uint64_t el) {this->addh(el);});
    }
    HyperMinHash &operator+=(const HyperMinHash &o) {
        using hll::detail::SIMDHolder;
        // This needs:
        // Vectorized maxes
        if(core_.bytes() >= sizeof(SIMDHolder)) {
            const SIMDHolder *optr = reinterpret_cast<const SIMDHolder *>(o.core_.get());
            SIMDHolder *ptr = reinterpret_cast<SIMDHolder *>(core_.get()),
                       *eptr = reinterpret_cast<SIMDHolder *>(core_.get() + core_.bytes());
            switch(simd_policy()) {
#define CASE_U(cse, op)\
                case cse:\
                    do {*ptr = SIMDHolder::op(*ptr, *optr++);} while(++ptr != eptr); break;
                CASE_U(U8, max_fn)
                CASE_U(U16, max_fn16)
                CASE_U(U32, max_fn32)
                CASE_U(U64, max_fn64)
                default: goto manual;
            }
        } else {
            manual:
            for(size_t i(0); i < core_.size(); ++i)
                if(core_[i] < o.core_[i])
                    core_[i] = o.core_[i];
        }
        return *this;
    }
    HyperMinHash &operator=(const HyperMinHash &a)
#if 0
    {
        core_ = DefaultCompactVectorType(q() + a.r(), 1 << a.p());
        seeds_as_sse() = a.seeds_as_sse();
        std::memcpy(core_.get(), a.core_.get(), core_.bytes());
    }
#else
    = delete;
#endif
    HyperMinHash(const HyperMinHash &a, const HyperMinHash &b): HyperMinHash(a)
    {
        *this += b;
    }
    HyperMinHash operator+(const HyperMinHash &a) const {
        if(__builtin_expect(a.p() != p() || a.q() != q(), 0)) throw std::runtime_error("Could not merge sketches of differing parameter sets");
        HyperMinHash ret(*this);
        ret += a;
        return ret;
    }
    //HyperMinHash(const HyperMinHash &) = delete;
    HyperMinHash(HyperMinHash &&) = default;
    HyperMinHash &operator=(HyperMinHash &&) = default;
    INLINE void add(__m128i hashval) {
    // TODO: Consider looking for a way to use the leading zero count to store the rest of a key
    // Not sure this is valid for the purposes of an independent hash.
        uint64_t arr[2];
        static_assert(sizeof(hashval) == sizeof(arr), "Size sanity check");
        std::memcpy(&arr[0], &hashval, sizeof(hashval));
        const uint64_t index(reinterpret_cast<uint64_t *>(&hashval)[0] >> (64 - p())),
                         lzt(integral::clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
        const uint64_t inserted_val = encode_register(lzt, reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval());
        assert(get_lzc(inserted_val) == lzt);
        assert((reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) == get_mhr(inserted_val));
        if(core_[index] < inserted_val) { // Consider other functions for specific register sizes.
            core_[index] = inserted_val;
            assert(encode_register(lzt, get_mhr(inserted_val)) == inserted_val);
            assert(lzt == get_lzc(inserted_val));
        }
    }
    double jaccard_index(const HyperMinHash &o) const {
        size_t C = 0, N = 0;
        std::fprintf(stderr, "core size: %zu\n", core_.size());
#if 0
        switch(simd_policy()) { // This can be accelerated for specific sizes
            default: //[[fallthrough]];
            //U8:      //[[fallthrough]]; // 2-bit minimizers. TODO: write this
            //U16:     //[[fallthrough]]; // 10-bit minimizers. TODO: write this
            //U32:     //[[fallthrough]]; // 26-bit minimizers. TODO: write this
            //U64:     //[[fallthrough]]; // 58-bit minimizers. TODO: write this
            Manual:
                for(size_t i = 0; i < core_.size(); ++i) {
                    C += core_[i] && (get_lzc(core_[i]) == get_lzc(o.core_[i]));
                    N += (core_[i] || o.core_[i]);
                }
            break;
        }
#else
#endif
        for(size_t i = 0; i < core_.size(); ++i) {
            C += core_[i] && (get_lzc(core_[i]) == get_lzc(o.core_[i]));
            N += (core_[i] || o.core_[i]);
        }
        const double n = this->report(), m = o.report(), ec = expected_collisions(n, m);
        std::fprintf(stderr, "C: %zu. ec: %lf. C / N: %lf\n", C, ec, static_cast<double>(C) / N);
        return std::max((C - ec) / N, 0.);
    }
    double expected_collisions(double n, double m, bool easy_way=false) const {
#if MY_WAY
        if(easy_way) {
            if(n < m) std::swap(n, m);
            auto l2n = ilog2(n);
            if(l2n > ((1 << q()) + r())) {
#if VERBOSE_AF
                std::fprintf(stderr, "Warning: too high to approximate\n");
#endif
                goto slow;
            }
            if(l2n > p() + 5) {
                const double nm = n/m;
                const double phi = 4 * nm / std::pow((1 + n) / m, 2);
                return std::ldexp(detail::HMH_C * phi, p_ - r());
            }
        }
        slow:
#if !NDEBUG
        std::fprintf(stderr, "Using slow expected collisions method\n");
#endif
        double x = 0.;
        for(size_t i = 1; i <= size_t(1) << p(); ++i) {
            for(size_t j = 1; j <= size_t(1) << r(); ++j) {
                double b1, b2;
                const int _p= p_, _r = r_; // Redeclaring as signed integers to avoid underflow
                if(i != size_t(1) << q()) {
                    b1 = std::ldexp((size_t(1) << r()) + j, -_p - _r - i);
                    b2 = std::ldexp((size_t(1) << r()) + j + 1, -_p - _r - i);
                } else {
                    b1 = std::ldexp(j, -p_ - _r - i - 1);
                    b2 = std::ldexp(j + 1, -p_ - _r - i - 1);
                }
                const double prx = std::pow(1.-b2, n) - std::pow(1.-b1, n);
                const double pry = std::pow(1.-b2, m) - std::pow(1.-b1, m);
                x += prx * pry;
            }
        }
        return easy_way ? x: std::ldexp(x, p());
#else
        size_t r2 = 1ull << r(), q2 = 1ull << q();
        double x = 0;
        for(size_t i = 1; i <= q2; ++i) {
            for(size_t j = 1; j <= r2; ++j) {
                auto b1 = i != q2 ? std::ldexp(r2 + j, -int32_t(p() + r() + i)): std::ldexp(j, -int32_t(p() + r() + i - 1));
                auto b2 = i != q2 ? std::ldexp(r2 + j + 1, -int32_t(p() + r() + i)): std::ldexp(j + 1,  -int32_t(p() + r() + i - 1));
                auto prx = std::pow(1 - b2, n) - std::pow(1 - b1, n);
                auto pry = std::pow(1 - b2, m) - std::pow(1 - b1, m);
                x += prx * pry;
            }
        }
        return std::ldexp(x, p());
#endif
    }
};
template<typename T, typename Hasher>
void swap(HyperMinHash<T,Hasher> &a, HyperMinHash<T,Hasher> &b) {a.swap(b);}

template<typename T>
double jaccard_index(const T &a, const T &b) {
    return a.jaccard_index(b);
}


} // inline namespace minhash
namespace mh = minhash;
} // namespace sketch

#undef ONLY_SKETCH_UNION
