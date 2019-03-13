#pragma once
#include <mutex>
//#include <queue>
#include "hll.h" // For common.h and clz functions

/*
 * TODO: support minhash using sketch size and a variable number of hashes.
 * Implementations: KMinHash (multiple hash functions)
 *                  RangeMinHash (the lowest range in a set, but only uses one hash)
 *                  HyperMinHash
 */

namespace sketch {
namespace minhash {
using namespace common;

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


template<typename Container, typename Cmp=typename Container::key_compare>
std::uint64_t intersection_size(const Container &c1, const Container &c2, const Cmp &cmp=Cmp()) {
    // These containers must be sorted.
    std::uint64_t ret = 0;
    auto it1 = c1.begin();
    auto it2 = c2.begin();
    const auto e1 = c1.end();
    const auto e2 = c2.end();
    for(;;) {
        if(cmp(*it1, *it2)) {
            if(++it1 == e1) break;
        } else if(cmp(*it2, *it1)) {
            if(++it2 == e2) break;
        } else {
            ++ret;
            if(++it1 == e1 || ++it2 == e2) break;
        }
    }
    return ret;
}


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

template<typename T, typename Cmp> struct FinalRMinHash; // Forward definition
/*
The sketch is the set of minimizers.

*/
template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash
        >
class RangeMinHash: public AbstractMinHash<T, Cmp> {
protected:
    Hasher hf_;
    Cmp cmp_;
    ///using HeapType = std::priority_queue<T, std::vector<T, Allocator<T>>>;
    std::set<T, Cmp> minimizers_; // using std::greater<T> so that we can erase from begin()

public:
    using final_type = FinalRMinHash<T, Cmp>;
    RangeMinHash(size_t sketch_size, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()):
        AbstractMinHash<T, Cmp>(sketch_size), hf_(std::move(hf)), cmp_(std::move(cmp))
    {
    }
    void read(const std::string &s) {
        throw NotImplementedError("This hasn't been implemented, but you should be serializing FinalRMinHash anyhow");
    }
    double cardinality_estimate() const {
        return double(std::numeric_limits<T>::max()) / this->max_element() * minimizers_.size();
#if 0
         const auto sz = minimizers_.size();
         common::detail::alloca_wrap<T> mem(sz - 1);
         T *diffs = mem.get(), *p = diffs;
         for(auto i1 = minimizers_.begin(), i2 = i1; ++i2 != minimizers_.end();++i1) {
             *p++ = *i1 > *i2 ? *i1 - *i2: *i2 - *i1;
         }
         sort::insertion_sort(diffs, p);
         const auto ret = double(UINT64_C(-1)) / ((diffs[(sz - 1)>>1] + diffs[((sz - 1) >> 1) - 1]) >> 1);
         return ret;
#endif
    }
    RangeMinHash(gzFile fp) {
        if(!fp) throw std::runtime_error("Null file handle!");
        this->read(fp);
    }
    RangeMinHash(const char *infname) {
        gzFile fp = gzopen(infname, "rb");
        this->read(fp);
        gzclose(fp);
    }
    void write(const std::string &s) const {
        gzFile fp = gzopen(s.data(), "wb");
        if(!fp) throw 1;
        this->write(fp);
        gzclose(fp);
    }
#if 0
    void write(gzFile fp) const {
        this->finalize().write(fp);
    }
#endif
    void read(gzFile fp) {
        gzread(fp, this, sizeof(*this));
        std::memset(&minimizers_, 0, sizeof(minimizers_));
        T v;
        for(ssize_t read; (read = gzread(fp, &v, sizeof(v))) == read;minimizers_.insert(v));
    }
    void write(const char *path) const {
        gzFile fp = gzopen(path, "wb");
        this->write(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        if(!fp) throw std::runtime_error("Null file handle!");
        gzwrite(fp, this, sizeof(*this));
        for(const auto v: minimizers_)
            gzwrite(fp, &v, sizeof(v));
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
                if(minimizers_.size() > this->ss_) {
#if 0
                    auto el = *minimizers_.begin();
                    bool gt = true, lt = true;
                    for(const auto e: minimizers_)
                        gt &= (el >= e), lt &= (el <= e);
                    //std::fprintf(stderr, "Element is greater than all in the set: %d. lt: %d\n", int(gt), int(lt));
#else
                    minimizers_.erase(minimizers_.begin());
#endif
                }
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
        return minhash::intersection_size(o, *this, Cmp());
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = this->intersection_size(o);
        return is / (minimizers_.size() + o.size() - is);
    }
    template<typename Container>
    Container to_container() const {
        if(this->ss_ != size()) // If the sketch isn't full, add UINT64_MAX to the end until it is.
            minimizers_.resize(this->ss_, std::numeric_limits<uint64_t>::max());
        return Container(std::rbegin(minimizers_), std::rend(minimizers_.end()));
    }
    void clear() {
        decltype(minimizers_)().swap(minimizers_);
    }
    final_type finalize() const {
        std::vector<T> reta(minimizers_.begin(), minimizers_.end());
        reta.insert(reta.end(), this->ss_ - reta.size(), std::numeric_limits<uint64_t>::max());
        return final_type{std::move(reta)};
    }
    std::vector<T> mh2vec() const {return to_container<std::vector<T>>();}
    size_t size() const {return minimizers_.size();}
    using key_compare = typename decltype(minimizers_)::key_compare;
    SET_SKETCH(minimizers_)
};

namespace weight {
struct EqualWeight {
    template<typename T>
    constexpr double operator()(T &x) const {return 1.;}
};
}

template<typename T, typename Cmp>
struct FinalRMinHash {
    std::vector<T> first;
    Cmp cmp;
    size_t intersection_size(const FinalRMinHash &o) const {
        return minhash::intersection_size(first, o.first, Cmp());
    }
    double jaccard_index(const FinalRMinHash &o) const {
        double is = intersection_size(o);
        return is / ((size() << 1) - is);
    }
    void print_all_cards() const {
        for(const auto x: {HARMONIC_MEAN, ARITHMETIC_MEAN, MEDIAN}) {
            std::fprintf(stderr, "cardest with x = %d is %lf\n", int(x), cardinality_estimate(x));
        }
    }
    double union_size(const FinalRMinHash &o) const {
        if(this->size() != o.size()) throw std::runtime_error("Non-matching parameters for FinalRMinHash comparison");
        return std::numeric_limits<T>::max() / double(std::max(this->max_element(), o.max_element())) * this->size();
    }
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        switch(mode) {
            case ARITHMETIC_MEAN: {
                // KMV estimate
                double sum = (std::numeric_limits<T>::max() / double(this->max_element()) * first.size());
                return sum;
            }
            default: std::fprintf(stderr, "Warning: unsupported case. Falling back to median\n");
            case MEDIAN: {
                const auto sz = first.size();
                common::detail::alloca_wrap<T> mem(sz - 1);
                T *diffs = mem.get(), *p = diffs;
                for(auto i1 = first.begin(), i2 = i1; ++i2 != first.end();++i1) {
                    *p++ = (*i1 - *i2);
                }
                sort::insertion_sort(diffs, p);
                return double(UINT64_C(-1)) / ((diffs[(sz - 1)>>1] + diffs[((sz - 1) >> 1) - 1]) >> 1);
             }
        }
    }
#define I1DF if(++i1 == lsz) break
#define I2DF if(++i2 == lsz) break
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalRMinHash &o, const WeightFn &fn=WeightFn()) const {
        assert(o.size() == size());
        const size_t lsz = size();
        double denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(cmp(first[i1], o.first[i2])) {
                denom += fn(first[i1]);
                I1DF;
            } else if(cmp(o.first[i2], first[i1])) {
                denom += fn(o.first[i2]);
                I2DF;
            } else {
                const auto v1 = fn(first[i1]), v2 = fn(o.first[i2]);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1DF; I2DF;
            }
        }
        return num / denom;
    }
    void free() {
        decltype(first) tmp; std::swap(tmp, first);
    }
    FinalRMinHash(std::vector<T> &&first): first(std::move(first)), cmp() {}
    FinalRMinHash(FinalRMinHash &&o): first(std::move(o.first)), cmp(std::move(o.cmp)) {}
    void write(const std::string &path, int comp=6) const {
        write(path.data(), comp);
    }
    void read(const char *path) {
        gzFile fp = gzopen(path, "rb");
        if(!fp) throw std::runtime_error(std::string("could not open file at ") + path);
        uint64_t sz;
        gzread(fp, &sz, sizeof(sz));
        first.resize(sz);
        gzread(fp, first.data(), first.size() * sizeof(first[0]));
        gzclose(fp);
    }
    void write(const char *path, int comp=6) const {
        gzFile fp = gzopen(path, "wb");
        if(!fp) throw std::runtime_error(std::string("could not open file at ") + path);
        uint64_t sz = this->size();
        gzwrite(fp, &sz, sizeof(sz));
        gzwrite(fp, first.data(), first.size() * sizeof(first[0]));
        gzclose(fp);
    }
    auto max_element() const {
        assert(std::accumulate(first.begin(), first.end(), true, [&](bool t, auto v) {return t && *first.begin() >= v;}));
        return *first.begin();
    }
    template<typename Hasher>
    FinalRMinHash(RangeMinHash<T, Cmp, Hasher> &&prefinal): FinalRMinHash(std::move(prefinal.finalize())) {
        prefinal.clear();
    }
    FinalRMinHash(const std::string &s): FinalRMinHash(s.data()) {}
    FinalRMinHash(const char *infname) {
        this->read(infname);
    }
    size_t size() const {return first.size();}
protected:
    FinalRMinHash() {}
};

template<typename T, typename Cmp, typename CountType> struct FinalCRMinHash; // Forward


template<typename T,
         typename Cmp=std::greater<T>,
         typename Hasher=common::WangHash,
         typename CountType=uint32_t
        >
class CountingRangeMinHash: public AbstractMinHash<T, Cmp> {
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
    };
    Hasher hf_;
    Cmp cmp_;
    std::set<VType> minimizers_; // using std::greater<T> so that we can erase from begin()
public:
    const auto &min() const {return minimizers_;}
    using size_type = CountType;
    using final_type = FinalCRMinHash<T, Cmp, CountType>;
    auto size() const {return minimizers_.size();}
    auto begin() {return minimizers_.begin();}
    auto begin() const {return minimizers_.begin();}
    auto rbegin() {return minimizers_.rbegin();}
    auto rbegin() const {return minimizers_.rbegin();}
    auto end() {return minimizers_.end();}
    auto end() const {return minimizers_.end();}
    auto rend() {return minimizers_.rend();}
    auto rend() const {return minimizers_.rend();}
    CountingRangeMinHash(size_t n, Hasher &&hf=Hasher(), Cmp &&cmp=Cmp()): AbstractMinHash<T, Cmp>(n), hf_(std::move(hf)), cmp_(std::move(cmp)) {}
    void read(const std::string &s) {
        throw NotImplementedError("This hasn't been implemented. You should probably be using FinalCRMinHash instead if you're serializing.");
    }
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        return double(std::numeric_limits<T>::max()) / *std::max_element(minimizers_.begin(), minimizers_.end(), [](auto x, auto y) {return x->first < y->first;}) * minimizers_.size();
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
    double histogram_intersection(const CountingRangeMinHash &o) const {
        assert(o.size() == size());
        size_t denom = 0, num = 0;
        auto i1 = minimizers_.begin(), i2 = o.minimizers_.begin();
#define I1DM if(++i1 == minimizers_.end()) break
#define I2DM if(++i2 == o.minimizers_.end()) break
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                denom += i1->second;
                I1DM;
            } else if(cmp_(i2->first, i1->first)) {
                denom += i2->second;
                I2DM;
            } else {
                const auto v1 = i1->second, v2 = i2->second;
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1DM;
                I2DM;
            }
        }
        return static_cast<double>(num) / denom;
    }
    template<typename Handle>
    void write(Handle handle) const {
        this->finalize().write(handle);
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
        for(;;) {
            if(cmp_(i1->first, i2->first)) {
                denom += (i1->second) * fn(i1->first);
                I1DM;
            } else if(cmp_(i2->first, i1->first)) {
                denom += (i2->second) * fn(i2->first);
                I2DM;
            } else {
                const auto v1 = i1->second * fn(i1->first), v2 = i2->second * fn(i2->first);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1DM;
                I2DM;
            }
        }
#undef I1DM
#undef I2DM
        return static_cast<double>(num) / denom;
    }
    final_type finalize() const {
        return FinalCRMinHash<T, Cmp, CountType>(*this);
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
    template<typename C2>
    size_t intersection_size(const C2 &o) const {
        return minhash::intersection_size(o, *this, [&](auto &x, auto &y) {return cmp_(x.first, y.first);});
    }
    template<typename C2>
    double jaccard_index(const C2 &o) const {
        double is = this->intersection_size(o);
        return is / (minimizers_.size() + o.size() - is);
    }
};


template<typename T, typename Cmp, typename CountType>
struct FinalCRMinHash: public FinalRMinHash<T, Cmp> {
    std::vector<CountType> second;
    FinalCRMinHash(const std::string &path): FinalCRMinHash(path.data()) {}
    FinalCRMinHash(const char *path) {
        this->read(path);
    }
    void free() {
        FinalRMinHash<T, Cmp>::free();
        std::vector<CountType> tmp;
        std::swap(tmp, second);
    }
    size_t countsum() const {return std::accumulate(second.begin(), second.end(), size_t(0), [](auto sz, auto sz2) {sz += sz2;});}
    double histogram_intersection(const FinalCRMinHash &o) const {
        assert(o.size() == this->size());
        const size_t lsz = this->size();
        size_t denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(this->cmp(this->first[i1], o.first[i2])) {
                denom += second[i1];
                I1DF;
            } else if(this->cmp(o.first[i2], this->first[i1])) {
                denom += o.second[i2];
                I2DF;
            } else {
                const auto v1 = o.second[i2], v2 = second[i1];
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1DF; I2DF;
            }
        }
        return static_cast<double>(num) / denom;
    }
    void read(const std::string &fn) {this->read(fn.data());}
    void read(const char *s) {
        gzFile fp = gzopen(s, "rb");
        if(!fp) throw 1;
        this->read(fp);
        gzclose(fp);
    }
    void read(gzFile fp) {
        uint64_t nelem;
        gzread(fp, &nelem, sizeof(nelem));
        this->first.resize(nelem);
        gzread(fp, this->first.data(), sizeof(this->first[0]) * nelem);
        this->second.resize(nelem);
        gzread(fp, this->second.data(), sizeof(this->second[0]) * nelem);
    }
    ssize_t write(const std::string &fn) const {return write(fn.data());}
    ssize_t write(const char *fn) const {
        gzFile fp = gzopen(fn, "wb");
        if(!fp) throw std::runtime_error("could not open file.");
        ssize_t ret = this->write(fp);
        gzclose(fp);
        return ret;
    }
    double union_size(const FinalCRMinHash &o) const {
        if(this->size() != o.size()) throw std::runtime_error("Non-matching parameters for FinalRMinHash comparison");
        return std::numeric_limits<T>::max() / double(std::max(this->max_element(), o.max_element())) * this->size();
    }
    ssize_t write(gzFile fp) const {
        uint64_t nelem = second.size();
        ssize_t ret = gzwrite(fp, &nelem, sizeof(nelem));
        ret += gzwrite(fp, this->first.data(), sizeof(this->first[0]) * nelem);
        ret += gzwrite(fp, this->second.data(), sizeof(this->second[0]) * nelem);
        return ret;
    }
    template<typename WeightFn=weight::EqualWeight>
    double tf_idf(const FinalCRMinHash &o, const WeightFn &fn=WeightFn()) const {
        assert(o.size() == this->size());
        const size_t lsz = this->size();
        double denom = 0, num = 0;
        for(size_t i1 = 0, i2 = 0;;) {
            if(this->cmp(this->first[i1], o.first[i2])) {
                denom += second[i1] * fn(this->first[i1]);
                I1DF;
            } else if(this->cmp(o.first[i2], this->first[i1])) {
                denom += o.second[i2] * fn(o.first[i2]);
                I2DF;
            } else {
                const auto v1 = second[i1] * fn(this->first[i1]), v2 = o.second[i2] * fn(o.first[i2]);
                denom += std::max(v1, v2);
                num += std::min(v1, v2);
                I1DF; I2DF;
            }
        }
#undef I1DF
#undef I2DF
        return num / denom;
    }
    FinalCRMinHash(std::vector<T> &&first, std::vector<CountType> &&second): FinalRMinHash<T, Cmp>(std::move(first)), second(std::move(second)) {
        if(this->first.size() != this->second.size()) {
            char buf[512];
            std::sprintf(buf, "Illegal FinalCRMinHash: hashes and counts must have equal length. Length one: %zu. Length two: %zu", this->first.size(), second.size());
            throw std::runtime_error(buf);
        }
    }
    template<typename Hasher>
    FinalCRMinHash(const CountingRangeMinHash<T, Cmp, Hasher, CountType> &prefinal): FinalRMinHash<T,Cmp>() {
        this->first.reserve(prefinal.size());
        this->second.reserve(prefinal.size());
        for(const auto &pair: prefinal)
            this->first.push_back(pair.first), this->second.push_back(pair.second);
        if(this->first.size() < prefinal.sketch_size()) {
            this->first.insert(this->first.end(), prefinal.sketch_size(), std::numeric_limits<T>::max());
            this->second.insert(this->second.end(), prefinal.sketch_size(), 0);
        }
    }
    template<typename Hasher>
    FinalCRMinHash(CountingRangeMinHash<T, Cmp, Hasher, CountType> &&prefinal): FinalCRMinHash(static_cast<const CountingRangeMinHash<T, Cmp, Hasher, CountType> &>(prefinal)) {
        prefinal.clear();
    }
    double cardinality_estimate(MHCardinalityMode mode=ARITHMETIC_MEAN) const {
        return FinalRMinHash<T, Cmp>::cardinality_estimate(mode);
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
        std::fprintf(stderr, "Pointer for data: %p\n", static_cast<void *>(core_.get()));
        common::detail::zero_memory(core_); // Second parameter is a dummy for interface compatibility with STL
#if !NDEBUG
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
        }
        return ComparePolicy::Manual;
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
    std::array<uint64_t, 64> sum_counts() const {
        using hll::detail::SIMDHolder;
        // TODO: this
        // Note: we have whip out more complicated vectorized maxes for
        // widths of 16, 32, or 64
        std::array<uint64_t, 64> ret{0};
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
                         lzt(hll::clz(((arr[0] << 1)|1) << (p_ - 1)) + 1);
        //std::fprintf(stderr, "Calling hash on thing. Size of core: %zu. Index: %zu\n", index, core_.size());
        //std::fprintf(stderr, "Calling hash on %zu\n", size_t(core_[index]));
        const uint64_t inserted_val = encode_register(lzt, reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval());
        //std::fprintf(stderr, "lzc: %d. oval: %zu. full val: %zu\n", int(get_lzc(inserted_val)), get_mhr(inserted_val), size_t(inserted_val));
        assert(get_lzc(inserted_val) == lzt);
        assert((reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) == get_mhr(inserted_val));
        //const uint64_t inserted_val = (reinterpret_cast<uint64_t *>(&hashval)[1] & max_mhval()) | (lzt << r_);
        //uint64_t oind = core_[index];
        //std::fprintf(stderr, "oind: %" PRIu64 "\n", oind);
        if(core_[index] < inserted_val) { // Consider other functions for specific register sizes.
            core_[index] = inserted_val;
            assert(encode_register(lzt, get_mhr(inserted_val)) == inserted_val);
            assert(lzt == get_lzc(inserted_val));
            //std::fprintf(stderr, "Register is now set to %zu with clz %d, which should match %d from other clz\n", size_t(core_[index]), int(get_lzc(inserted_val)), int(lzt));
        }
        //std::fprintf(stderr, "Register is now the same at%zu with clz %d, \n", size_t(core_[index]), int(get_lzc(core_[index])));
    }
    double jaccard_index(const HyperMinHash &o) const {
        size_t C = 0, N = 0;
        std::fprintf(stderr, "core size: %zu\n", core_.size());
        switch(simd_policy()) { // This can be accelerated for specific sizes
            default: //[[fallthrough]];
            U8:      //[[fallthrough]]; // 2-bit minimizers. TODO: write this
            U16:     //[[fallthrough]]; // 10-bit minimizers. TODO: write this
            U32:     //[[fallthrough]]; // 26-bit minimizers. TODO: write this
            U64:     //[[fallthrough]]; // 58-bit minimizers. TODO: write this
            Manual:
                for(size_t i = 0; i < core_.size(); ++i) {
                    C += core_[i] && (get_lzc(core_[i]) == get_lzc(o.core_[i]));
                    N += (core_[i] || o.core_[i]);
                }
            break;
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


} // namespace minhash
namespace mh = minhash;
} // namespace sketch
