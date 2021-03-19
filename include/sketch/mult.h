#ifndef DNB_SKETCH_MULTIPLICITY_H__
#define DNB_SKETCH_MULTIPLICITY_H__
#ifndef NO_BLAZE
#  if VECTOR_WIDTH <= 32 || AVX512_REDUCE_OPERATIONS_ENABLED
#    include "blaze/Math.h"
#  endif
#endif
#include <random>
#include "ccm.h" // Count-min sketch

#include "hk.h"
#include <cstdarg>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include "hash.h"

namespace sketch {


namespace cws {

#if defined(_BLAZE_MATH_MATRIX_H_) && (VECTOR_WIDTH <= 32 || AVX512_REDUCE_OPERATIONS_ENABLED)
template<typename FType=float>
struct CWSamples {
    using MType = blaze::DynamicMatrix<float>;
    MType r_, c_, b_;
    CWSamples(size_t nhist, size_t histsz, uint64_t seed=0xB0BAFe77C001D00D): r_(nhist, histsz), c_(nhist, histsz), b_(nhist, histsz) {
        std::mt19937_64 mt(seed);
        std::gamma_distribution<FType> dist(2, 1);
        std::uniform_real_distribution<FType> rdist;
        for(size_t i = 0; i < nhist; ++i) {
            auto rr = row(r_, i);
            auto cr = row(c_, i);
            auto br = row(b_, i);
            for(size_t j = 0; j < histsz; ++j)
                rr[j] = dist(mt), cr[j] = dist(mt), br[j] = rdist(mt);
        }
    }
};
#endif

template<typename FType=float, typename HashStruct=hash::WangHash, size_t decay_interval=0, bool conservative=false>
class realccm_t: public cm::ccmbase_t<update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,conservative> {
    using super = cm::ccmbase_t<update::Increment,std::vector<FType, Allocator<FType>>,HashStruct,conservative>;
    using FSpace = typename vec::SIMDTypes<FType>;
    using super::seeds_;
    using super::data_;
    using super::nhashes_;
    using super::mask_;
    using super::subtbl_sz_;
    using super::l2sz_;
    using super::hash;
    static constexpr size_t rescale_frequency_ = 1ul << 12;
    static constexpr bool decay = decay_interval != 0;

    FType scale_, scale_inv_, scale_cur_;
    std::atomic<uint64_t> total_added_;
#if __cplusplus <= 201703L
    std::mutex mut_;
#else
    std::shared_mutex mut_;
#endif
public:
    FType decay_rate() const {return scale_;}
    void addh(uint64_t val, FType inc=1.) {this->add(val, inc);}
    template<typename...Args>
    realccm_t(FType scale_prod, Args &&...args): super(std::forward<Args>(args)...), scale_(scale_prod), scale_inv_(1./scale_prod), scale_cur_(scale_prod) {
        total_added_.store(0);
        assert(scale_ >= 0. && scale_ <= 1.);
    }
    realccm_t(): realccm_t(1.-1e-7) {}
    void rescale(size_t exp=rescale_frequency_) {
        const auto scale_div = std::pow(scale_, exp);
#ifndef NO_BLAZE
        blaze::CustomMatrix<FType, blaze::aligned, blaze::unpadded> tmp(this->data_.data(), this->nhashes_, size_t(1) << this->subtbl_sz_);
        assert(this->data_.size() == (this->nhashes_ << this->subtbl_sz_));
        assert(tmp.rows() * tmp.columns() == (this->data_.size()));
        tmp *= scale_div;
#else
        auto ptr = reinterpret_cast<typename FSpace::VType *>(this->data_.data());
        auto eptr = reinterpret_cast<typename FSpace::VType *>(this->data_.data() + this->data_.size());
        auto mul = FSpace::set1(scale_div);
        while(eptr > ptr) {
            *ptr = FSpace::mullo(ptr->simd_, mul);
            ++ptr;
        }
        FType *rptr = reinterpret_cast<FType *>(ptr);
        while(rptr < this->data_.data() + this->data_.size())
            *rptr++ *= scale_div;
#endif
    }
    void flush_rescaling() {
        size_t exp = total_added_ % rescale_frequency_;
        rescale(exp);
        total_added_ += rescale_frequency_ - exp;
    }
    FType add(const uint64_t val, FType inc) {
        ++total_added_; // I don't care about ordering, I just want it to be atomic.
        std::shared_lock<decltype(mut_)> sloc(mut_);
        CONST_IF(decay) {
            std::lock_guard<decltype(mut_)> lock(mut_);
            inc *= scale_cur_;
            scale_cur_ *= scale_inv_;
            if(total_added_ % rescale_frequency_ == 0u) { // Power of two, bitmask is efficient
                {
                    rescale();
                }
                scale_cur_ = scale_; // So when we multiply inc by scale_cur, the insertion happens at 1
            }
        }
        unsigned nhdone = 0, seedind = 0;
        const auto nperhash64 = lut::nhashesper64bitword[l2sz_];
        const auto nbitsperhash = l2sz_;
        using Type = vec::SIMDTypes<uint64_t>::Type;
        const Type *sptr = reinterpret_cast<const Type *>(seeds_.data());
        typename FSpace::VType vb = FSpace::set1(val), tmp;
        FType ret;
        CONST_IF(conservative) {
            std::vector<uint64_t> indices, best_indices;
            indices.reserve(nhashes_);
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(FSpace::COUNT * nperhash64))
                FSpace::VType(hash(FSpace::xor_fn(vb.simd_, FSpace::load(sptr++)))).for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64; indices.push_back(((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_));
                }),
                seedind += FSpace::COUNT;
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[seedind]);
                for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone); indices.push_back(((hv >> (k++ * nbitsperhash)) & mask_) + subtbl_sz_ * nhdone++));
                ++seedind;
            }
            auto fi = indices[0];
            best_indices.push_back(fi);
            auto minval = data_[fi];
            unsigned score;
            for(size_t i(1); i < indices.size(); ++i) {
                // This will change with
                if((score = data_[indices[i]]) == minval) {
                    best_indices.push_back(indices[i]);
                } else if(score < minval) {
                    best_indices.clear();
                    best_indices.push_back(indices[i]);
                    minval = score;
                }
            }
            ret = (data_[fi] += inc);
            for(size_t i = 1; i < best_indices.size() - 1; data_[best_indices[i++]] += inc);
            // This is likely a scatter/gather candidate, but they aren't particularly fast operations.
            // This could be more valuable on a GPU.
        } else { // not conservative update. This means we support deletions
            ret = std::numeric_limits<decltype(ret)>::max();
            while(static_cast<int>(nhashes_) - static_cast<int>(nhdone) >= static_cast<ssize_t>(FSpace::COUNT * nperhash64)) {
                FSpace::VType(hash(FSpace::xor_fn(vb.simd_, FSpace::load(sptr++)))).for_each([&](uint64_t subval) {
                    for(unsigned k(0); k < nperhash64;) {
                        auto ref = data_[((subval >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                        ref += inc;
                        ret = std::min(ret, double(ref));
                    }
                });
                seedind += FSpace::COUNT;
            }
            while(nhdone < nhashes_) {
                uint64_t hv = hash(val ^ seeds_[seedind++]);
                for(unsigned k(0); k < std::min(static_cast<unsigned>(nperhash64), nhashes_ - nhdone);) {
                    auto ref = data_[((hv >> (k++ * nbitsperhash)) & mask_) + nhdone++ * subtbl_sz_];
                    throw NotImplementedError("The updater should be here");
                    ret = std::min(ret, ssize_t(ref));
                }
            }
        }
        return ret;
    }
}; // realccm_t

} // namespace cws
namespace nt {
using hash::WangHash;
template<typename Container=std::vector<uint32_t, Allocator<uint32_t>>, typename HashStruct=WangHash, bool filter=true>
struct Card {
    // If using a different kind of counter than a native integer
    // simply define another numeric limits class providing max().
    // Ref: https://www.ncbi.nlm.nih.gov/pubmed/28453674
    using CounterType = std::decay_t<decltype(Container()[0])>;
    Container core_;
    const uint16_t p_, r_, pshift_;
    const CounterType maxcnt_;
    std::atomic<uint64_t> total_added_;
    HashStruct hf_;
    // Create a table of 2 << r entries
    // because the people who made this structure are weird
    // and use inconsistent notation
    template<typename...Args>
    Card(unsigned r, unsigned p, CounterType maxcnt, Args &&... args):
        core_(std::forward<Args>(args)...), p_(p), r_(r), pshift_(64 - p), maxcnt_(maxcnt) {
        total_added_.store(0);
#if VERBOSE_AF
        std::fprintf(stderr, "size of sketch: %zu\n", core_.size());
#endif
    }
    template<typename...Args>
    Card(unsigned r, unsigned p): Card(r, p, std::numeric_limits<CounterType>::max()) {}
    Card(Card &&o): core_(std::move(o.core_)), p_(o.p_), r_(o.r_), pshift_(o.pshift_), maxcnt_(o.maxcnt_), hf_(std::move(o.hf_)) {
        total_added_.store(o.total_added_.load());
    }
    void addh(uint64_t v) {
        v = hf_(v);
        add(v);
    }
    template<typename... Args>
    void set_hash(Args &&...args) {
        hf_ = std::move(HashStruct(std::forward<Args>(args)...));
    }
    size_t rbuck() const {
        return size_t(1) << r_; //
    }
    Card operator+(const Card &x) const {
        if(x.r_ != r_ || x.p_ != p_ || x.maxcnt_ != maxcnt_) {
            throw std::runtime_error("Parameter mismatch");
        }
        Card ret(r_, p_, maxcnt_, core_); // First, copy this container.
        ret += x;
        return ret;
    }
    Card &operator+=(const Card &x) {
        if(!std::is_same<Container, std::vector<CounterType, Allocator<CounterType>>>::value) {
            throw NotImplementedError("Haven't implemented merging of nthashes for any container but aligned std::vectors\n");
        }
        total_added_.store(total_added_.load() + x.total_added_.load());
        if(core_.size() * sizeof(core_[0]) >= sizeof(typename vec::SIMDTypes<CounterType>::VType)) {
            using VSpace = vec::SIMDTypes<CounterType>;
            using VType = typename vec::SIMDTypes<CounterType>::VType;
            VType *optr = reinterpret_cast<VType *>(this->core_.data());
            const VType *iptr = reinterpret_cast<const VType *>(x.core_.data());
            const VType *const eptr = reinterpret_cast<const VType *>(this->core_.data() + this->core_.size());
            assert(core_.size() % sizeof(VType) / sizeof(core_[0]) == 0);
            FOREVER {
                VSpace::store(reinterpret_cast<typename VSpace::Type *>(optr), VSpace::add(VSpace::load(reinterpret_cast<const typename VSpace::Type *>(optr)), VSpace::load(reinterpret_cast<const typename VSpace::Type *>(iptr))));
                ++iptr;
                ++optr;
                if(eptr == optr)
                    break;
            }
        } else for(size_t i = 0; i < core_.size(); core_[i] += x.core_[i], ++i);
        return *this;
    }
    void add(uint64_t v) {
        ++total_added_;
        const bool lastbit = v >> (pshift_ - 1) & 1;
        CONST_IF(filter) {
            if(v >> pshift_)
                return;
        }
        v <<= (64 - r_);
        v >>= (64 - r_);
        if(lastbit) v += (size_t(1) << r_);
        if(core_[v] != maxcnt_)
#ifndef NOT_THREADSAFE
            __sync_fetch_and_add(&core_[v], 1);
#else
            ++core_[v];
#endif
    }
    static constexpr double l2 = M_LN2;
    static constexpr size_t nsubs = 1ull << 16; // Why? The paper doesn't say and their code is weird.
    struct Deleter {
        template<typename T>
        void operator()(const T *x) const {
            std::free(const_cast<T *>(x));
        }
    };
    struct ResultType {
        std::vector<float> data_;
        size_t total;
        void report(std::FILE *fp=stderr) const {
            std::fprintf(fp, "maxcount=%zu,F1=%zu,F0=%f", data_.size() - 1, total, data_[0]);
            for(size_t i = 1; i < data_.size(); std::fprintf(fp, ",%f", data_[i++]));
        }
    };
#ifndef NDEBUG
#  define __vector_access operator[]
#else
#  define __vector_access at
#endif
    ResultType report() const {
        const CounterType max_val = *std::max_element(core_.begin(), core_.end()),
                          nvals = max_val + 1;
        std::vector<unsigned> arr(2 * nvals);
#if VERBOSE_AF
        std::fprintf(stderr,"Made arr with nvals = %zu\n", size_t(nvals));
#endif
        for(size_t i = 0; i < 2u; ++i) {
            size_t core_offset = i << r_;
            size_t arr_offset = nvals * i;
            for(size_t j = 0; j < size_t(1) << r_; ++j) {
                ++arr.__vector_access(core_.__vector_access(j + core_offset) + arr_offset);
            }
        }
#if VERBOSE_AF
        std::fprintf(stderr,"Filled arr with nvals = %zu\n", size_t(nvals));
#endif
        std::vector<double> pmeans(nvals);
        for(size_t i = 0; i < nvals; ++i) {
            pmeans[i] = (arr[i] + arr[i + nvals]) * .5;
        }
        //std::free(arr);
        std::vector<float> f_i(nvals);
#if VERBOSE_AF
        std::fprintf(stderr,"Made f_i arr\n");
#endif
        //if(!f_i) throw std::bad_alloc();
        double logpm0 = std::log(pmeans[0]);
        double lpmml2r = logpm0 - r_ * l2;
        f_i[0] = std::ldexp(-lpmml2r, p_ + r_); // F0 mean
        f_i[1]= -pmeans[1] / (pmeans[0] * (lpmml2r));
        for(size_t i = 2; i < nvals; ++i) {
            double sum=0.0;
            for(size_t j = 1; j < i; j++)
                sum += j * pmeans[i-j] * f_i[j];
            f_i[i] = -1.0*pmeans[i]/(pmeans[0]*(logpm0))-sum/(i*pmeans[0]);
        }
#undef __vector_access
        for(size_t i=1; i<nvals; f_i[i] = std::abs(f_i[i] * f_i[0]), ++i);
        return ResultType{std::move(f_i), total_added_.load()};
    }
}; // Card
template<typename CType, typename HashStruct=WangHash, bool filter=true>
struct VecCard: public Card<std::vector<CType, Allocator<CType>>, HashStruct, filter> {
    using super = Card<std::vector<CType, Allocator<CType>>, HashStruct, filter>;
    static_assert(std::is_integral<CType>::value, "Must be integral.");
    VecCard(unsigned r, unsigned p, CType max=std::numeric_limits<CType>::max()): super(r, p, max, 2ull << r) {} // 2 << r
};

} // namespace nt

namespace wj { // Weighted jaccard


struct WangPairHasher: public hash::WangHash {
    template<typename CType>
    static uint64_t hash(uint64_t x, CType count) {
        return hash::WangHash::hash(x) ^ count;
    }
    template<typename CType>
    uint64_t operator()(uint64_t x, CType count) const {return hash::WangHash::hash(x) ^ count;}
};

class ExactCountingAdapter {
    ska::flat_hash_map<uint64_t, uint32_t> data_;
public:
    ExactCountingAdapter(size_t rsz=1<<16) {
        data_.reserve(rsz);
    }
    template<typename...Args>
    ExactCountingAdapter(Args &&...) {} // Do nothing otherwise.
    void clear() {
        data_.clear();
    }
    uint64_t addh(uint64_t key) {
        typename ska::flat_hash_map<uint64_t, uint32_t>::iterator it;
        uint64_t ret;
        if((it = data_.find(key)) == data_.end()) {
            data_.emplace(key, 1);
            ret = 0;
        } else ret = it->second++;
        return ret;
    }
};


template<typename CoreSketch, typename CountingSketchType=hk::HeavyKeeper<32,32>, typename HashStruct=hash::WangHash, typename PairHasher=XXH3PairHasher>
struct WeightedSketcher {
    CountingSketchType cst_;
    CoreSketch      sketch_;
    HashStruct          hf_;
    PairHasher   pair_hasher_;
    public:
    using final_type = typename CoreSketch::final_type;
    using base_type  = CoreSketch;
    using cm_type    = CountingSketchType;
    using hash_type  = HashStruct;
    WeightedSketcher(CountingSketchType &&cst, CoreSketch &&core,
                     HashStruct &&hf=HashStruct())
    : cst_(std::move(cst)), sketch_(std::move(core)), hf_(std::move(hf)) {}
    WeightedSketcher(std::string path): cst_(0,0), sketch_(path) {throw NotImplementedError("Reading weighted sketcher from disk");}

    template<typename...Args>
    WeightedSketcher(const cm_type &tplt, HashStruct &&hf, Args &&...args):
        WeightedSketcher(std::move(cm_type(tplt)) /* copy constructor to make an r-value */,
                         CoreSketch(std::forward<Args>(args)...),
                         std::move(hf))
    {
    }
    operator final_type() {
        return final_type(std::move(sketch_));
    }
    operator final_type() const {
        return sketch_;
    }
    void addh(uint64_t x) {add(x);}
    template<typename CType>
    uint64_t hash(uint64_t x, CType count) const {return pair_hasher_.hash(x, count);}
    void add(uint64_t x) {
        auto count = cst_.addh(x);
        sketch_.addh(pair_hasher_.hash(x, std::max(count, static_cast<decltype(count)>(0))));
    }
    uint64_t hash(uint64_t x) const {return hf_(x);}
    WeightedSketcher(const WeightedSketcher &) = default;
    WeightedSketcher(WeightedSketcher &&)      = default;
    WeightedSketcher &operator=(const WeightedSketcher &) = default;
    WeightedSketcher &operator=(WeightedSketcher &&) = default;
    template<typename...Args>
    final_type cfinalize(Args &&...args) const {
        return sketch_.cfinalize(std::forward<Args>(args)...);
    }
    template<typename...Args>
    final_type finalize(Args &&...args) const {
#if VERBOSE_AF
        std::fprintf(stderr, "const finalize\n");
#endif
        return this->cfinalize();
    }
    template<typename...Args>
    final_type finalize(Args &&...args) {
#if VERBOSE_AF
        std::fprintf(stderr, "nonconst finalize\n");
#endif
        return sketch_.finalize(std::forward<Args>(args)...);
    }
    template<typename...Args>
    auto write(Args &&...args) const {return sketch_.write(std::forward<Args>(args)...);}
    template<typename...Args>
    void read(Args &&...args) {throw NotImplementedError("Reading weighted sketcher from disk");}
    auto jaccard_index(const base_type &o) const {return sketch_.jaccard_index(o);}
    auto jaccard_index(const WeightedSketcher &o) const {return sketch_.jaccard_index(o);}
    template<typename...Args> auto jaccard_index(Args &&...args) const {return sketch_.jaccard_index(std::forward<Args>(args)...);}
    template<typename...Args> auto containment_index(Args &&...args) const {return sketch_.containment_index(std::forward<Args>(args)...);}
    template<typename...Args> auto full_set_comparison(Args &&...args) const {return sketch_.full_set_comparison(std::forward<Args>(args)...);}
    auto containment_index(const base_type &o) const {return sketch_.containment_index(o);}
    template<typename...Args> auto free(Args &&...args) {return sketch_.free(std::forward<Args>(args)...);}
    template<typename...Args> auto cardinality_estimate(Args &&...args) const {return sketch_.cardinality_estimate(std::forward<Args>(args)...);}
    auto size() const {return sketch_.size();}
    void clear() {
        CoreSketch tmp(std::move(sketch_));
        CountingSketchType tmp2(std::move(cst_));
    }
    void reset() {
        sketch_.reset();
        cst_.clear();
    }
};


template<typename CoreSketch, typename F, typename CountingSketchType=hk::HeavyKeeper<32,32>, typename HashStruct=hash::WangHash, typename PairHasher=WangPairHasher>
struct FWeightedSketcher: public WeightedSketcher<CoreSketch, CountingSketchType, HashStruct, PairHasher> {
    const F func_;
    template<typename... Args>
    FWeightedSketcher(Args &&...args):
        WeightedSketcher<CoreSketch, CountingSketchType, HashStruct, PairHasher>(std::forward<Args>(args)...),
        func_() {}
    template<typename... Args>
    FWeightedSketcher(F &&func, Args &&...args): WeightedSketcher<CoreSketch, CountingSketchType, HashStruct, PairHasher>(std::forward<Args>(args)...),
        func_(std::move(func)) {}
    void add(uint64_t x) {
        auto count = this->cst_.addh(x);
        DBG_ONLY(std::fprintf(stderr, "taking %zu to turn into %f\n", size_t(count), double(func_(count)));)
        this->sketch_.addh(this->hash(x, func_(std::max(count, static_cast<decltype(count)>(0)))));
    }
};

namespace weight_fn {
struct SqrtFn {
    template<typename T>
    T operator()(T x) const {
        x = std::sqrt(x);
        return x;
    }
};
struct NLogFn {
    template<typename T>
    T operator()(T x) const {
        x = std::log(x);
        return x;
    }
};
struct LogFn {
    const double v_;
    LogFn(double v): v_(1./std::log(v)) {}
    template<typename T>
    T operator()(T x) const {
        x = std::log(x) * v_;
        return x;
    }
};
} // weight_fn



template<typename T>
struct is_weighted_sketch: public std::false_type {};

template<typename CoreSketch,
         typename CountingSketch,
         typename HashStruct>
struct is_weighted_sketch<WeightedSketcher<CoreSketch, CountingSketch, HashStruct>>: public std::true_type {};

template<typename CoreSketch,
         typename F,
         typename CountingSketch,
         typename HashStruct>
struct is_weighted_sketch<FWeightedSketcher<CoreSketch, F, CountingSketch, HashStruct>>: public std::true_type {};

} // namespace wj

} // namespace sketch


#endif /* DNB_SKETCH_MULTIPLICITY_H__ */
