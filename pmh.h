#ifndef P_MINHASH_H__
#define P_MINHASH_H__
#include "blaze/Math.h"
#include "common.h"
#include "aesctr/wy.h"

namespace sketch {

namespace jp { // British Steel
using namespace common;

template<typename T>
void maxify(T &x);

template<typename Hasher=common::WangHash>
class PMinHasher {
    uint64_t *seeds_;
    size_t d_, n_;
    Hasher hf_;
public:
    template<typename... Args>
    PMinHasher(size_t dim, uint64_t nelem, uint64_t seed=137, Args &&...args): d_(dim), n_(nelem), hf_(std::forward<Args>(args)...) {
        if(posix_memalign((void **)&seeds_, sizeof(Space::VType), nelem * sizeof(*seeds_))) throw std::bad_alloc();
        DefaultRNGType rng(seed);
        std::for_each(seeds_, seeds_ + nelem, [&rng](uint64_t &x) {x = rng();});
    }
    PMinHasher(PMinHasher &&o): seeds_(o.seed_), d_(o.d_), n_(o.n_), hf_(std::move(o.hf_)) {
        o.seeds_ = nullptr; o.d_ = o.n_ = 0;
    }
    PMinHasher(const PMinHasher &o): seeds_(nullptr), d_(o.d_), n_(o.n_), hf_(o.hf_) {
        if(posix_memalign(&seeds_, sizeof(Space::VType), n_ * sizeof(*seeds_))) throw std::bad_alloc();
    }
    ~PMinHasher() {std::free(seeds_);}
    template<typename T, typename FType=double>
    auto operator()(T x, uint64_t seed) const {
        if(!x) return FType(0);
        uint64_t newseed;
        static_assert(sizeof(x) >= 4, "must be at least 4 bytes");
        seed ^= sizeof(x) == 8 ? *reinterpret_cast<uint64_t *>(&x): *reinterpret_cast<uint32_t *>(&x);
        wy::WyHash<uint64_t> rng(seed);
        std::uniform_real_distribution<FType> gen;
        return -std::log(gen(rng)) / x;
    }
    template<typename RAContainer, typename RetType=std::vector<uint32_t>, typename FType=double>
    auto hash(RAContainer &vec) const {
        if(vec.size() != d_) throw std::runtime_error("Wrong dimensions");
        using std::min;
        RetType ret(n_);
        for(auto &e: ret) e = n_; // To work with different containers
        std::vector<double> cvals;
        std::vector<uint32_t> nzs;
        for(size_t i = 0; i < vec.size(); ++i) {
            if(vec[i]) {
                nzs.push_back(i);
                cvals.resize(cvals.size() + n_);
                for(size_t j = 0; j < n_; ++j) {
                    cvals[(nzs.size() - 1) * n_ + j] = operator()(vec[i], seeds_[j]);
                }
            }
        }
        assert(cvals.size() == nzs.size() * n_);
        for(size_t i = 0; i < n_; ++i) {
            size_t minind = 0;
            FType minval = cvals[i];
            for(size_t j = 1; j < nzs.size(); ++j) {
                if(cvals[j * n_ + i] < minval) {
                    minval = cvals[j * n_ + i];
                    minind = j;
                }
            }
            ret[i] = nzs[minind];
        }
        return ret;
    }
};


} // Screaming For Vengeance

} // namespace sketch

#endif /* #ifndef P_MINHASH_H__ */
