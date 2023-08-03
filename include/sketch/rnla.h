#pragma once
#include "blaze/Math.h"
#include "blaze/util/Random.h"
#include "div.h"
#include "common.h"
#include "median.h"
#include <queue>
#include "macros.h"
#include "hash.h"
#ifndef NDEBUG
#  include <iostream>
#endif

namespace sketch {

inline namespace rnla {
enum Transform {
    CountSketch = 0,
    WoodruffZhang = 1
};

template<typename FT=float>
struct thresholded_cauchy_distribution {
    std::cauchy_distribution<FT> cd_;
    FT absmax_;
    template<typename...Args>
    thresholded_cauchy_distribution(FT absmax=30000., Args &&...args): cd_(std::forward<Args>(args)...), absmax_(std::abs(absmax))
    {
    }
    template<typename Gen>
    FT operator()(Gen &g) {
        FT ret;
        do ret = cd_(g); while(ret < -absmax_ || ret > absmax_);
        return ret;
    }
};

/***
   Utilities for compressing and decompressing
  */

//template<typename FT, bool SO>
//blaze::Dynamic

// TODO: write a generic method compatible with pytorch/ATen
//       (this actually does work, assuming we're working on the last level of a packedtensoraccessor), just not on GPU yet.
//       1. Random Laplace Transforms (Random Laplace Feature Maps for Semigroup Kernels on Histograms, Quasi-Monte Carlo Feature Maps for Shift-Invariant Kernels)
//       2. Tensor sketch: Count Sketch, but perform FFT on inputs and IFFT on outputs
//       3. Quasi Random Fourier Transform
//       4. Full matrix processing
//


template<typename C, typename C2, typename Hasher=KWiseHasherSet<4>>
auto &cs_compress(const C &in, C2 &ret, size_t newdim, const Hasher &hf) {
    using VT = std::decay_t<decltype(ret[0])>;
    if(newdim > in.size()) throw 1;
    std::fill(ret.begin(), ret.end(), static_cast<VT>(0));
    PREC_REQ(newdim <= in.size(), "newdim cannot be larger");
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    PREC_REQ(ret.size() == newdim * ns, "out size doesn't match parameters");
    for(unsigned j = 0; j < ns; ++j) {
        for(unsigned i = 0; i < in.size(); ++i) {
            const auto v = in[i];
            auto hv = hf(i, j);
            auto ind = div.mod(hv >> 1) * ns + j;
            auto upv = v * (hv & 1 ? 1: -1);
            ret[ind] += upv;
        }
    }
    return ret;
}

template<typename FT, typename C2, bool SO, typename Hasher=KWiseHasherSet<4>>
auto &cs_compress(const blaze::CompressedVector<FT, SO> &in, C2 &ret, size_t newdim, const Hasher &hf=Hasher()) {
    using VT = std::decay_t<decltype(ret[0])>;
    if(newdim > in.size()) throw 1;
    std::fill(ret.begin(), ret.end(), static_cast<VT>(0));
    PREC_REQ(newdim <= in.size(), "newdim cannot be larger");
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    PREC_REQ(ret.size() == newdim * ns, "out size doesn't match parameters");
    for(const auto &pair: in) {
        const auto idx = pair.index();
        const auto v = pair.value();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(idx, j);
            auto ind = div.mod(hv >> 1) * ns + j;
            const auto upv = v * (hv & 1 ? 1: -1);
            ret[ind] += upv;
        }
    }
    return ret;
}

template<typename C, typename Hasher=KWiseHasherSet<4>>
auto cs_compress(const C &in, size_t newdim, const Hasher &hf) {
    C ret(newdim * hf.size());
    cs_compress(in, ret, newdim, hf);
    return ret;
}

// Note: cs_compress could be a special case of wz_compress with Samplingist always returning 1
// or make it always a function of the remainder.

template<typename C, typename C2, typename Hasher=KWiseHasherSet<4>, typename SamplingDist=std::exponential_distribution<double>,
         typename RNG=blaze::DefaultRNG>
auto &wz_compress(const C &in, C2 &out, size_t newdim, const Hasher &hf, double p) {
    //using FT = std::decay_t<decltype(*std::begin(in))>;
    std::fill(out.begin(), out.end(), static_cast<std::decay_t<decltype(out[0])>>(0));
    if(newdim > in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    SamplingDist gen(p);
    for(unsigned j = 0; j < ns; ++j) {
        for(unsigned i = 0; i < in.size(); ++i) {
            const auto v = in[i];
            auto hv = hf(i, j);
            auto dm = div.divmod(hv);
            auto ind = dm.rem * ns + j;
            RNG rng(dm.quot >> 1);
            const double mult = gen(rng) * (dm.quot & 1 ? 1: -1) * v;
            out.operator[](ind) += mult;
        }
        // TODO: decompress.
        // Sample using the same seed, just multiply by inverse
    }
    return out;
}

template<typename FT, bool SO, typename C2, typename Hasher=KWiseHasherSet<4>, typename SamplingDist=std::exponential_distribution<double>,
         typename RNG=blaze::DefaultRNG>
auto &wz_compress(const blaze::CompressedVector<FT> &in, C2 &out, size_t newdim, const Hasher &hf, double p) {
    //using FT = std::decay_t<decltype(*std::begin(in))>;
    std::fill(out.begin(), out.end(), static_cast<std::decay_t<decltype(out[0])>>(0));
    if(newdim > in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(newdim);
    SamplingDist gen(p);
    for(const auto &p: in) {
        for(unsigned j = 0; j < ns; ++j) {
            const auto v = p.value();
            auto hv = hf(p.index(), j);
            auto dm = div.divmod(hv);
            auto ind = dm.rem * ns + j;
            RNG rng(dm.quot >> 1);
            const double mult = gen(rng) * (dm.quot & 1 ? 1: -1);
            out.operator[](ind) += v * mult;
        }
    }
    return out;
}

template<typename C, typename C2=C, typename Hasher=KWiseHasherSet<4>, typename RNG=blaze::DefaultRNG>
auto wz_compress(const C &in, size_t newdim, const Hasher &hf, double p) {
    C2 ret(newdim * hf.size());
    wz_compress(in, ret, newdim, hf, p);
    return ret;
}

template<typename C, typename OutC, typename Hasher=KWiseHasherSet<4>,
         typename=std::enable_if_t<!std::is_arithmetic<OutC>::value>,
         typename RNG=blaze::DefaultRNG>
auto &wz_decompress(const C &in, const Hasher &hf, OutC &ret, double p) {
    // TODO: some kind of importance sampling, weighting larger items more.
    PREC_REQ(in.size() % hf.size() == 0, "in dimension must be divisible by hf count");
    size_t olddim = in.size() / hf.size();
    size_t newdim = ret.size();
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    std::exponential_distribution<double> gen(p);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 9> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            auto dm = div.divmod(hv);
            RNG rng(dm.quot >> 1);
            tmp[j] = in[dm.rem * ns + j] / gen(rng) * (dm.quot & 1 ? 1: -1);
        }
        ret[i] = median(tmp, hf.size());
    }
    return ret;
}

template<typename C, typename OutC=C, typename Hasher=KWiseHasherSet<4>, typename RNG=blaze::DefaultRNG>
auto wz_decompress(const C &in, size_t outdim, const Hasher &hs, double p) {
    OutC ret(outdim);
    wz_decompress(in, hs, ret, p);
    return ret;
}

template<typename C, typename OutC, typename Hasher=KWiseHasherSet<4>,
         typename=std::enable_if_t<!std::is_arithmetic<Hasher>::value>>
auto &cs_decompress(const C &in, const Hasher &hf, OutC &ret) {
    PREC_REQ(in.size() % hf.size() == 0, "in dimension must be divisible by hf count");
    size_t olddim = in.size() / hf.size();
    size_t newdim = ret.size();
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    OMP_PRAGMA("omp parallel for")
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 9> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            tmp[j] = in[div.mod(hv >> 1) * ns + j] * (hv & 1 ? 1: -1);
        }
        ret[i] = median(tmp, hf.size());
    }
    return ret;
}

template<typename C, typename OutC=C, typename Hasher=KWiseHasherSet<4>>
auto cs_decompress(const C &in, size_t newdim, const Hasher &hf) {
    OutC ret(newdim);
    cs_decompress<C, OutC, Hasher>(in, hf, ret);
    return ret;
}

struct AbsMax {
    template<typename T>
    bool operator()(T x, T y) const {return std::abs(x) > std::abs(y);}
};

template<typename C, typename Functor=std::greater<void>>
auto top_indices_from_compressed(const C &in, size_t newdim, size_t olddim, const KWiseHasherSet<4> &hf, unsigned k) {
    //if(newdim < in.size()) throw 1;
    const size_t ns = hf.size();
    schism::Schismatic<uint32_t> div(olddim);
    using FT = std::decay_t<decltype(*std::begin(in))>;
    std::priority_queue<std::pair<FT, unsigned>, std::vector<std::pair<FT, unsigned>>, Functor> pq;
    OMP_PFOR
    for(size_t i = 0; i < newdim; ++i) {
        sketch::common::detail::tmpbuffer<float, 8> mem(hf.size());
        auto tmp = mem.get();
        for(unsigned j = 0; j < ns; ++j) {
            auto hv = hf(i, j);
            tmp[j] = in.operator[](div.mod(hv >> 1) * ns + j) * (hv & 1 ? 1: -1);
        }
        common::sort::insertion_sort(tmp, tmp + hf.size());
        FT med = median(tmp, hf.size());
        if(pq.size() < k || med > pq.top().first) {
            std::pair<FT, unsigned> pair(med, unsigned(i));
            OMP_CRITICAL
            {
                pq.push(pair);
                if(pq.size() > k) pq.pop();
            }
        }
    }
    std::pair<std::vector<FT>, std::vector<unsigned>> ret;
    ret.first.reserve(k);
    ret.second.reserve(k);
    for(unsigned i = 0; i < k; ++i) {
        const auto &r(pq.top());
        ret.first.push_back(r.first);
        ret.second.push_back(r.second);
        pq.pop();
    }
    return ret;
}

template<typename HasherSetType=XORSeedHasherSet<>>
struct SketchApplicator {
protected:
    size_t in_, out_;
    double p_;
    const Transform tx_;
    HasherSetType hs_;
public:
    SketchApplicator(size_t indim, size_t outdim, unsigned nh=3, uint64_t seed=13, double p=1., Transform tx=CountSketch):
        in_(indim), out_(outdim), p_(p), tx_(tx), hs_(nh, seed)
    {
        PREC_REQ(tx == CountSketch || tx == WoodruffZhang, "Unsupported");
    }
    template<typename C, typename C2>
    auto &compress(const C &in, C2 &out) const {
        assert(in_ == in.size());
        switch(tx_) {
            case CountSketch:
                return cs_compress(in, out, out_, hs_);
            case WoodruffZhang:
                return wz_compress(in, out, out_, hs_, p_);
            default: HEDLEY_UNREACHABLE();
        }
    }
    template<typename C, typename C2>
    auto &decompress(const C &in, C2 &out) const {
        assert(out_ * hs_.size() == in.size());
        switch(tx_) {
            case CountSketch:
                return cs_decompress(in, hs_, out);
            case WoodruffZhang:
                return wz_decompress(in, hs_, out, p_);
            default: HEDLEY_UNREACHABLE();
        }
    }
    template<typename C>
    auto decompress(const C &in) const {
        assert(out_ * hs_.size() == in.size());
        switch(tx_) {
            case CountSketch:
                return cs_decompress(in, out_, hs_);
            case WoodruffZhang:
                return wz_decompress(in, out_, hs_, p_);
            default: HEDLEY_UNREACHABLE();
        }
    }
    template<typename C>
    auto compress(const C &in) const {
        assert(in_ == in.size());
        switch(tx_) {
            case CountSketch:
                return cs_compress(in, out_, hs_);
            case WoodruffZhang:
                return wz_compress(in, out_, hs_, p_);
            default: HEDLEY_UNREACHABLE();
        }
    }
};

template<typename FloatType=float>
struct RNLASketcher {
    // nr: number of sketches/subtables
    // nc: dimension of sketches, e.g., dimensionality projected *to*
    auto       &mat()       & {return this->mat_;}
    const auto &mat() const & {return this->mat_;}
    auto && release() {return std::move(this->mat_);}
    auto ntables() const {return mat_.rows();}
    auto destdim() const {return mat_.columns();}
    blaze::DynamicMatrix<FloatType> mat_;
    RNLASketcher(size_t nr, size_t nc): mat_(nr, nc, static_cast<FloatType>(0)) {}
    // nr == number of subtables
};



template<typename FloatType=float, template<typename...> class DistType=thresholded_cauchy_distribution,
         typename Norm=blaze::L1Norm>
struct PStableSketcher: public RNLASketcher<FloatType> {
protected:
    using mtype = blaze::CompressedMatrix<FloatType>;
    using super = RNLASketcher<FloatType>;
    using this_type = PStableSketcher;
    std::unique_ptr<mtype> tx_;
    uint64_t seed_;
    DistType<FloatType> dist_;
    bool dense_;
    unsigned sourcedim_;
    Norm norm_;
public:
    using final_type = this_type;
    PStableSketcher(size_t ntables, size_t destdim, uint64_t sourcedim=0, uint64_t seed=137, bool dense=true, Norm &&norm=Norm(), DistType<FloatType> &&dist=DistType<FloatType>()):
        super(ntables, destdim), seed_(seed), dense_(dense),
        sourcedim_(sourcedim),
        norm_(std::move(norm))
    {
        init();
    }
    void init() {
        if(sourcedim_) {
            if(!dense_) throw 1;
            blaze::DefaultRNG gen(seed_);
            tx_.reset(new mtype(this->ntables() * this->destdim(), sourcedim_));
            auto &tx = *tx_;
            for(size_t sind = 0; sind < sourcedim_; ++sind) {
                for(size_t st = 0; st < this->ntables(); ++st) {
                    auto ind = gen() % this->destdim();
                    tx(st * this->destdim() + ind, sind) = dist_(gen);
                }
            }
            VERBOSE_ONLY(std::fprintf(stderr, "nonzeros: %zu. total: %zu\n", tx.nonZeros(), tx.rows() * tx.columns());)
        }
    }
    PStableSketcher(const PStableSketcher &o): super(o.ntables(), o.destdim()), seed_(o.seed_), dist_(o.dist_), dense_(o.dense_) {
        if(o.tx_) tx_.reset(new mtype(*o.tx_)); // manually call copy
    }
    PStableSketcher &operator-=(const PStableSketcher &o) {
        PREC_REQ(dense() == o.dense(), "must both be dense or not");
        PREC_REQ(seed_ == o.seed_, "must have the same seed");
        this->mat_ -= o.mat_;
        return *this;
    }
    PStableSketcher operator-(const PStableSketcher &o) {
        PStableSketcher ret = *this;
        ret -= o;
        return ret;
    }
    PStableSketcher &operator+=(const PStableSketcher &o) {
        PREC_REQ(dense() == o.dense(), "must both be dense or not");
        PREC_REQ(seed_ == o.seed_, "must have the same seed");
        this->mat_ += o.mat_;
        return *this;
    }
    PStableSketcher operator+(const PStableSketcher &o) {
        PStableSketcher ret = *this;
        ret += o;
        return ret;
    }
    double union_size(const PStableSketcher &o) const {
        PREC_REQ(o.mat_.rows() == this->mat_.rows(), "Mismatched row counts");
        PREC_REQ(o.mat_.columns() == this->mat_.columns(), "Mismatched row counts");
        std::vector<float> tmpvs(this->ntables());
        auto p = tmpvs.data();
        for(size_t i = 0; i < this->ntables(); ++i) {
            p[i] = norm(row(this->mat_, i) + row(o.mat_, i));
        }
        return median(p, this->ntables());
    }
    template<typename FT, template<typename, bool> class ContainerTemplate>
    void add(const ContainerTemplate<FT, blaze::columnVector> &x) {
        if(tx_.get()) {
            auto tmp = (*tx_) * x;
            for(size_t i = 0; i < this->ntables(); ++i) {
                row(this->mat_, i) += subvector(tmp, i * this->destdim(), this->destdim());
#if VERBOSE_AF
                std::cout << row(this->mat_, i) << '\n';
#endif
            }
            // tmp now has size k *
        } else {
            wy::WyHash<uint64_t, 0> vgen(seed_), indgen(vgen());
            for(size_t i = 0; i < this->ntables(); ++i) {
                auto r = row(this->mat_, i);
                for(size_t i = 0; i < x.size(); ++i) {
                    r[indgen() % this->destdim()] += dist_(vgen) * x[i];
                }
#if VERBOSE_AF
                std::cout << r << '\n';
#endif
            }
        }
    }
    template<typename FT, template<typename, bool> class ContainerTemplate>
    void add(const ContainerTemplate<FT, blaze::rowVector> &x) {
        if(tx_.get()) {
            auto tmp = (*tx_) * trans(x);
            for(size_t i = 0; i < this->ntables(); ++i) {
                row(this->mat_, i) += trans(subvector(tmp, i * this->destdim(), this->destdim()));
#if VERBOSE_AF
                std::cout << row(this->mat_, i) << '\n';
#endif
            }
            // tmp now has size k *
        } else {
            wy::WyHash<uint64_t, 0> vgen(seed_), indgen(vgen());
            for(size_t i = 0; i < this->ntables(); ++i) {
                auto r = row(this->mat_, i);
                for(size_t i = 0; i < x.size(); ++i) {
                    r[indgen() % this->destdim()] += dist_(vgen) * x[i];
                }
#if VERBOSE_AF
                std::cout << r << '\n';
#endif
            }
        }
    }
    auto pnorm() const {
        sketch::common::detail::tmpbuffer<FloatType> tmpvs(this->ntables());
        auto ptr = tmpvs.get();
        for(size_t i = 0; i < tmpvs.size(); ++i)
            ptr[i] = this->norm(row(this->mat_, i));
        return median(ptr, this->ntables());
    }
    std::vector<FloatType> pnorms() const {
        sketch::common::detail::tmpbuffer<FloatType> tmpvs(this->ntables());
        auto ptr = tmpvs.get();
        for(size_t i = 0; i < tmpvs.size(); ++i)
            ptr[i] = this->norm(row(this->mat_, i));
        common::sort::default_sort(ptr, ptr + this->ntables());
        return std::vector<FloatType>(ptr, ptr + this->ntables());
    }
    void addh(size_t hv, double w=1.) {
#if 1
        wy::WyHash<uint64_t, 0> gen(hv);
        for(size_t i = 0; i < this->ntables(); ++i) {
            auto r = row(this->mat_, i);
            assert(r.size() == this->destdim());
            auto v = gen();
            auto ind = v % this->destdim(), rem = v / this->destdim();
            wy::WyHash<uint64_t, 0> lrg(nosiam::wy61hash(rem, 4, 137));
            r[ind] += dist_(lrg) * w;
        }
#else
        blaze::CompressVector<FT, blaze::rowVector> cv;
        cv.reserfve(this->ntables());
#endif
    }
    bool dense() const {return dense_;}
    template<typename T>
    auto norm(const T &x) const {return norm_(x);}
    void clear() {
        this->mat_ = static_cast<FloatType>(0);
        init();
    }
};

template<typename FT>
struct IndykSketcher: public PStableSketcher<FT, thresholded_cauchy_distribution, blaze::L2Norm> {
    using super = PStableSketcher<FT, thresholded_cauchy_distribution, blaze::L2Norm>;
    template<typename...Args>
    IndykSketcher(Args &&...args): super(std::forward<Args>(args)...) {
    }
};

} // rnla

} // sketch
