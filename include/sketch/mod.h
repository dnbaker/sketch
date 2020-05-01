#ifndef MOD_SKETCH_H__
#include "flat_hash_map/flat_hash_map.hpp"
#include "./mh.h"
#include "./div.h"
#include "./hash.h"
#include "./policy.h"

namespace sketch {
inline namespace mod {

template<typename T, typename Allocator=std::allocator<T>>
struct FinalModHash;

template<typename HashStruct=hash::WangHash, typename VT=std::uint64_t, typename Policy=policy::SizePow2Policy<VT>, typename Allocator=std::allocator<VT>>
struct modsketch_t {
    Policy mod_;
    using final_type = FinalModHash<VT, Allocator>;
    ska::flat_hash_set<VT> set;
    HashStruct hs_;
    modsketch_t(VT mod, HashStruct &&hs=HashStruct()): mod_(mod), hs_(hs) {}
    template<typename...Args>
    bool addh(Args &&...args) {
        uint64_t v = hs_(std::forward<Args>(args)...);
        return add(v);
    }
    bool add(uint64_t el) {
        auto v = mod_.divmod(el);
        assert(v.quot * mod_.nelem() + v.rem == el);
        bool rv;
        if(v.rem == 0) {
            set.insert(v.quot);
            rv = true;
        } else rv = false;
        return rv;
    }
    final_type finalize() const {
        return final_type(mod_.nelem(), set.begin(), set.end());
    }
    modsketch_t reduce(VT factor) {
        if(factor < 1) throw std::invalid_argument("Have to increase, not decrease, factor");
        if(factor == 1) return *this;
        if(std::log2(factor) + std::log2(mod_.nelem()) > (sizeof(VT) * CHAR_BIT - std::is_signed<VT>::value)) {
            char buf[256];
            std::sprintf(buf, "Can't reduce from base %zu with factor %zu due to overflow\n", mod_.nelem(), size_t(factor));
            throw std::invalid_argument(buf);
        }
        modsketch_t ret(factor * mod_.nelem());
        if(is_pow2(factor)) {
            VT bitshift = integral::ilog2(factor);
            VT bitmask = (1ull << bitshift) - 1;
            for(const auto v: set)
                if(!(v & bitmask))
                    ret.set.insert(v >> bitshift);
        } else {
            Policy modnew(factor);
            for(const auto v: set) {
                auto dm = modnew.divmod(v);
                if(dm.rem == 0)
                    ret.set.insert(dm.quot);
            }
        }
        return ret;
    }
};

namespace func {

template<class ForwardIt, class UnaryPredicate, class Functor>
ForwardIt remove_if_else(ForwardIt first, ForwardIt last, UnaryPredicate p, Functor fct)
{
    first = std::find_if(first, last, p);
    if (first != last)
        for(ForwardIt i = first; ++i != last; )
            if (!p(*i))
                *first++ = std::move(fct(*i));
    return first;
}

}

template<typename T, typename Allocator>
struct FinalModHash: public minhash::FinalRMinHash<T, Allocator> {
    const T mod_;
    template<typename...Args>
    FinalModHash(T mod, Args &&...args): minhash::FinalRMinHash<T, Allocator>(std::forward<Args>(args)...), mod_(mod) {}
    double jaccard_index(const FinalModHash &o) const {
        PREC_REQ(mod_ == o.mod_, "Can't compare ModHashes of differing sizes");
        size_t isz = this->intersection_size(o);
        return double(isz) / (this->first.size() + o.size() - isz);
    }
    double containment_index(const FinalModHash &o) const {                                        
        return double(this->intersection_size(o)) / this->first.size();
    }
    double cardinality_estimate() const {
        return this->size() * mod_;
    }
    FinalModHash reduce(T factor) const {
        FinalModHash ret(*this);
        ret.reduce_inplace(factor);
        return ret;
    }
    FinalModHash &reduce_inplace(T factor) {
        if(factor < 1) throw std::invalid_argument("Have to increase, not decrease, factor");
        if(factor == 1) return *this;
        if(std::log2(factor) + std::log2(mod_) > (sizeof(T) * CHAR_BIT - std::is_signed<T>::value)) {
            char buf[256];
            std::sprintf(buf, "Can't reduce from base %zu with factor %zu due to overflow\n", mod_.d_, factor);
            throw std::invalid_argument(buf);
        }
        if(is_pow2(factor)) {
            T bitshift = integral::ilog2(factor);
            T bitmask = (1ull << bitshift) - 1;
            this->first.erase(func::remove_if_else(this->begin(), this->end(),
                    [bitmask](auto x) {return (x & bitmask) == 0;},
                    [bitshift](auto x) {return x >> bitshift;})
                , this->end()
            );
        } else {
            schism::Schismatic<T> modnew(factor);
            this->first.erase(func::remove_if_else(this->begin(), this->end(),
                [&](auto x) {
                    return modnew.mod(factor) == 0;
                },
                [&](auto x) {return modnew.div(x);}
            ), this->end());
        }
    }
};

} // mod
} // sketch

#endif
