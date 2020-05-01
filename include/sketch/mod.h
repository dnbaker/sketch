#ifndef MOD_SKETCH_H__
#include "flat_hash_map/flat_hash_map.hpp"
#include "./mh.h"
#include "./div.h"
#include "./hash.h"
#include "./policy.h"

namespace sketch {
inline namespace mod {

template<typename HashStruct=hash::WangHash, typename VT=std::uint64_t, typename Policy=policy::SizePow2Policy<VT>, typename Allocator=std::allocator<VT>>
struct modsketch_t {
    Policy mod_;
    using final_type = mh::FinalModHash<VT, Allocator>;
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
        return final_type(set.begin(), set.end());
    }
    modsketch_t reduce(VT factor) {
        if(factor < 1) throw std::invalid_argument("Have to increase, not decrease, factor");
        if(factor == 1) return *this;
        if(std::log2(factor) + std::log2(mod_.d_) > (sizeof(VT) * CHAR_BIT - std::is_signed<VT>::value)) {
            char buf[256];
            std::sprintf(buf, "Can't reduce from base %zu with factor %zu due to overflow\n", mod_.d_, factor);
            throw std::invalid_argument(buf);
        }
        Policy modnew(factor);
        modsketch_t ret(factor * mod_.nelem());
        if(is_pow2(factor)) {
            VT bitshift = integral::ilog2(factor);
            VT bitmask = (1ull << bitshift) - 1;
            for(const auto v: set)
                if(!(v & bitmask))
                    ret.set.insert(v >> bitshift);
        } else {
            for(const auto v: set) {
                auto dm = modnew.divmod(v);
                if(dm.rem == 0)
                    ret.set.insert(dm.quot);
            }
        }
        return ret;
    }
};

} // mod
} // sketch

#endif
