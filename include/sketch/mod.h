#ifndef MOD_SKETCH_H__
#include "flat_hash_map/flat_hash_map.hpp"
#include "./mh.h"
#include "./div.h"
#include "./policy.h"

namespace sketch {
inline namespace mod {

template<typename VT=std::uint64_t, typename Policy=policy::SizePow2Policy<VT>>
struct modsketch_t {
    Policy mod_;
    using final_type = mh::FinalRMinHash<VT>;
    ska::flat_hash_set<VT> set;
    modsketch_t(VT mod): mod_(mod) {}
    bool addh(uint64_t el) {
        auto v = mod_.divmod(el);
        bool rv;
        if(v.rem == 0) {
            set.insert(el);
            rv = true;
        } else rv = false;
        return rv;
    }
    final_type finalize() const {
        return final_type(set.begin(), set.end());
    }
};

} // mod
} // sketch

#endif
