#pragma once
#include "common.h" // This could be removed with copying or moving some items to a 


/* Currently incomplete! In the process of being drafted. Pieces will be taken from the bloom filter and hll implementations.
 * 
 *
*/
namespace cms {
using namespace common;

template<typename CountType, typename HashStruct=WangHash, typename Alloc=Alloctor<CountType>>
class cmsbase_t {
    std::vector<CounterType, Alloc> core_;
    HashStruct hf_;
    // TODO: Add constructor, add vectorized hash calculation and increments.
};

using cms_t = cmsbase_t<uint64_t, WangHash>;

} // namespace cms
