#pragma once
#include "hll.h" // This could be removed with copying or moving some items to a 


/* Currently incomplete! In the process of being drafted. Pieces will be taken from the bloom filter and hll implementations.
 * 
 *
*/
namespace cms {
using std::uint8_t;
using std::int8_t;
using std::uint16_t;
using std::int16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::int64_t;
using hll::WangHash;
using hll::MurFinHash;
using hll::Allocator;

template<typename CountType, typename HashStruct=WangHash, typename Alloc=Alloctor<CountType>>
class cmsbase_t {
    std::vector<CounterType, Alloc> core_;
    WangHash hf_;
    // TODO: Add constructor, add vectorized hash calculation and increments.
};

using cms_t = cmsbase_t<uint64_t, WangHash>;

} // namespace cms
