#ifndef SKETCH_SETSKETCH_INDEX_H__
#define SKETCH_SETSKETCH_INDEX_H__
#include <cstdint>
#include <map>
#include <vector>
#include <atomic>
#include <cstdio>
#include <iostream>
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include "sketch/div.h"
#include "sketch/integral.h"


namespace sketch {
using std::uint64_t;
using std::uint32_t;

namespace lsh {
static inline constexpr uint64_t _wymum(uint64_t x, uint64_t y) {
    __uint128_t l = x;
    l *= y;
    return l ^ (l >> 64);
}

// call wyhash64_seed before calling wyhash64
static inline constexpr uint64_t wyhash64_stateless(uint64_t *seed) {
  *seed += UINT64_C(0x60bee2bee120fc15);
  return _wymum(*seed ^ 0xe7037ed1a0b428dbull, *seed);
}


template<typename KeyT=uint64_t, typename IdT=uint32_t>
struct SetSketchIndex {
    /*
     * Maintains an LSH index over a set of sketches
     *
     */
private:
    size_t m_;
    using HashMap = ska::flat_hash_map<KeyT, std::vector<IdT>>;
    using HashV = std::vector<HashMap>;
    std::vector<HashV> packed_maps_;
    std::vector<uint64_t> regs_per_reg_;
    std::atomic<size_t> total_ids_;
    bool is_bottomk_only_ = false;
public:
    using key_type = KeyT;
    using id_type = IdT;
    size_t m() const {return m_;}
    size_t size() const {return total_ids_.load();}
    size_t ntables() const {return packed_maps_.size();}
    template<typename IT, typename Alloc, typename OIT, typename OAlloc>
    SetSketchIndex(size_t m, const std::vector<IT, Alloc> &nperhashes, const std::vector<OIT, OAlloc> &nperrows): m_(m) {
        if(nperhashes.size() != nperrows.size()) throw std::invalid_argument("SetSketchIndex requires nperrows and nperhashes have the same size");
        for(size_t i = 0, e = nperhashes.size(); i < e; ++i) {
            const IT v = nperhashes[i];
            regs_per_reg_.push_back(v);
            OIT v2 = nperrows[i];
            OIT v1 = m_ / v;
            if(v2 <= 0) v2 = v1;
            packed_maps_.emplace_back(v2);
        }
        total_ids_.store(0);
    }
    SetSketchIndex(): SetSketchIndex(1, std::vector<IdT>{1}) {
        packed_maps_.resize(1);
        packed_maps_.front().resize(1);
        regs_per_reg_ = {1};
        is_bottomk_only_ = true;
    }
    template<typename IT, typename Alloc>
    SetSketchIndex(size_t m, const std::vector<IT, Alloc> &nperhashes): m_(m) {
        total_ids_.store(0);
        for(const auto v: nperhashes) {
            regs_per_reg_.push_back(v);
            packed_maps_.emplace_back(HashV(m_ / v));
        }
    }
    SetSketchIndex(size_t m, bool densified=false): m_(m) {
        total_ids_.store(0);
        uint64_t rpr = 1;
        const size_t nrpr = densified ? m: size_t(ilog2(sketch::integral::roundup(m)));
        regs_per_reg_.reserve(nrpr);
        packed_maps_.reserve(nrpr);
        for(;rpr <= m_;) {
            regs_per_reg_.push_back(rpr);
            packed_maps_.emplace_back(HashV(m_ / rpr));
            if(densified) {
                ++rpr;
            } else {
                rpr <<= 1;
            }
        }
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>>
    update_query(const Sketch &item, size_t maxcand, size_t starting_idx = size_t(-1)) {
        if(item.size() < m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        if(starting_idx == size_t(-1) || starting_idx > regs_per_reg_.size()) starting_idx = regs_per_reg_.size();
        const size_t my_id = std::atomic_fetch_add(&total_ids_, size_t(1));
        const size_t n_subtable_lists = regs_per_reg_.size();
        ska::flat_hash_map<IdT, uint32_t> rset;
        std::vector<IdT> passing_ids;
        std::vector<uint32_t> items_per_row;
        rset.reserve(maxcand); passing_ids.reserve(maxcand); items_per_row.reserve(starting_idx);
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            const size_t nsubs = packed_maps_[i].size();
            for(size_t j = 0; j < nsubs; ++j) {
                auto &table = subtab[j];
                KeyT myhash = hash_index(item, i, j);
                auto it = table.find(myhash);
                if(it == table.end()) table.emplace(myhash, std::vector<IdT>{static_cast<IdT>(my_id)});
                else {
                    for(const auto id: it->second) {
                        auto rit2 = rset.find(id);
                        if(rit2 == rset.end()) {
                            rset.emplace(id, 1);
                            passing_ids.push_back(id);
                        } else ++rit2->second;
                    }
                    it->second.emplace_back(my_id);
                }
            }
        }
        std::vector<uint32_t> passing_counts(passing_ids.size());
        std::transform(passing_ids.begin(), passing_ids.end(), passing_counts.begin(), [&rset](auto x) {return rset[x];});
        return std::make_tuple(passing_ids, passing_counts, items_per_row);
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>> update_query_bottomk(const Sketch &item, size_t maxtoquery=-1) {
        std::map<IdT, uint32_t> matches;
        auto &map = packed_maps_.front().front();
        const size_t my_id = std::atomic_fetch_add(&total_ids_, size_t(1));
        for(const auto v: item) {
            auto it = map.find(v);
            if(it == map.end()) map.emplace(v, std::vector<IdT>{static_cast<IdT>(my_id)});
            else {
                for(const auto v: it->second) ++matches[v];
                it->second.emplace_back(my_id);
            }
        }
        std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>> ret;
        std::vector<std::pair<IdT, int32_t>> mvec(matches.begin(), matches.end());
        std::sort(mvec.begin(), mvec.end(), [](auto x, auto y) {return std::tie(x.second, x.first) > std::tie(y.second, y.first);});
        auto &first = std::get<0>(ret);
        auto &second = std::get<1>(ret);
        first.resize(matches.size());
        second.resize(matches.size());
        size_t i = 0;
        for(const auto &pair: mvec) first[i] = pair.first, second[i] = pair.second, ++i;
        if(first.size() > maxtoquery) {
            first.resize(maxtoquery);
            second.resize(maxtoquery);
        }
        return ret;
    }
    template<typename Sketch>
    void insert_bottomk(const Sketch &item, size_t my_id) {
        auto &map = packed_maps_.front().front();
        for(const auto v: item) {
            auto it = map.find(v);
            if(it == map.end()) {
                map.emplace(v, std::vector<IdT>{IdT(my_id)});
            } else it->second.emplace_back(my_id);
        }
    }
    template<typename Sketch>
    void update(const Sketch &item) {
        if(item.size() < m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        const size_t my_id = std::atomic_fetch_add(&total_ids_, size_t(1));
        if(is_bottomk_only_) {
            insert_bottomk(item, my_id);
            return;
        }
        const size_t n_subtable_lists = regs_per_reg_.size();
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            const size_t nsubs = subtab.size();
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = hash_index(item, i, j);
                subtab[j][myhash].push_back(my_id);
            }
        }
    }
    template<typename Sketch>
    KeyT hash_index(const Sketch &item, size_t i, size_t j) const {
        if(is_bottomk_only_) {
            return item[j];
        }
        const size_t nreg = regs_per_reg_[i];
        static constexpr size_t ITEMSIZE = sizeof(std::decay_t<decltype(item[0])>);
        if((j + 1) * nreg <= m_)
            return XXH3_64bits(&item[nreg * j], nreg * ITEMSIZE);
        uint64_t seed = ((i << 32) ^ (i >> 32)) | j;
        XXH64_state_t state;
        XXH64_reset(&state, seed);
        const schism::Schismatic<uint32_t> div(m_);
        for(size_t ri = 0; ri < nreg; ++ri)
            XXH64_update(&state, &item[div.mod(wyhash64_stateless(&seed))], ITEMSIZE);
        return XXH64_digest(&state);
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>>
    query_candidates(const Sketch &item, size_t maxcand, size_t starting_idx = size_t(-1)) const {
        if(starting_idx == size_t(-1) || starting_idx > regs_per_reg_.size()) starting_idx = regs_per_reg_.size();
        /*
         *  Returns ids matching input minhash sketches, in order from most specific/least sensitive
         *  to least specific/most sensitive
         *  Can be then used, along with sketches, to select nearest neighbors
         *  */
        ska::flat_hash_map<IdT, uint32_t> rset;
        std::vector<IdT> passing_ids;
        std::vector<uint32_t> items_per_row;
        rset.reserve(maxcand); passing_ids.reserve(maxcand); items_per_row.reserve(starting_idx);
        for(std::ptrdiff_t i = starting_idx;--i >= 0;) {
            auto &m = packed_maps_[i];
            const size_t nsubs = m.size();
            const size_t items_before = passing_ids.size();
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = hash_index(item, i, j);
                auto it = m[j].find(myhash);
                if(it == m[j].end()) continue;
                for(const auto id: it->second) {
                    auto rit2 = rset.find(id);
                    if(rit2 == rset.end()) {
                        rset.emplace(id, 1);
                        passing_ids.push_back(id);
                    } else ++rit2->second;
                }
            }
            items_per_row.push_back(passing_ids.size() - items_before);
            if(rset.size() >= maxcand) break;
        }
        std::vector<uint32_t> passing_counts(passing_ids.size());
        std::transform(passing_ids.begin(), passing_ids.end(), passing_counts.begin(), [&rset](auto x) {return rset[x];});
        return std::make_tuple(passing_ids, passing_counts, items_per_row);
    }
};


} // lsh


using lsh::SetSketchIndex;

} // namespace sketch

#endif /* #ifndef SKETCH_SETSKETCH_INDEX_H__ */
