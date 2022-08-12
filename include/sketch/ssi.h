#ifndef SKETCH_SETSKETCH_INDEX_H__
#define SKETCH_SETSKETCH_INDEX_H__
#include <cstdint>
#include <map>
#include <vector>
#include <atomic>
#include <cstdio>
#include <zlib.h>
#include <iostream>
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include "sketch/div.h"
#include "sketch/integral.h"
#include "sketch/hash.h"
#include <mutex>
#include <optional>


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
    size_t total_ids_;
    std::vector<std::vector<std::mutex>> mutexes_;
    bool is_bottomk_only_ = false;
public:
    using key_type = KeyT;
    using id_type = IdT;
    size_t m() const {return m_;}
    size_t size() const {return total_ids_;}
    size_t size(size_t total_ids) {return total_ids_ = total_ids;}
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
            mutexes_.emplace_back(v2);
        }
        total_ids_ = 0;
    }
    void unlock() {
        mutexes_.clear();
    }
    template<typename IT, typename Alloc>
    SetSketchIndex(size_t m, const std::vector<IT, Alloc> &nperhashes): m_(m) {
        total_ids_ = 0;
        for(const auto v: nperhashes) {
            regs_per_reg_.push_back(v);
            packed_maps_.emplace_back(HashV(m_ / v));
            mutexes_.emplace_back(std::vector<std::mutex>(m_ / v));
        }
    }
    SetSketchIndex(size_t m, bool densified=false): m_(m) {
        total_ids_ = 0;
        uint64_t rpr = 1;
        const size_t nrpr = densified ? m: size_t(ilog2(sketch::integral::roundup(m)));
        regs_per_reg_.reserve(nrpr);
        packed_maps_.reserve(nrpr);
        for(;rpr <= m_;) {
            regs_per_reg_.push_back(rpr);
            const int64_t num_maps = m_ / rpr;
            packed_maps_.emplace_back(HashV(num_maps));
            mutexes_.emplace_back(std::vector<std::mutex>(num_maps));
            if(densified) {
                ++rpr;
            } else {
                rpr <<= 1;
            }
        }
    }

    SetSketchIndex &operator=(const SetSketchIndex &o) {
        total_ids_ = o.total_ids_;
        regs_per_reg_ = o.regs_per_reg_;
        packed_maps_ = o.packed_maps_;
        mutexes_.resize(o.mutexes_.size());
        for(size_t i = 0; i < o.mutexes_.size(); ++i) {
            mutexes_[i] = std::vector<std::mutex>(o.mutexes_[i].size());
        }
        is_bottomk_only_ = o.is_bottomk_only_;
        return *this;
    }
    SetSketchIndex(const SetSketchIndex &o) {*this = o;}
    bool operator==(const SetSketchIndex &o) {
        return total_ids_ == o.total_ids_ &&
            regs_per_reg_.size() == o.regs_per_reg_.size() &&
            packed_maps_.size() == o.packed_maps_.size() &&
            std::equal(packed_maps_.begin(), packed_maps_.end(), o.packed_maps_.begin());
    }

    SetSketchIndex &operator=(SetSketchIndex &&o) = default;
    SetSketchIndex(SetSketchIndex &&o) = default;
    SetSketchIndex(): SetSketchIndex(1, std::vector<IdT>{1}) {
        packed_maps_.resize(1);
        packed_maps_.front().resize(1);
        mutexes_.emplace_back(1);
        regs_per_reg_ = {1};
        is_bottomk_only_ = true;
    }
    static SetSketchIndex clone_like(const SetSketchIndex &o) {
        if(o.is_bottomk_only_)
            return SetSketchIndex();
        SetSketchIndex res;
        res.packed_maps_.resize(o.packed_maps_.size());
        res.regs_per_reg_ = o.regs_per_reg_;
        for(size_t i = 0; i < o.packed_maps_.size(); ++i) {
            res.packed_maps_[i].resize(o.packed_maps_[i].size());
        }
        res.mutexes_.clear();
        for(size_t i = 0; i < o.mutexes_.size(); ++i)
            res.mutexes_.emplace_back(o.mutexes_[i].size());
        res.is_bottomk_only_ = o.is_bottomk_only_;
        assert(res.is_bottomk_only_ == o.is_bottomk_only_);
        assert(res.mutexes_.size() == o.mutexes_.size() || !std::fprintf(stderr, "mutex sizes: %zu, %zu\n", res.mutexes_.size(), o.mutexes_.size()));
#ifndef NDEBUG
        for(size_t i = 0; i < res.mutexes_.size(); ++i) {
            assert(res.mutexes_[i].size() == o.mutexes_[i].size());
        }
#endif
        return res;
    }
    SetSketchIndex clone() const {
        return clone_like(*this);
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>>
    update_query(const Sketch &item, size_t maxcand, size_t starting_idx = size_t(-1)) {
        if(item.size() < m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        if(starting_idx == size_t(-1) || starting_idx > regs_per_reg_.size()) starting_idx = regs_per_reg_.size();
        const size_t my_id = std::atomic_fetch_add(reinterpret_cast<std::atomic<size_t> *>(&total_ids_), size_t(1));
        const size_t n_subtable_lists = regs_per_reg_.size();
        ska::flat_hash_map<IdT, uint32_t> rset;
        std::vector<IdT> passing_ids;
        std::vector<uint32_t> items_per_row;
        rset.reserve(maxcand); passing_ids.reserve(maxcand); items_per_row.reserve(starting_idx);
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            auto &submut = mutexes_[i];
            std::vector<std::mutex> *mptr = nullptr;
            if(mutexes_.size() > i) mptr = &mutexes_[i];
            const size_t nsubs = subtab.size();
            for(size_t j = 0; j < nsubs; ++j) {
                assert(j < subtab.size());
                auto &table = subtab[j];
                KeyT myhash = hash_index(item, i, j);
                std::optional<std::lock_guard<std::mutex>> lock(mptr ? std::optional<std::lock_guard<std::mutex>>((*mptr)[j]): std::optional<std::lock_guard<std::mutex>>());
                auto it = table.find(myhash);
                if(it == table.end()) {
                    table.emplace(myhash, std::vector<IdT>{static_cast<IdT>(my_id)});
                    //std::fprintf(stderr, "my hash %zu has a new key %zu\n", size_t(myhash), my_id);
                } else {
                    //std::fprintf(stderr, "my key %zu has %zu neighbors:", my_id, it->second.size());
                    for(const auto id: it->second) {
                        //std::fprintf(stderr, "%u now in use\t", id);
                        assert(id < total_ids_);
                        auto rit2 = rset.find(id);
                        if(rit2 == rset.end()) {
                            //std::fprintf(stderr, "ID %u is present now with new weight of 1\n");
                            rset.emplace(id, 1);
                            passing_ids.push_back(id);
                        } else {
                            assert(std::find(passing_ids.begin(), passing_ids.end(), id) != passing_ids.end());
                            ++rit2->second;
                            //std::fprintf(stderr, "ID %u has new count of %u\n", id, rit2->second);
                        }
                    }
                    it->second.emplace_back(my_id);
                }
            }
        }
        std::vector<uint32_t> passing_counts;
        if(passing_ids.size()) {
            passing_counts.resize(passing_ids.size());
            std::transform(passing_ids.begin(), passing_ids.end(), passing_counts.begin(), [&rset](auto x) {return rset[x];});
        }
        return std::make_tuple(passing_ids, passing_counts, items_per_row);
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>> update_query_bottomk(const Sketch &item, size_t maxtoquery=-1) {
        std::fprintf(stderr, "Warning: bottom-k update-query is untested\n");
        std::map<IdT, uint32_t> matches;
        auto &map = packed_maps_.front().front();
        const size_t my_id = std::atomic_fetch_add(reinterpret_cast<std::atomic<size_t> *>(&total_ids_), size_t(1));
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
        std::optional<std::lock_guard<std::mutex>> lock(
            !mutexes_.empty() && !mutexes_.front().empty()
            ? std::optional<std::lock_guard<std::mutex>>(mutexes_.front().front())
            : std::optional<std::lock_guard<std::mutex>>());
        for(const auto v: item) {
            auto it = map.find(v);
            if(it == map.end()) {
                map.emplace(v, std::vector<IdT>{IdT(my_id)});
            } else it->second.emplace_back(my_id);
        }
    }
    template<typename Sketch>
    size_t update_mt(const Sketch &item) {
        if(item.size() < m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        const size_t my_id = std::atomic_fetch_add(reinterpret_cast<std::atomic<size_t> *>(&total_ids_), size_t(1));
        if(is_bottomk_only_) {
            insert_bottomk(item, my_id);
            return my_id;
        }
        const size_t n_subtable_lists = regs_per_reg_.size();
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            std::vector<std::mutex> *mptr = nullptr;
            if(mutexes_.size() > i) mptr = &mutexes_[i];
            const size_t nsubs = subtab.size();
            OMP_PFOR
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = hash_index(item, i, j);
                auto &subsub = subtab[j];
                std::optional<std::lock_guard<std::mutex>> lock(mptr ? std::optional<std::lock_guard<std::mutex>>((*mptr)[j]): std::optional<std::lock_guard<std::mutex>>());
                auto it = subsub.find(myhash);
                if(it == subsub.end()) subsub.emplace(myhash, std::vector<IdT>{static_cast<IdT>(my_id)});
                else it->second.push_back(my_id);
            }
        }
        return my_id;
    }
    static constexpr size_t DEFAULT_ID = size_t(0xFFFFFFFFFFFFFFFF);
    template<typename Sketch>
    size_t update(const Sketch &item, size_t my_id = DEFAULT_ID) {
        if(item.size() < m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        if(my_id == DEFAULT_ID)
            my_id = std::atomic_fetch_add(reinterpret_cast<std::atomic<size_t> *>(&total_ids_), size_t(1));
        if(is_bottomk_only_) {
            insert_bottomk(item, my_id);
            return my_id;
        }
        const size_t n_subtable_lists = regs_per_reg_.size();
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            std::vector<std::mutex> *mptr = nullptr;
            if(mutexes_.size() > i) mptr = &mutexes_[i];
            const size_t nsubs = subtab.size();
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = hash_index(item, i, j);
                assert(j < subtab.size());
                std::optional<std::lock_guard<std::mutex>> lock(mptr ? std::optional<std::lock_guard<std::mutex>>((*mptr)[j]): std::optional<std::lock_guard<std::mutex>>());
                subtab[j][myhash].push_back(my_id);
            }
        }
        return my_id;
    }
    INLINE KeyT hashmem256(const uint64_t *x) const {
        sketch::hash::CEHasher ceh;
        uint64_t v[4];
        std::memcpy(&v, x, sizeof(v));
        return sketch::hash::WangHash::hash(ceh(v[0]) ^ (ceh(v[1]) * ceh(v[2]) - v[3]));
    }
    INLINE KeyT hashmem128(const uint64_t *x) const {
        uint64_t v[2];
        std::memcpy(&v, x, sizeof(v));
        v[0] = sketch::hash::WangHash::hash(v[0]);
        v[1] = sketch::hash::WangHash::hash(v[1] ^ v[0]);
        return v[0] ^ v[1];
    }
    INLINE KeyT hashmem64(const uint64_t *x) const {
        uint64_t v;
        std::memcpy(&v, x, sizeof(v));
        v = sketch::hash::WangHash::hash(v);
        return v;
    }
    INLINE KeyT hashmem32(const uint32_t *x) const {
        // MurMur3 finalizer
        uint32_t v;
        std::memcpy(&v, x, sizeof(v));
        v ^= v >> 16;
        v *= 0x85ebca6b;
        v ^= v >> 13;
        v *= 0xc2b2ae35;
        v ^= v >> 16;
        return v;
    }
    INLINE KeyT hashmem16(const uint16_t *x) const {
        uint32_t v = 0;
        std::memcpy(&v, x, sizeof(*x));
        v = ((v + 0x428eca6b) * 0x85ebca6b);
        v ^= v >> 16;
        return v;
    }
    INLINE KeyT hashmem8(const uint8_t *x) const {
        KeyT v = ((*x + 0x428eca6b) * 0x85ebca6b);
        v ^= v >> 16;
        return v;
    }
    template<typename T>
    INLINE KeyT hashmem(const T &x, size_t n) const {
        KeyT ret;
        switch(sizeof(T) * n) {
            case 1: ret =  hashmem8((const uint8_t *)&x); break;
            case 2: ret =  hashmem16((const uint16_t *)&x); break;
            case 4: ret =  hashmem32((const uint32_t *)&x); break;
            case 8: ret =  hashmem64((const uint64_t *)&x); break;
            case 16: ret =  hashmem128((const uint64_t *)&x); break;
            case 32: ret =  hashmem256((const uint64_t *)&x); break;
            default: ret = XXH3_64bits(&x, n * sizeof(T));
        }
        return ret;
    }
    template<typename Sketch>
    INLINE KeyT hash_index(const Sketch &item, size_t i, size_t j) const {
        if(is_bottomk_only_) {
            return item[j];
        }
        const size_t nreg = regs_per_reg_[i];
        static constexpr size_t ITEMSIZE = sizeof(std::decay_t<decltype(item[0])>);
        if((j + 1) * nreg <= m_) {
            return hashmem(item[nreg * j], nreg);
        }
        uint64_t seed = ((i << 32) ^ (i >> 32)) | j;
        XXH64_state_t state;
        XXH64_reset(&state, seed);
        const schism::Schismatic<uint32_t> div(m_);
#define SINGLE_UPDATE \
    XXH64_update(&state, &item[div.mod(wyhash64_stateless(&seed))], ITEMSIZE);
        for(size_t ri8 = nreg / 8;ri8--;) {
#define TWICE(X) X X
            TWICE(TWICE(TWICE(SINGLE_UPDATE)))
        }
        for(size_t ri = 0; ri < nreg; ++ri) SINGLE_UPDATE
#undef TWICE
#undef SINGLE_UPDATE
        return XXH64_digest(&state);
    }
    template<typename Sketch>
    std::tuple<std::vector<IdT>, std::vector<uint32_t>, std::vector<uint32_t>>
    query_candidates(const Sketch &item, size_t maxcand, size_t starting_idx = size_t(-1), bool early_stop=true) const {
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
        if(is_bottomk_only_) {
            auto &m = packed_maps_.front().front();
            for(size_t j = 0; j < item.size() && rset.size() < maxcand; ++j) {
                if(auto it = m.find(item[j]); it != m.end()) {
                    for(const auto id: it->second) {
                        auto rit2 = rset.find(id);
                        if(rit2 == rset.end()) {
                            rset.emplace(id, 1);
                            passing_ids.push_back(id);
                            if(early_stop && rset.size() == maxcand)
                                goto bk_end;
                        } else ++rit2->second;
                    }
                }
            }
            bk_end:
            items_per_row.push_back(passing_ids.size());
        } else {
            for(std::ptrdiff_t i = starting_idx;--i >= 0 && rset.size() < maxcand;) {
                auto &m = packed_maps_[i];
                const size_t nsubs = m.size();
                const size_t items_before = passing_ids.size();
                for(size_t j = 0; j < nsubs; ++j) {
                    KeyT myhash = hash_index(item, i, j);
                    auto it = m[j].find(myhash);
                    if(it != m[j].end()) {
                        for(const auto id: it->second) {
                            //auto rit2 = rset.find(id);
                            if(auto rit2 = rset.find(id); rit2 == rset.end()) {
                                rset.emplace(id, 1);
                                passing_ids.push_back(id);
                                if(early_stop && rset.size() == maxcand) {
                                    items_per_row.push_back(passing_ids.size() - items_before);
                                    goto end;
                                }
                            } else ++rit2->second;
                        }
                    }
                }
                items_per_row.push_back(passing_ids.size() - items_before);
            }
        }
        end:
        std::vector<uint32_t> passing_counts(passing_ids.size());
        std::transform(passing_ids.begin(), passing_ids.end(), passing_counts.begin(), [&rset](auto x) {return rset[x];});
        return std::make_tuple(passing_ids, passing_counts, items_per_row);
    }
    void write(std::string path) const {
        gzFile fp = gzopen(&path[0], "w");
        write(fp);
        gzclose(fp);
    }
    void write(gzFile fp) const {
        gzwrite(fp, &total_ids_, sizeof(total_ids_));
        size_t nms = packed_maps_.size();
        gzwrite(fp, &nms, sizeof(nms));
        for(size_t i = 0; i < nms; ++i) {
            size_t v = packed_maps_[i].size();
            gzwrite(fp, &v, sizeof(v));
        }
        gzwrite(fp, regs_per_reg_.data(), regs_per_reg_.size() * sizeof(regs_per_reg_.front()));
        uint8_t ibk = is_bottomk_only_, islocked = !mutexes_.empty();
        gzwrite(fp, &ibk, 1);
        gzwrite(fp, &islocked, 1);
        for(size_t i = 0; i < packed_maps_.size(); ++i) {
            for(size_t j = 0; j < packed_maps_[i].size(); ++j) {
                auto &map = packed_maps_[i][j];
                uint64_t sz = map.size();
                gzwrite(fp, &sz, sizeof(sz));
                for(auto &pair: map) {
                    uint64_t psz = pair.second.size();
                    gzwrite(fp, &psz, sizeof(psz));
                    gzwrite(fp, &pair.first, sizeof(pair.first));
                    gzwrite(fp, pair.second.data(), sizeof(KeyT) * pair.second.size());
                }
            }
        }
    }
    SetSketchIndex(gzFile fp, bool clear=false) {
        size_t nmapsets;
        std::vector<size_t> mapsizes;
        regs_per_reg_.clear();

        gzread(fp, &total_ids_, sizeof(total_ids_));
        gzread(fp, &nmapsets, sizeof(size_t));
        mapsizes.reserve(nmapsets);
        packed_maps_.resize(mapsizes.size());
        while(mapsizes.size() < nmapsets) {
            size_t v;
            gzread(fp, &v, sizeof(size_t));
            mapsizes.push_back(v);
            packed_maps_.emplace_back(HashV(v));
        }
        while(regs_per_reg_.size() < nmapsets) {
            size_t v;
            gzread(fp, &v, sizeof(size_t));
            regs_per_reg_.push_back(v);
        }
        uint8_t ibk, islocked;
        gzread(fp, &ibk, 1);
        gzread(fp, &islocked, 1);
        is_bottomk_only_ = ibk;
        for(size_t i = 0; i < packed_maps_.size(); ++i) {
            for(size_t j = 0; j < packed_maps_[i].size(); ++j) {
                auto &map = packed_maps_[i][j];
                uint64_t sz;
                gzread(fp, &sz, sizeof(sz));
                for(size_t k = 0; k < sz; ++k) {
                    uint64_t psz;
                    KeyT key;
                    gzread(fp, &psz, sizeof(psz));
                    gzread(fp, &key, sizeof(key));
                    std::vector<KeyT> vals(psz);
                    gzread(fp, vals.data(), sizeof(KeyT) * vals.size());
                    map.emplace(key, vals);
                }
            }
        }
        if(islocked) {
            mutexes_.resize(mapsizes.size());
            for(size_t i = 0; i < mapsizes.size(); ++i)
                mutexes_[i] = std::vector<std::mutex>(mapsizes[i]);
        }
        if(clear) gzclose(fp);
    }
    SetSketchIndex(std::string path): SetSketchIndex(gzopen(path.data(), "r"), true) {}
    void clear() {
        total_ids_ = 0;
        packed_maps_.clear();
        mutexes_.clear();
        regs_per_reg_.clear();
    }
};


} // lsh


using lsh::SetSketchIndex;

} // namespace sketch

#endif /* #ifndef SKETCH_SETSKETCH_INDEX_H__ */
