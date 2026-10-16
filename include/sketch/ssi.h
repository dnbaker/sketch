#ifndef SKETCH_SETSKETCH_INDEX_H__
#define SKETCH_SETSKETCH_INDEX_H__
#include <cstdint>
#include <map>
#include <vector>
#include <atomic>
#include <cstdio>
#include <zlib.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <utility>
#include "xxHash/xxh3.h"
#include "flat_hash_map/flat_hash_map.hpp"
#include "sketch/div.h"
#include "sketch/integral.h"
#include "sketch/hash.h"
#include "sketch/common.h"
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
    static void gzwrite_all_checked(gzFile fp, const void *ptr, size_t nb) {
        if(common::gzwrite_all(fp, ptr, nb) != ssize_t(nb)) throw ZlibError("Failed to write set sketch index to file");
    }
    static void gzread_all_checked(gzFile fp, void *ptr, size_t nb) {
        if(common::gzread_all(fp, ptr, nb) != ssize_t(nb)) throw ZlibError("Failed to read set sketch index from file");
    }
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
        gzwrite_all_checked(fp, &total_ids_, sizeof(total_ids_));
        size_t nms = packed_maps_.size();
        gzwrite_all_checked(fp, &nms, sizeof(nms));
        for(size_t i = 0; i < nms; ++i) {
            size_t v = packed_maps_[i].size();
            gzwrite_all_checked(fp, &v, sizeof(v));
        }
        gzwrite_all_checked(fp, regs_per_reg_.data(), regs_per_reg_.size() * sizeof(regs_per_reg_.front()));
        uint8_t ibk = is_bottomk_only_, islocked = !mutexes_.empty();
        gzwrite_all_checked(fp, &ibk, 1);
        gzwrite_all_checked(fp, &islocked, 1);
        for(size_t i = 0; i < packed_maps_.size(); ++i) {
            for(size_t j = 0; j < packed_maps_[i].size(); ++j) {
                auto &map = packed_maps_[i][j];
                uint64_t sz = map.size();
                gzwrite_all_checked(fp, &sz, sizeof(sz));
                for(auto &pair: map) {
                    uint64_t psz = pair.second.size();
                    gzwrite_all_checked(fp, &psz, sizeof(psz));
                    gzwrite_all_checked(fp, &pair.first, sizeof(pair.first));
                    gzwrite_all_checked(fp, pair.second.data(), sizeof(KeyT) * pair.second.size());
                }
            }
        }
    }
    SetSketchIndex(gzFile fp, bool clear=false) {
        size_t nmapsets;
        std::vector<size_t> mapsizes;
        regs_per_reg_.clear();

        gzread_all_checked(fp, &total_ids_, sizeof(total_ids_));
        gzread_all_checked(fp, &nmapsets, sizeof(size_t));
        mapsizes.reserve(nmapsets);
        packed_maps_.resize(mapsizes.size());
        while(mapsizes.size() < nmapsets) {
            size_t v;
            gzread_all_checked(fp, &v, sizeof(size_t));
            mapsizes.push_back(v);
            packed_maps_.emplace_back(HashV(v));
        }
        while(regs_per_reg_.size() < nmapsets) {
            size_t v;
            gzread_all_checked(fp, &v, sizeof(size_t));
            regs_per_reg_.push_back(v);
        }
        uint8_t ibk, islocked;
        gzread_all_checked(fp, &ibk, 1);
        gzread_all_checked(fp, &islocked, 1);
        is_bottomk_only_ = ibk;
        for(size_t i = 0; i < packed_maps_.size(); ++i) {
            for(size_t j = 0; j < packed_maps_[i].size(); ++j) {
                auto &map = packed_maps_[i][j];
                uint64_t sz;
                gzread_all_checked(fp, &sz, sizeof(sz));
                for(size_t k = 0; k < sz; ++k) {
                    uint64_t psz;
                    KeyT key;
                    gzread_all_checked(fp, &psz, sizeof(psz));
                    gzread_all_checked(fp, &key, sizeof(key));
                    std::vector<KeyT> vals(psz);
                    gzread_all_checked(fp, vals.data(), sizeof(KeyT) * vals.size());
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


// ---------------------------------------------------------------------------
// BandLSHIndex (round 14): LSH table variants over fixed-length signatures.
//
// A signature is a random-access sequence of m registers (e.g., the winning-id
// registers of pmh2_t / bmh_t, or any minhash-style signature). A band is a
// window of band_size consecutive registers; a band KEY is a seeded hash of the
// window's values, and ids are bucketed by key. Two signatures whose band
// values agree produce the same key and collide in the bucket.
//
// Variants exposed (see BandTier / constructor):
//  - Contiguous vs overlapping bands: stride < band_size overlaps windows.
//  - Composite keys (dashing2-style tiered retrieval): several tiers with
//    different band_sizes are indexed simultaneously; queries walk tiers from
//    most specific (largest band_size) to least, accumulating candidates.
//  - Single vs multi-table: ntables tables per tier. Tables keyed from ONE
//    shared signature have perfectly correlated band-match events (a band
//    match collides in every table), so insert()/query() with ntables > 1
//    buys redundancy, not recall. The classic independent multi-table
//    construction needs one signature per table: insert_mt()/query_mt().
//  - AND/OR composition: vote_threshold = 1 is OR (any band hit counts);
//    vote_threshold = nbands is AND (all bands in one table must hit);
//    intermediate values are fractional votes. Applied per table, unioned
//    across tables and tiers.
//  - Single-probe vs bounded multiprobe: mp_budget > 0 additionally indexes
//    and probes single-register-drop keys per band (probe p drops band offset
//    p-1, so budget B covers all differences in the first B band positions).
//    This is symmetric (insert and query enumerate the same probe keys) and is
//    the natural bounded multiprobe for signature-keyed tables: a neighbor
//    agreeing in >= band_size - 1 band registers is still found. Costs
//    (1 + B) keys per band of memory.
//
// Honest caveats: band-match probability is modeled as J^band_size (exact for
// independent minhash rows; for pmh2/bmh registers it is the probability that
// the winning ids agree, which for weighted sets is a likelihood-ratio event,
// not J_w^r -- see bench/lsh_pr_ablation.md phase B notes).
// ---------------------------------------------------------------------------

struct BandTier {
    size_t band_size = 4;    // registers hashed into one key
    size_t stride = 0;       // distance between band starts; 0 -> band_size (contiguous)
    size_t nbands = 0;       // bands per table; 0 -> derived: (siglen - band_size)/stride + 1
    unsigned mp_budget = 0;  // single-register-drop multiprobe keys per band (0 = single-probe)
};

template<typename KeyT=uint64_t, typename IdT=uint32_t>
struct BandLSHIndex {
    using key_type = KeyT;
    using id_type = IdT;
private:
    using BucketMap = ska::flat_hash_map<KeyT, std::vector<IdT>>;
    size_t siglen_;
    size_t ntables_;
    size_t vote_threshold_;
    std::vector<BandTier> tiers_;                  // configuration order
    std::vector<size_t> qorder_;                   // tier visitation order, most specific first
    std::vector<std::vector<std::vector<BucketMap>>> tables_; // [tier][table][band]
    uint64_t nkeys_ = 0;                           // total (key, id) insertions incl. multiprobe keys
    uint64_t nbuckets_ = 0;                        // distinct keys across all tables
    static constexpr uint64_t kSeedBase = 0x9E3779B97F4A7C15ULL;

    static INLINE uint64_t key_seed(size_t tier, size_t table, size_t band, unsigned probe) {
        uint64_t s = hash::WangHash::hash(kSeedBase ^ (tier + 0xD6E8FEB86659FD93ULL));
        s = hash::WangHash::hash(s ^ (table + 0xC2B2AE3D27D4EB4FULL));
        s = hash::WangHash::hash(s ^ (band  + 0x165667B19E3779F9ULL));
        return hash::WangHash::hash(s ^ (probe + 0x85EBCA77C2B2AE63ULL));
    }
    // Hash of sig[start, start + len), dropping offset `drop` (< len) if drop < len.
    template<typename Sig>
    static INLINE KeyT band_key(const Sig &sig, size_t start, size_t len, size_t drop, uint64_t seed) {
        uint64_t h = seed;
        for(size_t i = 0; i < len; ++i) {
            if(i == drop) continue;
            h = hash::WangHash::hash(h ^ (uint64_t(sig[start + i]) + kSeedBase * (i + 1)));
        }
        return KeyT(h);
    }
    size_t derived_nbands(const BandTier &t) const {
        const size_t stride = t.stride ? t.stride: t.band_size;
        if(t.band_size > siglen_ || stride == 0) return 0;
        return (siglen_ - t.band_size) / stride + 1;
    }
public:
    BandLSHIndex(size_t siglen, std::vector<BandTier> tiers, size_t ntables=1, size_t vote_threshold=1):
        siglen_(siglen), ntables_(ntables), vote_threshold_(vote_threshold), tiers_(std::move(tiers)) {
        if(!siglen_) throw std::invalid_argument("BandLSHIndex: empty signature");
        if(!ntables_) throw std::invalid_argument("BandLSHIndex: ntables must be >= 1");
        if(tiers_.empty()) throw std::invalid_argument("BandLSHIndex: need at least one tier");
        tables_.resize(tiers_.size());
        for(size_t i = 0; i < tiers_.size(); ++i) {
            auto &t = tiers_[i];
            if(!t.band_size || t.band_size > siglen_)
                throw std::invalid_argument("BandLSHIndex: band_size out of range");
            if(t.stride >= t.band_size && t.stride)
                throw std::invalid_argument("BandLSHIndex: stride must be < band_size (0 = contiguous)");
            if(t.stride == 0) t.stride = t.band_size;
            if(!t.nbands) t.nbands = derived_nbands(t);
            if(!t.nbands)
                throw std::invalid_argument("BandLSHIndex: tier has no bands");
            if(t.mp_budget > t.band_size) t.mp_budget = t.band_size;
            tables_[i].resize(ntables_);
            for(size_t l = 0; l < ntables_; ++l)
                tables_[i][l].resize(t.nbands);
        }
        // Query order: largest band_size first (most specific), stable.
        qorder_.resize(tiers_.size());
        std::iota(qorder_.begin(), qorder_.end(), size_t(0));
        std::stable_sort(qorder_.begin(), qorder_.end(), [this](size_t a, size_t b) {
            return tiers_[a].band_size > tiers_[b].band_size;
        });
        if(!vote_threshold_) vote_threshold_ = 1;
    }
    size_t siglen() const {return siglen_;}
    size_t ntables() const {return ntables_;}
    size_t vote_threshold() const {return vote_threshold_;}
    const std::vector<BandTier> &tiers() const {return tiers_;}
    size_t nbands(size_t tier) const {return tiers_[tier].nbands;}
    uint64_t n_keys() const {return nkeys_;}        // (key, id) insertions, incl. multiprobe keys
    uint64_t n_buckets() const {return nbuckets_;} // distinct keys
    uint64_t n_bands() const {
        uint64_t ret = 0;
        for(const auto &t: tiers_) ret += uint64_t(t.nbands) * ntables_;
        return ret;
    }
    // Rough resident-memory estimate: map buckets + per-bucket vector headers + ids.
    size_t memory_estimate() const {
        size_t bytes = nbuckets_ * (sizeof(KeyT) + 3 * sizeof(uint64_t)) + nkeys_ * sizeof(IdT);
        return bytes;
    }
    template<typename Sig>
    void insert_table(const Sig &sig, size_t l, IdT id) {
        if(size_t(sig.size()) < siglen_) throw std::invalid_argument("BandLSHIndex::insert: signature too short");
        for(size_t i = 0; i < tiers_.size(); ++i) {
            const auto &t = tiers_[i];
            const unsigned nprobes = 1u + t.mp_budget;
            for(size_t j = 0; j < t.nbands; ++j) {
                const size_t start = j * t.stride;
                for(unsigned p = 0; p < nprobes; ++p) {
                    const size_t drop = p ? p - 1: t.band_size; // p == 0 -> full band
                    const KeyT key = band_key(sig, start, t.band_size, drop, key_seed(i, l, j, p));
                    auto &vec = tables_[i][l][j][key];
                    if(vec.empty()) ++nbuckets_;
                    vec.push_back(id);
                    ++nkeys_;
                }
            }
        }
    }
public:
    // Insert with ONE signature broadcast to every table. Note: tables then
    // differ only in key seeds, and their band-match events are perfectly
    // correlated (a band match produces a key collision in every table), so
    // this multiplies memory without multiplying recall. For the classic
    // independent multi-table construction, build one signature per table
    // and use insert_mt.
    template<typename Sig>
    void insert(const Sig &sig, IdT id) {
        for(size_t l = 0; l < ntables_; ++l) insert_table(sig, l, id);
    }
    // Insert with one signature per table (sigs[l] feeds table l in every
    // tier): the classic independent multi-table LSH construction.
    template<typename SigVec>
    void insert_mt(const SigVec &sigs, IdT id) {
        if(size_t(sigs.size()) < ntables_) throw std::invalid_argument("BandLSHIndex::insert_mt: need one signature per table");
        for(size_t l = 0; l < ntables_; ++l) insert_table(sigs[l], l, id);
    }
private:
    template<typename Provider>
    std::vector<std::pair<IdT, uint32_t>> query_gen(Provider &&sig_for, size_t maxcand) const {
        ska::flat_hash_map<IdT, uint32_t> rset;
        ska::flat_hash_map<IdT, uint32_t> band_hits; // per (tier, table) vote counting
        const bool use_vote = vote_threshold_ > 1;
        auto flush_table = [&]() {
            if(!use_vote) return;
            for(auto &pair: band_hits) {
                if(pair.second < vote_threshold_) continue;
                auto it = rset.find(pair.first);
                if(it == rset.end()) rset.emplace(pair.first, pair.second);
                else it->second += pair.second;
            }
            band_hits.clear();
        };
        for(const size_t qi: qorder_) {
            const auto &t = tiers_[qi];
            const unsigned nprobes = 1u + t.mp_budget;
            for(size_t l = 0; l < ntables_; ++l) {
                const auto &sig = sig_for(l);
                if(size_t(sig.size()) < siglen_) throw std::invalid_argument("BandLSHIndex::query: signature too short");
                if(use_vote) band_hits.clear();
                for(size_t j = 0; j < t.nbands; ++j) {
                    const size_t start = j * t.stride;
                    if(nprobes > 1) {
                        // multiprobe: an id counts at most one hit per band
                        ska::flat_hash_set<IdT> seen;
                        for(unsigned p = 0; p < nprobes; ++p) {
                            const size_t drop = p ? p - 1: t.band_size;
                            const KeyT key = band_key(sig, start, t.band_size, drop, key_seed(qi, l, j, p));
                            auto it = tables_[qi][l][j].find(key);
                            if(it == tables_[qi][l][j].end()) continue;
                            for(const IdT id: it->second) {
                                if(!seen.insert(id).second) continue;
                                if(use_vote) ++band_hits[id];
                                else ++rset[id];
                                if(!use_vote && maxcand && rset.size() >= maxcand) goto done;
                            }
                        }
                    } else {
                        const KeyT key = band_key(sig, start, t.band_size, t.band_size, key_seed(qi, l, j, 0));
                        auto it = tables_[qi][l][j].find(key);
                        if(it == tables_[qi][l][j].end()) continue;
                        for(const IdT id: it->second) {
                            if(use_vote) ++band_hits[id];
                            else ++rset[id];
                            if(!use_vote && maxcand && rset.size() >= maxcand) goto done;
                        }
                    }
                }
                flush_table();
                if(use_vote && maxcand && rset.size() >= maxcand) goto done;
            }
        }
        done:;
        std::vector<std::pair<IdT, uint32_t>> ret(rset.begin(), rset.end());
        std::sort(ret.begin(), ret.end(), [](const auto &a, const auto &b) {
            return a.second != b.second ? a.second > b.second: a.first < b.first;
        });
        return ret;
    }
public:
    // Query with one signature broadcast to every table (see insert caveat).
    // Returns (id, hit count) candidates, hit count desc, id asc.
    // maxcand > 0 caps the number of distinct ids gathered (early stop, as in
    // SetSketchIndex::query_candidates); 0 gathers all.
    template<typename Sig>
    std::vector<std::pair<IdT, uint32_t>> query(const Sig &sig, size_t maxcand=0) const {
        return query_gen([&sig](size_t) -> const Sig &{return sig;}, maxcand);
    }
    // Query with one signature per table (sigs[l] probes table l), matching
    // insert_mt for independent multi-table constructions.
    template<typename SigVec>
    std::vector<std::pair<IdT, uint32_t>> query_mt(const SigVec &sigs, size_t maxcand=0) const {
        if(size_t(sigs.size()) < ntables_) throw std::invalid_argument("BandLSHIndex::query_mt: need one signature per table");
        return query_gen([&sigs](size_t l) -> const decltype(sigs[l]) &{return sigs[l];}, maxcand);
    }
};


// ---------------------------------------------------------------------------
// TieredProbeIndex (v2): lazy-stopping probing over band tiers.
//
// Tiers are band widths ordered most-specific -> most-sensitive (e.g.
// {8, 4, 2}). A query walks tiers in order, band by band, UNIONING bucket
// contents into a candidate set, and STOPS as soon as the accumulated
// distinct candidate count reaches oversample * k. Exact/sketch distances
// then run only on the accumulated set, so easy queries (tight keys already
// yield candidates) stop after a few specific probes while hard queries
// degrade the key until enough candidates appear. The stop point (tier,
// bands probed, candidates) is the measured behavior of interest.
// ---------------------------------------------------------------------------
struct TieredProbeIndex {
    struct ProbeStats {
        size_t bands_probed = 0;   // total bands probed across tiers
        size_t tier_reached = 0;   // tier index (0 = most specific) where the stop landed
        size_t candidates = 0;     // distinct candidates accumulated at the stop
    };
private:
    using BucketMap = ska::flat_hash_map<uint64_t, std::vector<uint32_t>>;
    size_t siglen_;
    std::vector<size_t> widths_;             // most-specific first
    std::vector<size_t> nbands_;
    std::vector<std::vector<BucketMap>> tables_; // [tier][band]
    uint64_t nkeys_ = 0, nbuckets_ = 0;
    static constexpr uint64_t kSeedBase = 0x9E3779B97F4A7C15ULL;
    static uint64_t key_seed(size_t tier, size_t band) {
        uint64_t s = hash::WangHash::hash(kSeedBase ^ (tier + 0xD6E8FEB86659FD93ULL));
        return hash::WangHash::hash(s ^ (band + 0x165667B19E3779F9ULL));
    }
    template<typename Sig>
    static uint64_t band_key(const Sig &sig, size_t start, size_t len, uint64_t seed) {
        uint64_t h = seed;
        for(size_t i = 0; i < len; ++i)
            h = hash::WangHash::hash(h ^ (uint64_t(sig[start + i]) + kSeedBase * (i + 1)));
        return h;
    }
public:
    TieredProbeIndex(size_t siglen, std::vector<size_t> widths):
        siglen_(siglen), widths_(std::move(widths)) {
        if(!siglen_) throw std::invalid_argument("TieredProbeIndex: empty signature");
        if(widths_.empty()) throw std::invalid_argument("TieredProbeIndex: need at least one tier");
        for(size_t i = 1; i < widths_.size(); ++i)
            if(widths_[i] > widths_[i - 1]) throw std::invalid_argument("TieredProbeIndex: widths must be most-specific first (descending)");
        tables_.resize(widths_.size());
        for(size_t t = 0; t < widths_.size(); ++t) {
            const size_t w = widths_[t];
            if(!w || w > siglen_) throw std::invalid_argument("TieredProbeIndex: tier width out of range");
            nbands_.push_back(siglen_ / w);
            tables_[t].resize(nbands_.back());
        }
    }
    size_t siglen() const {return siglen_;}
    size_t ntiers() const {return widths_.size();}
    size_t width(size_t tier) const {return widths_[tier];}
    uint64_t n_keys() const {return nkeys_;}
    uint64_t n_buckets() const {return nbuckets_;}
    size_t memory_estimate() const {
        return nbuckets_ * (sizeof(uint64_t) + 3 * sizeof(uint64_t)) + nkeys_ * sizeof(uint32_t);
    }
    template<typename Sig>
    void insert(const Sig &sig, uint32_t id) {
        for(size_t t = 0; t < widths_.size(); ++t) {
            const size_t w = widths_[t];
            for(size_t j = 0; j < nbands_[t]; ++j) {
                const uint64_t key = band_key(sig, j * w, w, key_seed(t, j));
                auto &vec = tables_[t][j][key];
                if(vec.empty()) ++nbuckets_;
                vec.push_back(id);
                ++nkeys_;
            }
        }
    }
    // Accumulate candidates until distinct count >= oversample * k (or all
    // tiers are exhausted), then return the accumulated ids with stop stats.
    template<typename Sig>
    std::vector<uint32_t> query(const Sig &sig, size_t k, double oversample, ProbeStats *stats=nullptr) const {
        const size_t stop_at = size_t(oversample * double(k) + 0.5);
        ska::flat_hash_map<uint32_t, uint32_t> seen;
        std::vector<uint32_t> ret;
        size_t bands = 0;
        size_t stop_tier = widths_.size() - 1;
        bool stopped = false;
        for(size_t t = 0; t < widths_.size() && !stopped; ++t) {
            const size_t w = widths_[t];
            for(size_t j = 0; j < nbands_[t]; ++j) {
                const uint64_t key = band_key(sig, j * w, w, key_seed(t, j));
                auto it = tables_[t][j].find(key);
                if(it != tables_[t][j].end()) {
                    for(const uint32_t id: it->second) {
                        if(seen.emplace(id, 1).second) ret.push_back(id);
                    }
                }
                ++bands;
                if(stop_at && ret.size() >= stop_at) {stop_tier = t; stopped = true; break;}
            }
        }
        if(stats) {
            stats->bands_probed = bands;
            stats->tier_reached = stop_tier;
            stats->candidates = ret.size();
        }
        return ret;
    }
};


} // lsh


using lsh::SetSketchIndex;
using lsh::BandLSHIndex;
using lsh::BandTier;
using lsh::TieredProbeIndex;

} // namespace sketch

#endif /* #ifndef SKETCH_SETSKETCH_INDEX_H__ */
