#include <cstdint>
#include <map>
#include <vector>
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
    size_t total_ids_ = 0;
public:
    using key_type = KeyT;
    using id_type = IdT;
    size_t m() const {return m_;}
    size_t size() const {return total_ids_;}
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
    }
    template<typename IT, typename Alloc>
    SetSketchIndex(size_t m, const std::vector<IT, Alloc> &nperhashes): m_(m) {
        for(const auto v: nperhashes) {
            if(size_t(v) > m) throw std::invalid_argument("Cannot create LSH keys with v > m");
            regs_per_reg_.push_back(v);
            packed_maps_.emplace_back(HashV(m_ / v));
        }
    }
    SetSketchIndex(size_t m, bool densified=false): m_(m) {
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
    void update(const Sketch &item) {
        if(item.size() != m_) throw std::invalid_argument(std::string("Item has wrong size: ") + std::to_string(item.size()) + ", expected" + std::to_string(m_));
        const auto my_id = total_ids_++;
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
        const size_t nreg = regs_per_reg_.at(i);
        static constexpr size_t ITEMSIZE = sizeof(std::decay_t<decltype(item[0])>);
        if(nreg * packed_maps_[i].size() == m_)
            return XXH3_64bits(&item[nreg * j], nreg * ITEMSIZE);
        uint64_t seed = (i << 32) | j;
        XXH64_state_t state;
        XXH64_reset(&state, seed);
        const schism::Schismatic<uint32_t> div(m_);
        for(size_t ri = 0; ri < nreg; ++ri) {
            XXH64_update(&state, &item[div.mod(wyhash64_stateless(&seed))], ITEMSIZE);
        }
        return XXH64_digest(&state);
    }
    template<typename Sketch>
    std::pair<std::vector<IdT>, std::vector<uint32_t>>
    query_candidates(const Sketch &item, size_t maxcand, size_t starting_idx = size_t(-1)) const {
        if(starting_idx == size_t(-1)) starting_idx = regs_per_reg_.size();
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
            if(rset.size() >= maxcand) {
                break;
            }
        }
        return std::make_pair(passing_ids, items_per_row);
    }
};


#if 0
template<typename RegT, typename ValT=uint32_t>
struct BurstPrefixTrie {
    using Str = std::basic_string<RegT>;
    using reg_t = RegT;
    using creg_t = const RegT;
    struct Node;
    using CMap = std::map<Str, Node, std::less<>>;
    struct Node {
        CMap children_;
        std::vector<ValT> values_;
        Node() {}
        Node(ValT &&val): values_{{std::move(val)}} {}
        Node(const ValT &val): values_{val} {std::fprintf(stderr, "Copied\n");}
        bool is_leaf() const {return children_.empty();}
    };
    Node tree_;
    size_t sz_ = 0;
    BurstPrefixTrie() {
    };
    size_t size() const {return sz_;}
    static constexpr size_t nmatch(creg_t *lhs, creg_t *rhs, const size_t n) {
        for(size_t i = 0; i < n; ++i) if(lhs[i] != rhs[i]) return i;
        return n;
    }
    Node *tree() {return &tree_;}
    const Node *tree() const {return &tree_;}
    bool contained(Str s) const {
        const Node *ptr = tree();
        size_t nm = 0;
        for(size_t i = 1; i <= s.size(); ++i) {
            std::fprintf(stderr, "s: %s. i: %zu\n", s.data(), i);
            if(ptr->children_.find(s) != ptr->children_.end()) return true;
            auto su = s.substr(0, i);
            auto it = ptr->children_.lower_bound(su);
            if(it != ptr->children_.end() && su == it->first) {
                if(i == s.size()) return true;
                ptr = &it->second;
                i = 1;
                s = s.substr(su.size());
            } else if(it != ptr->children_.end()) {
                nm = i - 1;
                break;
            }
        }
        return false;
    }
    Node *insert(creg_t *keystart, creg_t *keystop, ValT &&val) {
        Node *ptr = tree();
        for(;;) {
            Str s(keystart, keystop);
            size_t nm = 0;
            if(s.empty()) {
                ptr->values_.push_back(std::move(val));
                ++sz_;
                return ptr;
            }
            typename CMap::iterator it;
            for(size_t i = 1; i < s.size(); ++i) {
                auto su = s.substr(0, i);
                it = ptr->children_.lower_bound(su);
                if(it == ptr->children_.end() || (su.size() <= it->first.size() && !std::equal(su.begin(), su.end(), it->first.begin())))
                    break;
                std::cerr << "lb for " << su << " is " << it->first << '\n';
                ++nm;
            }
            if(nm == 0) {
                if(s.empty()) {
                    std::fprintf(stderr, "Inserting empty string:\n");
                    ptr->values_.push_back(std::move(val));
                    ++sz_;
                    return ptr;
                }
                if(sizeof(RegT) == 1)
                    std::fprintf(stderr, "Inserting new string key with %zu match: %s\n", nm, s.data());
                else {
                    for(const auto i: s) std::fprintf(stderr, "%llu,", i);
                    std::fprintf(stderr, "Inserted string key:\n");
                }
                auto pit = ptr->children_.emplace(s, Node(val));
                ++sz_;
                std::fprintf(stderr, "emplaced string node %s: %zu\n", s.data(), sz_);
                return &pit.first->second;
            }
            if(nm == it->first.size()) {
                std::fprintf(stderr, "Matched full node, now moving own\n");
                ptr = &it->second;
                keystart += nm;
                continue;
            }
            std::fprintf(stderr, "matched prefix %s. Now splitting the node with existing matches\n", Str(s.data(), s.data() + nm).data());
            Node tmpnode = std::move(it->second);
            Node middle_node;
            std::pair<Str, Node> pair = std::move(*it);
            pair.first = Str(keystart, keystart + nm);
            auto nit = it, startit = nit;
            for(;nit != ptr->children_.end() && nmatch(pair.first.data(), nit->first.data(), std::min(nit->first.size(), nm)) == nm; ++nit) {
                auto &pair = *nit;
                std::fprintf(stderr, "Pair nm = %zu, %s/%zu with %zu values and %zu in dictionary\n", nm, pair.first.data(), pair.first.size(), pair.second.values_.size(), pair.second.children_.size());
                Str middlestr(pair.first.begin() + nm, pair.first.end());
                std::fprintf(stderr, "New middle node string: %s\n", middlestr.data());
                middle_node.children_.emplace(middlestr, pair.second);
            }
            ptr->children_.erase(startit, nit);
            pair.second = std::move(middle_node);
            ptr->children_.insert(pair);
            ptr = &pair.second;
            keystart += nm;
        }
    }
    void insert(const Str &s, ValT &&val) {insert(s.data(), s.data() + s.size(), std::move(val));}
    void print_all(const Node *ptr = nullptr, Str pref=Str()) const {
        if(ptr == nullptr) ptr = &tree_;
        auto eit = ptr->children_.find(Str());
        if(!ptr->values_.empty()) {
            for(const auto v: ptr->values_) {
                std::cerr << pref << ":" <<  v << '\n';
            }
        }
        for(const auto &pair: ptr->children_) {
            Str localstr(pref + pair.first);
            print_all(&pair.second, localstr);
        }
    }
};

template<typename RegT>
struct LSHForest {
    const size_t k_, l_;
    std::vector<RegT> data_;
    std::vector<uint64_t> ids_;
    uint64_t id = 0;
    LSHForest(size_t k, size_t l): k_(k), l_(l) {}
    LHSForest(size_t k, size_t l, const RegT *ptr, size_t nsketches, size_t step=0): LSHForest(k, l) {
        if(step == 0) step = k * l;
        if(step < k * l) throw std::invalid_argument("step must be <= k * l");
        data_.reserve(l * k * nsketches);
        ids_.reserve(nsketches);
        for(size_t i = 0; i < nsketches; add(ptr + step * i++));
    }
    void add(const RegT *start) {
        ids_.push_back(id++);
        data_.insert(data_.end(), start, start + l_ * k_);
    }
    static inline int veclt(const RegT *lhs, const RegT *rhs, size_t l) {
        for(size_t i = 0; i < l;++i)
            if(lhs[i] != rhs[i])
                return lhs[i] < rhs[i];
        return 0;
    }
    void sort() {
        std::vector<RegT> datacpy(data_.size());
        for(size_t i = 0; i < l_; ++i) {
            const size_t L = i;
            std::sort(ids_.begin(), ids_.end(), [l=l_](auto x, auto y) {
                return veclt(&data_[k_ * l_ * x + L * k_], &data_[k_ * l_ * y + L * k_], k_);
            });
            for(size_t k = 0; k < id; ++k) {
                auto start = &data_[k_ * l_ * k + L * k_], stop = start + k_;
                std::copy(start, stop, &datacpy[k_ * k + i * id * k_]);
            }
        }
        data_ = std::move(datacpy);
    }
    auto topk(const RegT *ptr) const {
        ska::flat_hash_set<uint64_t> ret;
    }
};
#endif

} // lsh

} // namespace sketch
