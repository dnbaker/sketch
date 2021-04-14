#include <cstdint>
#include <cstdio>
#include "flat_hash_map/flat_hash_map.hpp"


namespace sketch {
using std::uint64_t;
using std::uint32_t;

namespace lsh {


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
            if(v2 < 0) v2 = std::numeric_limits<OIT>::max();
            packed_maps_.emplace_back(std::min(v2, OIT(m_ / v)));
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
        using ResT = typename std::decay_t<decltype(item[0])>;
        for(size_t i = 0; i < n_subtable_lists; ++i) {
            auto &subtab = packed_maps_[i];
            const size_t nsubs = subtab.size();
            const size_t nelem = regs_per_reg_[i];
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = XXH3_64bits(&item[nelem * j], nelem * sizeof(ResT));
                subtab[j][myhash].push_back(my_id);
            }
        }
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
        using ResT = typename std::decay_t<decltype(item[0])>;
        ska::flat_hash_map<IdT, uint32_t> rset;
        rset.reserve(maxcand);
        std::vector<IdT> passing_ids;
        std::vector<uint32_t> items_per_row;
        for(std::ptrdiff_t i = starting_idx;--i >= 0;) {
            auto &m = packed_maps_[i];
            const size_t nelem = regs_per_reg_[i];
            const size_t nsubs = m.size();
            const size_t items_before = passing_ids.size();
            for(size_t j = 0; j < nsubs; ++j) {
                KeyT myhash = XXH3_64bits(&item[nelem * j], nelem * sizeof(ResT));
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
                std::fprintf(stderr, "Candidate set of size %zu, > maxc %zu\n", rset.size(), maxcand);
                break;
            }
        }
        return std::make_pair(passing_ids, items_per_row);
    }
};

#if 0
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
