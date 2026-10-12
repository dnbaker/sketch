// Flat compact binary LOUDS trie over frozen sorted keys.
// Level-order single array (the Eytzinger-family layout per the layout
// guidance), SoA arrays with no pointer chasing: fixed node records
// (skip length + child presence + first-child), skip bits packed in one
// global bitvector, leaves as (fp-truncated suffix, count). Subtree =
// contiguous node range: bands are slices, recursive intersection
// prunes whole ranges. Built from the sorted freeze output in one
// monotone pass (sorted-range splits; each key consumed once per
// level). Binary fanout per the Khuong-Morin reasoning: implicit
// level-order beats wide nodes in-cache; hashed keys have no dense
// prefixes for a 256-ary map to exploit.
#ifndef SKETCH_BCQF_TRIE_H__
#define SKETCH_BCQF_TRIE_H__
#include "cqf.h"
#include "bcqf_frozen.h"
#include <vector>
#include <algorithm>
#include <cstdint>

namespace sketch {
inline namespace cqf {

template<size_t KEY_BITS, size_t VAL_BITS, size_t FPW = 16>
struct compact_trie {
    static constexpr size_t SUFFIX = FPW;
    static constexpr uint64_t SUFMASK = (uint64_t(1) << SUFFIX) - 1;
    struct node {
        uint64_t skip : 8;      // shared leading bits at this node
        uint64_t child_bits : 2; // bit0 = 0-child, bit1 = 1-child
        uint64_t first_child : 26;
        uint64_t leaf_begin : 28; // leaves [leaf_begin, leaf_end) if no children
    };
    std::vector<node> nodes_;     // level order, root at 0
    std::vector<uint64_t> skipbits_; // packed per-node skips, level order
    std::vector<uint64_t> leaf_suf_; // sorted within parent
    std::vector<uint16_t> leaf_cnt_;
    uint64_t nkeys_ = 0, nbits_skip_ = 0;
public:
    template<typename It>
    void build(It first, It last) {
        const size_t n = size_t(std::distance(first, last));
        if(!n) return;
        // Pass 1: DFS over sorted ranges with common-prefix skipping.
        // Each interior range emits a node; children recorded by index.
        struct rec { uint64_t skip; uint8_t skip_len, child_bits; int child[2]; uint32_t leaf_begin, leaf_end; };
        std::vector<rec> rs;
        struct ctx { size_t lo, hi; uint64_t depth; int my; int slot; };
        std::vector<ctx> st;
        st.push_back({0, n, 0, -1, -1});
        std::vector<int> children_flat; // (parent, slot, child) triples flattened
        while(!st.empty()) {
            const ctx t = st.back(); st.pop_back();
            if(t.hi - t.lo <= 1) { // leaf pre-node
                rs.push_back(rec{0, 0, 0, {-1, -1}, uint32_t(t.lo), uint32_t(t.hi)});
                if(t.my >= 0) children_flat.insert(children_flat.end(), {t.my, t.slot, int(rs.size()) - 1});
                continue;
            }
            // skip common bits from t.depth
            const uint64_t firstv = uint64_t((first + t.lo)->first);
            uint64_t d = t.depth, skip = 0;
            while(d < KEY_BITS) {
                const uint64_t bit = uint64_t(1) << (KEY_BITS - 1 - d);
                size_t k = t.lo + 1;
                for(; k < t.hi; ++k) if((uint64_t((first + k)->first) & bit) != (firstv & bit)) break;
                if(k < t.hi) break;
                skip = (skip << 1) | ((firstv >> (KEY_BITS - 1 - d)) & 1);
                ++d;
            }
            const uint64_t bit = uint64_t(1) << (KEY_BITS - 1 - d);
            size_t split = t.lo + 1;
            for(; split < t.hi; ++split) if(uint64_t((first + split)->first) & bit) break;
            rs.push_back(rec{skip, uint8_t(d - t.depth), 0, {-1, -1}, 0, 0});
            const int my = int(rs.size()) - 1;
            if(t.my >= 0) children_flat.insert(children_flat.end(), {t.my, t.slot, my});
            // push right then left so children resolve left-to-right
            st.push_back({split, t.hi, d + 1, my, 1});
            st.push_back({t.lo, split, d + 1, my, 0});
        }
        for(size_t i = 0; i + 2 < children_flat.size(); i += 3)
            rs[children_flat[i]].child[children_flat[i + 1]] = children_flat[i + 2];
        // Pass 2: BFS relayout into level-order flat arrays.
        nodes_.clear();
        std::vector<uint64_t> bfs_skip;
        {
            std::vector<int> q(1, int(rs.size()) - 1); // root = last pushed interior (slot -1 parent)
            // root is the record with parent -1: find it
            q[0] = -1;
            for(size_t i = 0; i < rs.size(); ++i) { (void)i; }
            // root = the only record whose (my,slot) was (-1,-1): it is the first interior pushed
            q[0] = 0;
            // locate actual root: record referenced by no parent
            std::vector<char> has_parent(rs.size(), 0);
            for(size_t i = 0; i + 2 < children_flat.size(); i += 3) has_parent[children_flat[i + 2]] = 1;
            for(size_t i = 0; i < rs.size(); ++i) if(!has_parent[i]) { q[0] = int(i); break; }
            size_t head = 0;
            uint32_t next_child = 1;
            while(head < q.size()) {
                const int r = q[head++];
                node nn;
                nn.skip = rs[r].skip_len;
                nn.child_bits = 0;
                nn.first_child = next_child;
                nn.leaf_begin = rs[r].leaf_begin;
                if(rs[r].child[0] >= 0 || rs[r].child[1] >= 0) {
                    if(rs[r].child[0] >= 0) nn.child_bits |= 1;
                    if(rs[r].child[1] >= 0) nn.child_bits |= 2;
                    next_child += __builtin_popcount(nn.child_bits);
                    if(rs[r].child[0] >= 0) q.push_back(rs[r].child[0]);
                    if(rs[r].child[1] >= 0) q.push_back(rs[r].child[1]);
                } else {
                    // leaf node: leaf_begin/end name the (single) entry
                    nn.leaf_begin = rs[r].leaf_begin;
                }
                bfs_skip.push_back(rs[r].skip);
                nodes_.push_back(nn);
            }
        }
        // pack skips level-order
        uint64_t total_bits = 0;
        for(const auto &nn: nodes_) total_bits += nn.skip;
        nbits_skip_ = total_bits;
        skipbits_.assign(total_bits / 64 + 1, 0);
        uint64_t wpos = 0;
        for(size_t i = 0; i < nodes_.size(); ++i) {
            const uint64_t sk = bfs_skip[i], len = nodes_[i].skip;
            for(uint64_t b = 0; b < len; ++b) {
                if((sk >> (len - 1 - b)) & 1) skipbits_[wpos >> 6] |= uint64_t(1) << (wpos & 63);
                ++wpos;
            }
        }
        // leaves from the sorted input (single-entry pres in order)
        for(size_t i = 0; i < n; ++i) {
            leaf_suf_.push_back(uint64_t((first + i)->first) & SUFMASK);
            leaf_cnt_.push_back(uint16_t((first + i)->second));
        }
        nkeys_ = n;
    }
    uint64_t size() const {return nkeys_;}
    uint64_t nodes() const {return nodes_.size();}
    size_t memory() const {
        return nodes_.size() * sizeof(node) + skipbits_.size() * 8
             + leaf_suf_.size() * 8 + leaf_cnt_.size() * 2;
    }
    // bits consumed per stored key (trie total, leaf suffix included)
    double bits_per_key() const {return nkeys_ ? 8. * double(memory()) / nkeys_: 0.;}
};
} // namespace cqf
} // namespace sketch
#endif
