// Prefix-grouped frozen sketch: sorted entries grouped by shared hash prefix.
// Groups of consecutive sorted entries share the top P bits; the
// directory stores group prefixes once (plus an Eytzinger level-order
// copy for probes, per the layout guidance); entries store the low
// bits of the group's range, width-packed (u16/u32/u64 by stored
// width); the fp-truncation dial (FPSTORE < group's low range) keeps
// the HIGH bits of that range and merges truncated collisions at
// build, accumulating counts -- the reduced-space semantics of the
// live fp filter. The counter sidecar exploits the observed
// finding: groups whose counts are all 1 store no counts at all.
#ifndef SKETCH_BCQF_FROZEN_H__
#define SKETCH_BCQF_FROZEN_H__
#include "cqf.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iterator>

namespace sketch {
inline namespace cqf {

template<size_t KEY_BITS, size_t VAL_BITS, size_t FPSTORE = KEY_BITS>
struct frozen_groups {
    static_assert(FPSTORE >= 4 && FPSTORE <= KEY_BITS, "fp dial within the entry width");
    // entry storage packed to the stored width
    using WT = typename std::conditional<FPSTORE <= 16, uint16_t,
              typename std::conditional<FPSTORE <= 32, uint32_t, uint64_t>::type>::type;
    struct group { uint64_t prefix; uint32_t begin, end; uint32_t sidecar; }; // sidecar = ~0u if all counts are 1
    std::vector<group> dir_;        // sorted by prefix (merge-joinable)
    std::vector<group> eytz_;       // level-order copy for probes
    std::vector<WT> low_;           // sorted within group, truncated-collision-free
    std::vector<uint16_t> sidecar_; // counts of groups with any non-1 count (VAL_BITS <= 16 is exact)
    uint64_t nkeys_ = 0, P_ = 0, storebits_ = 0, storeshift_ = 0, lowmask_ = 0;

    const group *find_group(uint64_t q) const {
        size_t lo = 0, hi = dir_.size();
        while(lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if(dir_[mid].prefix < q) lo = mid + 1; else hi = mid;
        }
        return lo < dir_.size() && dir_[lo].prefix == q ? &dir_[lo]: nullptr;
    }
    void build_eytz() {
        const size_t n = dir_.size();
        eytz_.resize(n);
        if(!n) return;
        size_t out = 0;
        std::vector<std::pair<size_t, size_t>> stack;
        stack.push_back({0, n});
        while(!stack.empty()) {
            const auto lr = stack.back(); stack.pop_back();
            if(lr.first >= lr.second) continue;
            const size_t mid = (lr.first + lr.second) / 2;
            eytz_[out++] = dir_[mid];
            stack.push_back({mid + 1, lr.second});
            stack.push_back({lr.first, mid}); // left processed next (level order)
        }
    }
    template<typename It>
    void build(It first, It last, uint64_t target_group = 256) {
        const size_t n = size_t(std::distance(first, last));
        nkeys_ = n;
        if(!n) return;
        for(uint64_t cand = 8; cand <= KEY_BITS; ++cand) { // group-count ~ n/target
            const uint64_t shift = KEY_BITS - cand;
            size_t distinct = 1;
            uint64_t prev = uint64_t(first->first >> shift);
            for(auto it = first + 1; it != last; ++it) {
                const uint64_t p = uint64_t(it->first >> shift);
                if(p != prev) { ++distinct; prev = p; }
            }
            P_ = cand;
            if(distinct * target_group >= n || cand == KEY_BITS) break;
        }
        const uint64_t pshift = KEY_BITS - P_;
        const uint64_t lowrange = pshift;
        storebits_ = FPSTORE < lowrange ? FPSTORE: lowrange;
        storeshift_ = lowrange - storebits_;
        lowmask_ = (uint64_t(1) << storebits_) - 1;
        // partition into prefix groups, filling the (group, stored-low)
        // plane and merging truncated collisions as they appear
        dir_.reserve(n / target_group + 16);
        struct grp { uint64_t prefix; uint32_t begin; };
        std::vector<grp> building;
        low_.reserve(n);
        std::vector<uint32_t> tmpcnt;
        tmpcnt.reserve(n);
        uint64_t cur_prefix = uint64_t(first->first >> pshift);
        uint32_t cur_begin = 0;
        auto flush = [&](size_t end) {
            building.push_back(grp{cur_prefix, cur_begin});
            (void)end;
        };
        for(size_t i = 0; i < n; ++i) {
            const uint64_t hv = uint64_t((first + i)->first);
            const uint64_t p = hv >> pshift;
            const uint64_t sl = (hv >> storeshift_) & lowmask_;
            const uint32_t c = uint32_t((first + i)->second);
            if(p != cur_prefix) {
                flush(i);
                cur_prefix = p;
                cur_begin = uint32_t(low_.size());
                low_.push_back(WT(sl));
                tmpcnt.push_back(c);
            } else if(!low_.empty() && WT(sl) == low_.back()) {
                tmpcnt.back() += c; // truncated collision: merge into the kept entry
            } else {
                low_.push_back(WT(sl));
                tmpcnt.push_back(c);
            }
        }
        flush(n);
        dir_.reserve(building.size());
        for(size_t gi = 0; gi < building.size(); ++gi) {
            const uint32_t begin = building[gi].begin;
            const uint32_t end = gi + 1 < building.size() ? building[gi + 1].begin: uint32_t(low_.size());
            dir_.push_back(group{building[gi].prefix, begin, end, ~0u});
        }
        // counter sidecar: groups with any non-1 count store their counts
        uint32_t side_idx = 0;
        for(auto &g: dir_) {
            bool all1 = true;
            for(uint32_t k = g.begin; k < g.end; ++k) if(tmpcnt[k] != 1) { all1 = false; break; }
            if(all1) continue;
            g.sidecar = side_idx;
            for(uint32_t k = g.begin; k < g.end; ++k) sidecar_.push_back(tmpcnt[k]);
            side_idx = uint32_t(sidecar_.size());
        }
        nkeys_ = low_.size();
        build_eytz();
    }
    uint64_t size() const {return nkeys_;}
    uint64_t prefix_bits() const {return P_;}
    uint64_t stored_bits() const {return storebits_;}
    uint64_t groups() const {return dir_.size();}
    size_t memory() const {
        return dir_.size() * sizeof(group) + eytz_.size() * sizeof(group)
             + low_.size() * sizeof(WT) + sidecar_.size() * sizeof(uint16_t);
    }
    uint64_t count(uint64_t hv) const {
        const group *g = find_group(hv >> (KEY_BITS - P_));
        if(!g) return 0;
        const uint64_t lowv = (hv >> storeshift_) & lowmask_;
        size_t lo = g->begin, hi = g->end;
        while(lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if(low_[mid] < lowv) lo = mid + 1; else hi = mid;
        }
        if(lo >= g->end || low_[lo] != lowv) return 0;
        return g->sidecar == ~0u ? 1: sidecar_[g->sidecar + (lo - g->begin)];
    }
    bool contains(uint64_t hv) const {return count(hv) != 0;}
    template<typename F>
    static void group_merge(const frozen_groups &A, const group &ga,
                            const frozen_groups &B, const group &gb, F &&fn) {
        size_t i = ga.begin, j = gb.begin;
        while(i < ga.end && j < gb.end) {
            const uint64_t la = A.low_[i], lb = B.low_[j];
            if(la == lb) { fn(la); ++i; ++j; }
            else if(la < lb) ++i;
            else ++j;
        }
    }
};
// merge-join over sorted directories: aligned groups merge in-group.
// Sequential, zero probes (the sorted-materialization merge bridge).
template<size_t KB, size_t VB, size_t FA, size_t FB>
uint64_t intersect_count(const frozen_groups<KB,VB,FA> &A, const frozen_groups<KB,VB,FB> &B) {
    uint64_t n = 0;
    size_t i = 0, j = 0;
    while(i < A.dir_.size() && j < B.dir_.size()) {
        if(A.dir_[i].prefix < B.dir_[j].prefix) ++i;
        else if(B.dir_[j].prefix < A.dir_[i].prefix) ++j;
        else {
            frozen_groups<KB,VB,FA>::group_merge(A, A.dir_[i], B, B.dir_[j], [&](uint64_t){++n;});
            ++i; ++j;
        }
    }
    return n;
}
template<size_t KB, size_t VB, size_t FA, size_t FB>
uint64_t union_count(const frozen_groups<KB,VB,FA> &A, const frozen_groups<KB,VB,FB> &B) {
    return A.size() + B.size() - intersect_count(A, B);
}
} // namespace cqf
} // namespace sketch
#endif
