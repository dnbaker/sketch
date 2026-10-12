// hnsw.hpp: minimal HNSW over sketch signatures with an agreement distance.
//
// Vectors are fixed-length register sequences (e.g., pmh2/bmh winning-id
// registers); the distance is register disagreement, d(a, b) = 1 -
// |{i: a[i] == b[i]}| / m, so nearest neighbors in sketch space maximize the
// minhash/pmh2 agreement estimate. The estimate is monotone in Jaccard for
// minhash-style signatures, so sketch-space kNN reranked by agreement is an
// exact-kNN candidate generator for set similarity.
//
// Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs" (TPAMI
// 2020). Single-threaded; callers should shuffle insertion order.

#ifndef SKETCH_HNSW_H__
#define SKETCH_HNSW_H__
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include "sketch/hash.h"

namespace sketch {
namespace lsh {

template<typename Vec, typename Dist>
struct HNSW {
    struct Node {
        Vec v;
        std::vector<std::vector<uint32_t>> nb; // nb[layer] = neighbor ids
        int level;
    };
    std::vector<Node> nodes_;
    size_t maxM_;       // max neighbors per node at layer 0 (upper layers use maxM_ / 2)
    size_t efC_;        // construction beam
    double ml_;         // level multiplier (1 / ln maxM_)
    uint32_t enter_ = 0;
    int maxlayer_ = 0;
    mutable uint64_t dist_evals_ = 0; // distance computations (search and construction)
    mutable std::vector<uint32_t> stamp_; // per-query visited stamps
    mutable uint32_t stamp_ctr_ = 0;
    Dist dist_;

    HNSW(size_t maxM=16, size_t efConstruction=128, uint64_t seed=0, Dist dist=Dist()):
        maxM_(maxM), efC_(efConstruction), dist_(std::move(dist)) {
        ml_ = 1. / std::log(double(maxM));
        rng_.seed(seed ? seed: 0xB5297A4D3F84D5B5ULL);
    }
    size_t size() const {return nodes_.size();}
    uint64_t neighbor_ids() const {
        uint64_t ret = 0;
        for(const auto &n: nodes_) for(const auto &l: n.nb) ret += l.size();
        return ret;
    }
    template<typename RNG> static int sample_level(RNG &rng, double ml) {
        const double u = std::uniform_real_distribution<double>(1e-9, 1)(rng);
        return int(std::floor(-std::log(u) * ml));
    }
    double d(const Vec &a, const Vec &b) const {
        ++dist_evals_;
        return dist_(a, b);
    }
    // Diversity-aware neighbor selection (Malkov & Yashunin sec 4.2.1): from
    // candidates ascending by distance to the query, keep a candidate only if
    // it is closer to the query than to every already-selected neighbor; this
    // retains far-reaching links that closest-M selection prunes away, which
    // is what keeps the graph navigable (and every node reachable).
    std::vector<std::pair<double, uint32_t>> select_diverse(
            const Vec &q, std::vector<std::pair<double, uint32_t>> cand, size_t M) const {
        std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
            const uint64_t ha = hash::WangHash::hash(a.second), hb = hash::WangHash::hash(b.second);
            return a.first != b.first ? a.first < b.first: ha < hb;
        });
        std::vector<std::pair<double, uint32_t>> sel;
        for(const auto &c: cand) {
            if(sel.size() >= M) break;
            bool good = true;
            for(const auto &s: sel) {
                if(d(nodes_[c.second].v, nodes_[s.second].v) < c.first) {good = false; break;}
            }
            if(good) sel.push_back(c);
        }
        for(const auto &c: cand) { // fill up with closest leftovers if the heuristic kept too few
            if(sel.size() >= std::min(M, cand.size())) break;
            bool have = false;
            for(const auto &s: sel) have |= s.second == c.second;
            if(!have) sel.push_back(c);
        }
        return sel;
    }
    void add(const Vec &v) {
        const uint32_t id = uint32_t(nodes_.size());
        Node nd;
        nd.v = v;
        nd.level = sample_level(rng_, ml_);
        if(nodes_.empty()) {
            nd.nb.resize(size_t(nd.level) + 1);
            nodes_.push_back(std::move(nd));
            enter_ = 0; maxlayer_ = nodes_[0].level;
            return;
        }
        uint32_t ep = enter_;
        for(int lc = maxlayer_; lc > nd.level; --lc) {
            bool moved = true;
            while(moved) {
                moved = false;
                double bd = d(nodes_[ep].v, v);
                for(const uint32_t w: nodes_[ep].nb[lc]) {
                    const double dw = d(nodes_[w].v, v);
                    if(dw < bd) {bd = dw; ep = w; moved = true;}
                }
            }
        }
        nd.nb.resize(size_t(nd.level) + 1);
        for(int lc = std::min(maxlayer_, nd.level); lc >= 0; --lc) {
            auto cand = search_layer(v, ep, lc, efC_);
            const size_t M = lc ? maxM_ / 2: maxM_;
            auto sel = select_diverse(v, std::move(cand), M);
            for(const auto &pr: sel) {
                const uint32_t w = pr.second;
                if(w == id) continue;
                nd.nb[lc].push_back(w);
                auto &wnb = nodes_[w].nb[lc];
                wnb.push_back(id);
                if(wnb.size() > M) {
                    std::vector<std::pair<double, uint32_t>> ws(wnb.size());
                    for(size_t i = 0; i < wnb.size(); ++i) {
                        // id is the in-flight node: its vector is v, distance to w already known
                        ws[i] = {wnb[i] == id ? pr.first: d(nodes_[wnb[i]].v, nodes_[w].v), wnb[i]};
                    }
                    std::sort(ws.begin(), ws.end(), [](const auto &a, const auto &b) {
                        const uint64_t ha = hash::WangHash::hash(a.second), hb = hash::WangHash::hash(b.second);
                        return a.first != b.first ? a.first < b.first: ha < hb;
                    });
                    ws.resize(M);
                    wnb.clear();
                    for(const auto &pr2: ws) wnb.push_back(pr2.second);
                }
            }
            ep = sel.empty() ? ep: sel[0].second;
        }
        if(nd.level > maxlayer_) {enter_ = id; maxlayer_ = nd.level;}
        nodes_.push_back(std::move(nd));
    }
    // ef-best-first search at one layer from entry point ep; returns (dist, id) ascending
    std::vector<std::pair<double, uint32_t>> search_layer(const Vec &q, uint32_t ep, int lc, size_t ef) const {
        if(stamp_.size() < nodes_.size()) stamp_.resize(nodes_.size(), 0);
        const uint32_t my_stamp = ++stamp_ctr_;
        std::vector<std::pair<double, uint32_t>> result, cand;
        result.reserve(ef + 1);
        cand.reserve(ef + 1);
        const double dep = d(nodes_[ep].v, q);
        result.push_back({dep, ep});
        cand.push_back({dep, ep});
        stamp_[ep] = my_stamp;
        // min-heap on distance for the expansion queue; max-heap for the ef-bounded result
        auto cmp_min = [](const std::pair<double, uint32_t> &a, const std::pair<double, uint32_t> &b) {
            return a.first != b.first ? a.first > b.first: a.second > b.second;
        };
        auto cmp_max = [](const std::pair<double, uint32_t> &a, const std::pair<double, uint32_t> &b) {
            return a.first != b.first ? a.first < b.first: a.second < b.second;
        };
        std::make_heap(result.begin(), result.end(), cmp_max);
        std::make_heap(cand.begin(), cand.end(), cmp_min);
        while(!cand.empty()) {
            std::pop_heap(cand.begin(), cand.end(), cmp_min);
            const auto cur = cand.back();
            cand.pop_back();
            const double fworst = result.front().first;
            if(cur.first > fworst) break;
            for(const uint32_t w: nodes_[cur.second].nb[lc]) {
                if(stamp_[w] == my_stamp) continue;
                stamp_[w] = my_stamp;
                const double dw = d(nodes_[w].v, q);
                if(dw < fworst || result.size() < ef) {
                    result.push_back({dw, w});
                    std::push_heap(result.begin(), result.end(), cmp_max);
                    cand.push_back({dw, w});
                    std::push_heap(cand.begin(), cand.end(), cmp_min);
                    if(result.size() > ef) {
                        std::pop_heap(result.begin(), result.end(), cmp_max);
                        result.pop_back();
                    }
                }
            }
        }
        std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first < b.first: a.second < b.second;
        });
        return result;
    }
    // k-NN over the whole index; n_dist (if given) receives search distance evals
    std::vector<std::pair<double, uint32_t>> search(const Vec &q, size_t k, size_t ef, uint64_t *n_dist=nullptr) const {
        const uint64_t before = dist_evals_;
        uint32_t ep = enter_;
        for(int lc = maxlayer_; lc > 0; --lc) {
            bool moved = true;
            while(moved) {
                moved = false;
                double bd = d(nodes_[ep].v, q);
                for(const uint32_t w: nodes_[ep].nb[lc]) {
                    const double dw = d(nodes_[w].v, q);
                    if(dw < bd) {bd = dw; ep = w; moved = true;}
                }
            }
        }
        auto res = search_layer(q, ep, 0, std::max(ef, k));
        if(res.size() > k) res.resize(k);
        if(n_dist) *n_dist = dist_evals_ - before;
        return res;
    }
private:
    std::mt19937_64 rng_;
};

} // namespace lsh
} // namespace sketch

#endif
