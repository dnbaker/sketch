#ifndef SKETCH_HEAP_H__
#define SKETCH_HEAP_H__
#include "common.h"
#include <mutex>
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {
namespace heap {
using namespace common;

// https://arxiv.org/abs/1711.00975
template<typename Obj, typename Cmp=std::greater<Obj>, typename HashFunc=std::hash<Obj>
        >
class ObjHeap {
#ifndef NOT_THREADSAFE
#define GET_LOCK \
            std::lock_guard<std::mutex> lock(mut_); \
            if(core_.size() >= m_ && !cmp_(o, core_.front())) return;
#else
#define GET_LOCK
#endif
    std::vector<Obj> core_;
    HashFunc h_;
    using HType = uint64_t;
    ska::flat_hash_set<HType> hashes_;
#ifndef NOT_THREADSAFE
    std::mutex mut_;
#endif
    const Cmp cmp_;
    const uint64_t m_;
public:
    template<typename... Args>
    ObjHeap(size_t n, HashFunc &&hf=HashFunc(), Args &&...args): h_(std::move(hf)), cmp_(std::forward<Args>(args)...), m_(n) {
        core_.reserve(n);
    }
#define ADDH_CORE(op)\
        using std::to_string;\
        auto hv = h_(o);\
        if((core_.size() < m_ || cmp_(o, core_[0]))) { \
            if(hashes_.find(hv) != hashes_.end()) {\
                std::fprintf(stderr, "hv present. Ignoring\n");\
                return;\
            } \
            GET_LOCK\
            hashes_.emplace(hv);\
            core_.emplace_back(op(o));\
            std::push_heap(core_.begin(), core_.end(), cmp_);\
            if(core_.size() > m_) {\
                std::pop_heap(core_.begin(), core_.end(), cmp_);\
                hashes_.erase(hashes_.find(h_(core_.back()))); \
                core_.pop_back();\
                /* std::fprintf(stderr, "new min: %s\n", to_string(core_.front()).data()); */\
            }\
        }
    void addh(Obj &&o) {
        ADDH_CORE(std::move)
    }
    void addh(const Obj &o) {
        ADDH_CORE()
    }
#undef ADDH_CORE
    size_t max_size() const {return m_;}
    size_t size() const {return core_.size();}
};


template<typename Obj, typename HashFunc=std::hash<Obj>, typename ScoreType=std::uint64_t>
class ObjScoreHeap {

    using TupType = std::pair<Obj, ScoreType>;
    struct MainCmp {
        bool operator()(const TupType &a, const TupType &b) const {
            return a.second > b.second;
        }
        bool operator()(ScoreType score, const TupType &b) const {
            return score > b.second;
        }
    };

    HashFunc h_;
    using HType = uint64_t;
    std::vector<TupType> core_;
    const MainCmp cmp_;
    ska::flat_hash_set<HType> hashes_;
#ifndef NOT_THREADSAFE
    std::mutex mut_;
#endif
    const uint64_t m_;
public:
    template<typename... Args>
    ObjScoreHeap(size_t n, HashFunc &&hf=HashFunc(), Args &&...args):
        h_(std::move(hf)), m_(n), cmp_()
    {
        core_.reserve(n);
    }

    void addh(Obj &&o, ScoreType score) {
#define ADDH_CORE(op)\
        auto hv = h_(o);\
        if(core_.size() < m_ || cmp_(score, core_[0])) {\
            if(hashes_.find(hv) != hashes_.end()) {\
                /*std::fprintf(stderr, "Found hash: %zu\n", size_t(*hashes_.find(hv))); */\
                return;\
            }\
            std::lock_guard<std::mutex> lock(mut_);\
            if(core_.size() >= m_ && !cmp_(score, core_[0])) return;\
            hashes_.emplace(hv);\
            core_.emplace_back(std::make_pair(op(o), score));\
            std::push_heap(core_.begin(), core_.end(), cmp_);\
            if(core_.size() > m_) {\
                std::pop_heap(core_.begin(), core_.end(), cmp_);\
                hashes_.erase(hashes_.find(h_(core_.back().first))); \
                core_.pop_back();\
            }\
        }
        ADDH_CORE(std::move)
    }
    void addh(const Obj &o, ScoreType score) {
        ADDH_CORE()
    }
    size_t size() const {return core_.size();}
    size_t max_size() const {return m_;}
};

#undef GET_LOCK
} // namespace heap

} // namespace sketch


#endif /* #ifndef SKETCH_HEAP_H__ */
