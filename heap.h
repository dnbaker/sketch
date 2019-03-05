#ifndef SKETCH_HEAP_H__
#define SKETCH_HEAP_H__
#include "common.h"
#include <mutex>
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {
namespace heap {
using namespace common;

using std::hash;
// https://arxiv.org/abs/1711.00975
template<typename Obj, typename Cmp=std::greater<Obj>, typename HashFunc=hash<Obj> >
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
#undef GET_LOCK
    size_t max_size() const {return m_;}
    size_t size() const {return core_.size();}
};
template<typename ScoreType>
struct DefaultScoreCmp {
    template<typename T>
    bool operator()(const std::pair<T, ScoreType> &a, const std::pair<T, ScoreType> &b) const {
        return a.second > b.second;
    }
    template<typename T>
    bool operator()(const std::pair<std::unique_ptr<T>, ScoreType> &a, const std::pair<std::unique_ptr<T>, ScoreType> &b) const {
        return a.second > b.second;
    }
    template<typename T>
    bool operator()(ScoreType score, const std::pair<T, ScoreType> &b) const {
        return score > b.second;
    }
    template<typename T>
    bool operator()(ScoreType score, const std::pair<std::unique_ptr<T>, ScoreType> &b) const {
        return score > b.second;
    }
};

template<typename Obj, typename HashFunc=hash<Obj>, typename ScoreType=std::uint64_t, typename MainCmp=DefaultScoreCmp<ScoreType>>
class ObjScoreHeap {

    using TupType = std::pair<Obj, ScoreType>;
    HashFunc h_;
    using HType = uint64_t;
    std::vector<TupType> core_;
    const MainCmp cmp_;
    ska::flat_hash_set<HType> hashes_;
#ifndef NOT_THREADSAFE
    std::mutex mut_;
#define GET_LOCK \
            std::lock_guard<std::mutex> lock(mut_); \
            if(core_.size() >= m_ && !cmp_(o, core_.front())) return;
#else
#define GET_LOCK
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
            GET_LOCK\
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
#undef GET_LOCK
    size_t size() const {return core_.size();}
    size_t max_size() const {return m_;}
};
template<typename Obj, typename HashFunc=hash<Obj>, typename ScoreType=std::uint64_t, typename MainCmp=DefaultScoreCmp<ScoreType>>
class ObjPtrScoreHeap: public ObjScoreHeap<Obj, HashFunc, ScoreType, MainCmp> {
};

template<typename CSketchType>
struct SketchCmp {
    CSketchType &csketch_;
    SketchCmp(CSketchType &sketch): csketch_(sketch) {}
    template<typename Obj>
    bool operator()(const Obj &a, const Obj &b) const {
        return csketch_.est_count(a) > csketch_.est_count(b);
    }
    template<typename Obj>
    bool add_cmp(const Obj &a, const Obj &b) {
        return csketch_.addh_val(a) > csketch_.est_count(b);
    }
    template<typename Obj>
    bool operator()(const std::unique_ptr<Obj> &a, const std::unique_ptr<Obj> &b) const {
        return csketch_.est_count(*a) > csketch_.est_count(*b);
    }
    template<typename Obj>
    bool add_cmp(const std::unique_ptr<Obj> &a, const std::unique_ptr<Obj> &b) {
        return csketch_.addh_val(*a) > csketch_.est_count(*b);
    }
};

template<typename Obj, typename CSketchType, typename HashFunc=hash<Obj>>
class SketchHeap {
    HashFunc h_;
    using HType = uint64_t;
    std::vector<Obj> core_;
    ska::flat_hash_set<HType> hashes_;
    SketchCmp<CSketchType> cmp_;
#ifndef NOT_THREADSAFE
    std::mutex mut_;
#define GET_LOCK \
            std::lock_guard<std::mutex> lock(mut_); \
            if(core_.size() >= m_ && !cmp_(o, core_.front())) return;
#else
#define GET_LOCK
#endif
    const uint64_t m_;
public:
    template<typename... Args>
    SketchHeap(size_t n, CSketchType &csketch,  HashFunc &&hf=HashFunc(), Args &&...args):
        h_(std::move(hf)), cmp_(csketch), m_(n)
    {
        core_.reserve(n);
    }

    void addh(Obj &&o) {
#define ADDH_CORE(op)\
        auto hv = h_(o);\
        if(core_.size() < m_ || cmp_.add_cmp(o, core_[0])) {\
            if(hashes_.find(hv) != hashes_.end()) {\
                /*std::fprintf(stderr, "Found hash: %zu\n", size_t(*hashes_.find(hv))); */\
                return;\
            }\
            GET_LOCK\
            hashes_.emplace(hv);\
            core_.emplace_back(op(o));\
            std::push_heap(core_.begin(), core_.end(), cmp_);\
            if(core_.size() > m_) {\
                std::pop_heap(core_.begin(), core_.end(), cmp_);\
                hashes_.erase(hashes_.find(h_(core_.back().first))); \
                core_.pop_back();\
            }\
        }
        ADDH_CORE(std::move)
    }
    void addh(const Obj &o) {
        ADDH_CORE()
    }
#undef GET_LOCK
    size_t size() const {return core_.size();}
    size_t max_size() const {return m_;}
};

} // namespace heap

} // namespace sketch


#endif /* #ifndef SKETCH_HEAP_H__ */
