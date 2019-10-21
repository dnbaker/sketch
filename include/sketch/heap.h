#ifndef SKETCH_HEAP_H__
#define SKETCH_HEAP_H__
#include "common.h"
#include <mutex>
#include <cstdarg>
#include "flat_hash_map/flat_hash_map.hpp"

namespace sketch {
#ifndef LOG_DEBUG
#    define UNDEF_LDB
#    if !NDEBUG
#        define LOG_DEBUG(...) log_debug(__PRETTY_FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
static int log_debug(const char *func, const char *filename, int line, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int ret(std::fprintf(stderr, "[D:%s:%s:%d] ", func, filename, line));
    ret += std::vfprintf(stderr, fmt, args);
    va_end(args);
    return ret;
}
#    else
#        define LOG_DEBUG(...)
#    endif
#endif
namespace heap {

using std::hash; // This way, someone can provide a hash within an object's namespace for argument dependant lookup.
// https://arxiv.org/abs/1711.00975
template<typename Obj, typename Cmp=std::greater<Obj>, typename HashFunc=hash<Obj> >
class ObjHeap {
#define GET_LOCK_AND_CHECK
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
        if(!n) throw 1;
    }
    template<typename T1, typename T2> auto cmp(const T1 &x, const T2 &y) {return cmp_(x, y);}
    bool check(const Obj &x) const {
        return cmp_(x, top());
    }
    auto &top() {return core_.front();}
    const auto &top() const {return core_.front();}
    void addh(const Obj &x) {
        Obj tmp(x);
        addh(static_cast<Obj &&>(tmp));
    }
    void addh(Obj &&o) {
        using std::to_string;
        auto hv = h_(o);
        if(hashes_.find(hv) != hashes_.end()) {
            std::fprintf(stderr, "Object present or collision, doing nothing\n");
            return;
        }
        if(core_.size() < m_) {
#ifndef NOT_THREADSAFE
            std::lock_guard<std::mutex> lock(mut_);
#endif
            if(core_.size() >= m_) {
                std::fprintf(stderr, "Set surpassed size unexpected\n. Try again\n");
                addh(std::move(o));
                return;
            }
            if(hashes_.find(hv) != hashes_.end()) std::fprintf(stderr, "Warning: hash value occurring twice. There will be duplicates, so delete carefully.\n");
            hashes_.emplace(hv);
            core_.emplace_back(std::move(o));
            std::push_heap(core_.begin(), core_.end(), cmp_);
        } else {
            if(check(o)) {
#ifndef NOT_THREADSAFE
                std::lock_guard<std::mutex> lock(mut_);
#endif
                if(!check(o)) return;
                std::pop_heap(core_.begin(), core_.end(), cmp_);
                hashes_.erase(hashes_.find(h_(core_.back())));
                hashes_.emplace(hv);
                core_.back() = std::move(o);
                std::push_heap(core_.begin(), core_.end(), cmp_);
            }
        }
    }
#undef GET_LOCK_AND_CHECK
    size_t max_size() const {return m_;}
    size_t size() const {return core_.size();}
    template<typename Func>
    void for_each(const Func &func) const {
        std::for_each(core_.begin(), core_.end(), func);
    }
    template<typename VecType=std::vector<Obj, Allocator<Obj>>>
    VecType to_container() const {
        VecType ret; ret.reserve(size());
        for(const auto &v: core_)
            ret.push_back(v);
        return ret;
    }
};

template<typename HashType, typename Cmp>
struct HashCmp {
    const HashType hash_;
    const Cmp cmp_;
    HashCmp(HashType &&hash=HashType()): hash_(std::move(hash)), cmp_() {}
    template<typename T>
    bool operator()(const T &a, const T &b) const {
        return cmp_(hash_(a), hash_(b));
    }
};
template<typename Obj, typename Cmp=std::greater<Obj>, typename HashFunc=hash<Obj> >
class ObjHashHeap: public ObjHeap<Obj, HashCmp<HashFunc, Cmp>, HashFunc> {
public:
    using super = ObjHeap<Obj, HashCmp<HashFunc, Cmp>, HashFunc>;
    template<typename... Args> ObjHashHeap(Args &&...args): super(std::forward<Args>(args)...) {}
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

template<typename Obj, typename MainCmp=std::less<uint64_t>, typename HashFunc=hash<Obj>>
class ObjScoreHeap {
public:
    using ScoreType = uint64_t;
    using Pair = std::pair<Obj, ScoreType>;
    struct Cmp {
        INLINE bool operator()(const Pair &x, const Pair &y) const {
            return MainCmp()(x.second, y.second);
        }
        INLINE bool operator()(uint64_t x, uint64_t y) const {
            return MainCmp()(x, y);
        }
    };
    struct Hash {
        HashFunc func_;
        auto operator()(const Obj &x) const {return func_(x);}
        auto operator()(const Pair &x) const {return func_(x.first);}
    };
    using Heap = ObjHeap<Pair, Cmp, Hash>;
    Heap heap_;

    template<typename...Args>
    ObjScoreHeap(Args &&...args): heap_(std::forward<Args>(args)...) {}

    template<typename T1, typename T2> auto cmp(const T1 &x, const T2 &y) {return cmp_(x, y);}

    void addh(const Obj &x, ScoreType score) {
        heap_.addh(Pair(x, score));
    }
    template<typename VecType=std::vector<Obj, Allocator<Obj>>>
    VecType to_container() const {
        auto con = heap_.to_container();
        VecType ret;
        ret.reserve(size());
        for(const auto &v: con)
            ret.emplace_back(std::move(v.first));
        return ret;
    }
    size_t size() const {return heap_.size();}
    size_t max_size() const {return heap_.max_size();}
};
template<typename Obj, typename MainCmp=DefaultScoreCmp<uint64_t>, typename HashFunc=hash<Obj>>
class ObjPtrScoreHeap: public ObjScoreHeap<Obj, MainCmp, HashFunc> {};

template<typename CSketchType>
struct SketchCmp {
    CSketchType &csketch_;
    SketchCmp(CSketchType &sketch): csketch_(sketch) {}
    auto &sketch() {return csketch_;}
    const auto &sketch() const {return csketch_;}
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
#define GET_LOCK_AND_CHECK \
            std::lock_guard<std::mutex> lock(mut_); \
            if(core_.size() >= m_ && !cmp_(o, top())) return;
#else
#define GET_LOCK_AND_CHECK
#endif
    const uint64_t m_;
public:
    template<typename... Args>
    SketchHeap(size_t n, CSketchType &&csketch,  HashFunc &&hf=HashFunc(), Args &&...args):
        h_(std::move(hf)), cmp_(csketch), m_(n)
    {
        core_.reserve(n);
    }

    template<typename T1, typename T2> auto cmp(const T1 &x, const T2 &y) {return cmp_(x, y);}

    auto &top() {return core_.front();}
    const auto &top() const {return core_.front();}
    void addh(Obj &&o) {
        auto hv = h_(o);
        if(core_.size() < m_) {
            if(hashes_.find(hv) != hashes_.end()) {
                /*std::fprintf(stderr, "Found hash: %zu\n", size_t(*hashes_.find(hv))); */
                return;
            }
            GET_LOCK_AND_CHECK
            hashes_.emplace(hv);
            cmp_.sketch().addh(hv);
            core_.emplace_back(std::move(o));
            std::push_heap(core_.begin(), core_.end(), cmp_);
        } else if(cmp_(o, core_.front()) && o != top()) {
            std::pop_heap(core_.begin(), core_.end(), cmp_);
            auto it = hashes_.find(h_(core_.front()));
            if(it != hashes_.end())
                hashes_.erase(it);
            hashes_.emplace(hv);
            //Obj s(std::move(core_.back()));
            core_.back() = std::move(o);
            std::push_heap(core_.begin(), core_.end(), cmp_);
        }

    }
    void addh(const Obj &o) {
        Obj t(o);
        addh(static_cast<Obj &&>(t));
    }
#undef GET_LOCK_AND_CHECK
    size_t size() const {return core_.size();}
    size_t max_size() const {return m_;}
    template<typename VecType=std::vector<Obj, Allocator<Obj>>>
    VecType to_container() const {
        VecType ret; ret.reserve(size());
        for(auto v: core_)
            ret.push_back(v);
        return ret;
    }
};

} // namespace heap

} // namespace sketch
#ifdef UNDEF_LDB
#  undef LOG_DEBUG
#  undef UNDEF_LDB
#endif


#endif /* #ifndef SKETCH_HEAP_H__ */
