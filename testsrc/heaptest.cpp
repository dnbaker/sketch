#include "heap.h"
#include "ccm.h"
#include "mh.h"
#include "hash.h"
#include <cassert>
#define show(...)
//template<typename T>
//void show(const T &x, std::string t="no name") {int i = 0; for(auto _i: x) {if(++i >= 10) break; std::fprintf(stderr, "%s-%zu\n", t.data(), size_t(_i));}}
#define ZOMG3 1

using namespace sketch::heap;
using namespace sketch::cm;
using namespace sketch;
using Hash = sketch::hash::WangHash;
auto cmp = std::less<>();
int main() {
    std::mt19937_64 mt(1337);
    using cmp = std::less<>;
    ObjHeap<uint64_t, cmp> zomg(100);
    std::set<uint64_t, cmp> zomgset;
    ObjScoreHeap<uint64_t, cmp, std::hash<uint64_t>> zomg2(100);
    csbase_t<> cs(10, 5);
#if ZOMG3
    SketchHeap<uint64_t, decltype(cs)> zomg3(100, csbase_t<>(cs));
#endif
    std::vector<uint64_t> all;
    for(size_t i=0; i < 1000000; ++i) {
        auto v = mt();
        all.push_back(v);
        zomg.addh(v);
        //zomg4.addh(v);
        zomgset.insert(v);
        if(zomgset.size() > zomg.size())
            zomgset.erase(zomgset.find(*zomgset.rbegin()));
    }
    for(const auto v: all) zomg2.addh(v, v);
    auto zomgvec = zomg.template to_container<>();
    auto ozomgvec = std::vector<uint64_t, Allocator<uint64_t>>(zomgset.begin(), zomgset.end());
#if ZOMG4
    auto zomgvec4 = zomg4.template to_container<>();
#endif
#if ZOMG3
    auto zomgvec3 = zomg3.template to_container<>();
#endif
    auto zomgvec2 = zomg2.template to_container<>();
    std::sort(zomgvec.begin(), zomgvec.end());
    std::sort(zomgvec2.begin(), zomgvec2.end());
    //std::sort(zomgvec3.begin(), zomgvec3.end());
    //std::sort(zomgvec4.begin(), zomgvec4.end());
    std::sort(all.begin(), all.end());
    show(ozomgvec, "zomgset");
    show(zomgvec, "zomgvec");
    show(zomgvec2, "zomgvec2");
    //show(zomgvec3, "zomgvec3");
    show(all, "all");
    assert(ozomgvec == zomgvec);
    assert(ozomgvec == zomgvec2);
    //assert(zomgvec3 == zomgvec);
    //auto isz = sketch::common::intersection_size(zomgvec3, zomgvec3, std::less<uint64_t>());
#if VERBOSE_AF
    std::fprintf(stderr, "Done with adding zomg2\n");
#endif
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg.max_size(), zomg.size());
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg2.max_size(), zomg2.size());
    //size_t usz = zomgvec3.size() + zomgvec3.size() - isz;
    //std::fprintf(stderr, "isz: %zu. usz: %zu\n", size_t(isz), usz);
}
