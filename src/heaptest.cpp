#include "heap.h"
#include "ccm.h"
#include "mh.h"
#include <cassert>



using namespace sketch::heap;
using namespace sketch::cm;
int main() {
    std::mt19937_64 mt(1337);
    ObjHeap<uint64_t, std::less<uint64_t>> zomg(100);
    ObjHashHeap<uint64_t, std::less<uint64_t>> zomg4(100);
    std::set<uint64_t> zomgset;
    ObjScoreHeap<uint64_t> zomg2(100);
    csbase_t<uint64_t> cs(10, 5);
    SketchHeap<uint64_t, csbase_t<uint64_t>> zomg3(100, cs);
#if VERBOSE_AF
    std::fprintf(stderr, "Done with initializing\n");
#endif
    for(size_t i=0; i < 1000000; ++i) {
        auto v = mt();
        zomg.addh(v);
        zomg4.addh(v);
        zomgset.insert(v);
        if(zomgset.size() > zomg.size())
            zomgset.erase(zomgset.find(*zomgset.rbegin()));
        zomg3.addh(v);
    }
#if VERBOSE_AF
    std::fprintf(stderr, "Done with adding zomg\n");
#endif
    for(size_t i=0; i < 10000; ++i) { 
        auto v = mt();
        zomg2.addh(v, v * v  % ((1u << 31) - 1));
    }
    auto zomgvec = zomg.template to_container<>();
    auto zomgvec3 = zomg4.template to_container<>();
    auto ozomgvec = std::vector<uint64_t, Allocator<uint64_t>>(zomgset.begin(), zomgset.end());
    auto zomgvec4 = zomg3.template to_container<>();
    std::sort(zomgvec.begin(), zomgvec.end());
    std::sort(zomgvec3.begin(), zomgvec3.end());
    std::sort(zomgvec4.begin(), zomgvec4.end());
    assert(ozomgvec == zomgvec);
    assert(zomgvec3 == zomgvec);
    auto isz = sketch::common::intersection_size(zomgvec3, zomgvec3, std::less<uint64_t>());
#if VERBOSE_AF
    std::fprintf(stderr, "Done with adding zomg2\n");
#endif
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg.max_size(), zomg.size());
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg2.max_size(), zomg2.size());
    size_t usz = zomgvec3.size() + zomgvec4.size() - isz;
    std::fprintf(stderr, "isz: %zu. usz: %zu\n", isz, usz);
}
