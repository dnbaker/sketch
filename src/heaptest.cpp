#include "heap.h"
#include "ccm.h"
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
    std::sort(zomgvec.begin(), zomgvec.end());
    std::sort(zomgvec3.begin(), zomgvec3.end());
    assert(ozomgvec == zomgvec);
    assert(zomgvec3 == zomgvec);
#if VERBOSE_AF
    std::fprintf(stderr, "Done with adding zomg2\n");
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg.max_size(), zomg.size());
    std::fprintf(stderr, "max: %zu. csize: %zu\n", zomg2.max_size(), zomg2.size());
#endif
}
