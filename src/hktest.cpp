#include "hk.h"
#include "common.h"
#include <set>
#include <unordered_map>

using namespace sketch::hk;
using namespace sketch;
int main() {
    HeavyKeeper<32,32> hk(1000, 5, 1.03);
    const size_t ninsert = 20;
    std::vector<uint64_t> items{1u,3u,5u,101u};
    for(size_t i = 0; i < 500; ++i)
        items.push_back(items.size() * items.size());
    {
        std::set<uint64_t> is(items.begin(), items.end());
        items.assign(is.begin(), is.end());
    }
    size_t missing = 0;
    for(const size_t item: items) {
        for(size_t i = 0; i < ninsert; ++i) {
            /*auto val = */hk.addh(item);
            //auto postq = hk.query(item);
            //std::fprintf(stderr, "for item %zu prev: %zu. post: %zu\n", item, size_t(val), size_t(postq));
        }
        //auto current_val = size_t(hk.query(item));
        //std::fprintf(stderr, "after bulk insert %zu prev: %zu. inserted: %zu\n", item, current_val, ninsert);
        std::fflush(stderr);
        missing += hk.query(item) == 0;
    }
    std::fprintf(stderr, "missing: %zu of total %zu\n", missing, items.size());
    assert(missing < items.size() / 10);
    assert(missing <= 20);
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash()(x);});
    missing = 0;
    for(const size_t item: items) {
        auto prev = hk.query(item);
        for(size_t i = 0; i < ninsert * 10; ++i) {
            /*auto val = */hk.addh(item);
            //std::fprintf(stderr, "postq: %zu\n", postq);
        }
        //std::fprintf(stderr, "for item %zu prev: %zu. post: %zu\n", item, size_t(prev), hk.query(item));
        assert(prev == 0);
        if(hk.query(item) == 0) ++missing;
    }
    assert(missing <= 42); // This could actually change and still be fine, but I want to know about these changes.
    std::fprintf(stderr, "missing 10x items: %zu of total %zu\n", missing, items.size());
    assert(missing < items.size() / 10);
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash().inverse(x);});
    for(const size_t item: items) {
        auto postq = hk.query(item);
        assert(postq <= ninsert);
    }
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash()(x);});
    std::unordered_map<std::string, unsigned> sset;
    std::string line;
    for(const size_t item: items) {
        char buf[256];
        auto postq = hk.query(item);
        std::sprintf(buf, "After being inverting back, %zu replaces %zu\n", size_t(postq), ninsert);
        line = buf;
        ++sset[line];
    }
    std::vector<std::pair<std::string, unsigned>> svec(sset.begin(), sset.end());
    std::sort(svec.begin(), svec.end(), [](const auto &x, const auto &y) {return x.second < y.second;});
    for(const auto &pair: svec) {
        std::fprintf(stderr, "%u:%s", pair.second, pair.first.data());
    }
}
