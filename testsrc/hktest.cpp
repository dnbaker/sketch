#include "hk.h"
#include "heap.h"
#include <set>
#include <unordered_map>

using namespace sketch::hk;
using namespace sketch;
static constexpr int defaulttbsz = 1000;
int tbsz = defaulttbsz;
unsigned nh = 5;
unsigned nelem = 500;
void run_hk_point();
void run_hkh();
void run_random();
int main(int argc, char *argv[]) {
    if(argc > 1) tbsz = std::atoi(argv[1]);
    if(argc > 2) nh =   std::atoi(argv[2]);
    if(argc > 3) nelem = std::max(nelem, unsigned(std::atoi(argv[3])));
    run_hk_point();
    run_random();
    run_hkh();
}

namespace std {
    template<>
    struct hash<std::pair<uint32_t, uint32_t>> {
        uint64_t operator()(const std::pair<uint32_t, uint32_t> &p) const {
            return WangHash()(*reinterpret_cast<const uint64_t *>(&p));
        }
    };
}

void run_random() {
    size_t d = nelem;
    auto p = std::make_unique<uint64_t[]>(d);
    std::vector<uint32_t> n(d);
    wy::WyRand<uint64_t, 2> wy(d);
    std::poisson_distribution<> dist(4);
    for(auto it = p.get(), e = p.get() + d; it != e;)
        *it++ = wy();
    for(auto &i: n) {
        i = std::pow(std::max(uint32_t(dist(wy)), 1u), 3);
    }
    std::vector<uint64_t> vals;
    for(size_t i = 0; i < d; ++i) {
        vals.insert(vals.end(), n[i], p.get()[i]);
    }
    std::shuffle(vals.begin(), vals.end(), wy);
    HeavyKeeper<32,32> hk(tbsz, nh, 1.03);
    for(const auto v: vals) hk.addh(v);
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> m;
    std::pair<uint32_t, uint32_t> stuff;
    for(auto it = p.get(), e = p.get() + d; it != e; ++it) {
        auto i = it - p.get();
        stuff.first = n[i];
        stuff.second = hk.queryh(*it);
        ++m[stuff];
    }
    for(const auto &pair: m) {
        std::fprintf(stderr, "expect: %u. got: %u. count: %u\n", pair.first.first, pair.first.second, pair.second);
    }
}

void run_hkh() {
    //using hkt = HeavyKeeper<32,32>;
}

void run_hk_point() {
    std::fprintf(stderr, "tbsz: %d. nh: %d. nelem: %d\n", tbsz, nh, nelem);
    static_assert(is_hk<HeavyKeeper<32,32>>::value, "Must be hk");
    static_assert(!is_hk<int>::value, "Must be hk");
    HeavyKeeper<32,32> hk(tbsz, nh, 1.03);
    const size_t ninsert = 20;
    std::vector<uint64_t> items{1u,3u,5u,101u};
    for(size_t i = 0; i < nelem; ++i)
        items.push_back(items.size() * items.size());
    std::set<uint64_t> is(items.begin(), items.end());
    {
        items.assign(is.begin(), is.end());
        std::vector<uint64_t> tmp;
        tmp.reserve(ninsert * items.size());
        for(const auto v: items) tmp.insert(tmp.end(), ninsert, v);
        std::swap(tmp, items);
        wy::WyRand<uint32_t, 2> wy(13);
        std::shuffle(items.begin(), items.end(), wy);
    }
    size_t missing = 0;
    for(const size_t item: items) {
            /*auto val = */hk.addh(item);
            //auto postq = hk.query(item);
            //std::fprintf(stderr, "for item %zu prev: %zu. post: %zu\n", item, size_t(val), size_t(postq));
        //auto current_val = size_t(hk.query(item));
        //std::fprintf(stderr, "after bulk insert %zu prev: %zu. inserted: %zu\n", item, current_val, ninsert);
    }
    for(const auto i: is) missing += hk.queryh(i) == 0;
    std::fprintf(stderr, "missing: %zu of total %zu\n", missing, items.size());
    assert(missing < items.size() / 10);
    //assert(missing <= 20);
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash()(x);});
    missing = 0;
    items.reserve(8 * items.size());
    for(unsigned i = 0; i < 3; ++i)
        items.insert(items.end(), items.begin(), items.end());
    for(const size_t item: items) {
        auto prev = hk.queryh(item);
        //std::fprintf(stderr, "for item %zu prev: %zu. post: %zu\n", item, size_t(prev), hk.queryh(item));
        assert(prev == 0);
    }
    for(const auto item: items) {
        hk.addh(item);
    }
    is.clear();
    is.insert(items.begin(), items.end());
    //for(auto &i: is) i = hash::WangHash()(i);
    for(const auto item: is) {
        if(hk.queryh(item) == 0) ++missing;
    }
    assert(missing <= 100 || tbsz != defaulttbsz); // This could actually change and still be fine, but I want to know about these changes.
    std::fprintf(stderr, "missing 10x items: %zu of total %zu\n", missing, items.size());
    assert(missing < items.size() / 10);
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash().inverse(x);});
    for(const size_t item: items) {
        auto postq = hk.queryh(item);
        assert(postq <= ninsert);
    }
    std::for_each(items.begin(), items.end(), [](auto &x) {x = hash::WangHash()(x);});
    std::unordered_map<std::string, unsigned> sset;
    std::string line;
    for(const size_t item: items) {
        char buf[256];
        auto postq = hk.queryh(item);
        std::sprintf(buf, "After inverting back, %zu replaces %zu\n", size_t(postq), ninsert);
        line = buf;
        ++sset[line];
    }
    std::vector<std::pair<std::string, unsigned>> svec(sset.begin(), sset.end());
    std::sort(svec.begin(), svec.end(), [](const auto &x, const auto &y) {return x.second < y.second;});
    for(const auto &pair: svec) {
        std::fprintf(stderr, "%u:%s", pair.second, pair.first.data());
    }
    HeavyKeeper<32,32> hkcp(hk);
    HeavyKeeperHeap<HeavyKeeper<32,32>, uint64_t> hkh(20, std::move(hk));
    HeavyKeeperHeavyHitters<HeavyKeeper<32,32>, uint64_t> hkhh(.2, 20, std::move(hkcp));
    for(const size_t item: items) {
        hkh.addh(item);
        hkhh.addh(item);
    }
    auto c = hkh.to_container();
    for(const auto &x: std::get<0>(c)) std::fprintf(stderr, "Element: %zu. Count: %zu\n", size_t(x), size_t(hkh.est_count(x)));
}
