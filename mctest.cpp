#include "cmbf.h"
#include <unordered_map>


using namespace sketch::cmbf;

int main(int argc, char *argv[]) {
    cmbf_exp_t thing(4, 16, 1);
    cmbf_t thingexact(4, 16, 5);
    auto [x, y] = thing.est_memory_usage();
    std::fprintf(stderr, "stack space: %zu\theap space:%zu\n", x, y);
    size_t nitems = argc > 1 ? std::strtoull(argv[1], nullptr, 10): 100000;
    std::vector<uint64_t> items;
    std::mt19937_64 mt;
    while(items.size() < nitems) items.emplace_back(mt());
    for(const auto el: thing.ref()) {
        assert(unsigned(el) == 0);
    }
    auto &ref = thing.ref();
    for(const auto item: items) thing.add_conservative(item), thingexact.add_conservative(item);
    //, std::fprintf(stderr, "Item %" PRIu64 " has count %u\n", item, unsigned(thing.est_count(item)));
    std::unordered_map<uint64_t, uint64_t> histexact, histapprox;
    for(const auto j: items) {
        //std::fprintf(stderr, "est count: %zu\n", size_t(thing.est_count(j)));
        ++histapprox[thing.est_count(j)];
        ++histexact[thingexact.est_count(j)];
    }
    for(const auto &pair: histexact) {
        std::fprintf(stderr, "Exact %" PRIu64 "\t%" PRIu64 "\n", pair.first, pair.second);
    }
    for(const auto &pair: histapprox) {
        std::fprintf(stderr, "Approx %" PRIu64 "\t%" PRIu64 "\n", pair.first, pair.second);
    }
}
