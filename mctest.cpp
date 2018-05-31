#include "ccm.h"
#include <unordered_map>
#include <getopt.h>


using namespace sketch::cm;

int main(int argc, char *argv[]) {
    int nbits = 4, l2sz = 16, nhashes = 8, c;
    //if(argc == 1) goto usage;
    while((c = getopt(argc, argv, "n:l:b:h")) >= 0) {
        switch(c) {
            case 'h':
                usage:
                std::fprintf(stderr, "%s [flags] [niter=10000]\n"
                                     "-n\tNumber of subtables\n"
                                     "-l\tLog2 size of the sketch\n"
                                     "-b\tNumber of bits per entry\n",
                              *argv);
                std::exit(1);
            case 'n': nhashes = std::atoi(optarg); break;
            case 'b': nbits = std::atoi(optarg); break;
            case 'l': l2sz = std::atoi(optarg); break;
        }
    }
    pccm_t thing(nbits, l2sz, nhashes);
    ccm_t thingexact(nbits, l2sz, nhashes);
    auto [x, y] = thing.est_memory_usage();
    std::fprintf(stderr, "stack space: %zu\theap space:%zu\n", x, y);
    size_t nitems = optind == argc - 1 ? std::strtoull(argv[optind], nullptr, 10): 100000;
    std::vector<uint64_t> items;
    std::mt19937_64 mt;
    while(items.size() < nitems) items.emplace_back(mt());
    for(const auto el: thing.ref()) {
        assert(unsigned(el) == 0);
    }
    for(const auto item: items) thing.add_conservative(item), thingexact.add_conservative(item);
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
