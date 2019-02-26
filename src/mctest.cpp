#include "ccm.h"
#include "mh.h"
#include <unordered_map>
#include <getopt.h>


using namespace sketch::cm;

int main(int argc, char *argv[]) {
    int nbits = 8, l2sz = 16, nhashes = 8, c;
    //if(argc == 1) goto usage;
    while((c = getopt(argc, argv, "n:l:b:h")) >= 0) {
        switch(c) {
            case 'h':
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
    pccm_t thing(nbits >> 1, l2sz, nhashes);
    ccm_t thingexact(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, DefaultCompactVectorType, sketch::common::WangHash, false> thingwithnonminmal(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, std::vector<float, Allocator<float>>, sketch::common::WangHash, false> thingwithfloats(nbits, l2sz, nhashes);
    cs_t thingcs(l2sz, nhashes);
    sketch::mh::RangeMinHash<uint64_t> rm(1 << l2sz);
#if __cplusplus >= 201703L
    auto [x, y] = thing.est_memory_usage();
#else
    auto p = thing.est_memory_usage();
    auto x = p.first; auto y = p.second;
#endif
    std::fprintf(stderr, "probabilistic method stack space: %zu\theap space:%zu\n", x, y);
    std::tie(x, y) = thingexact.est_memory_usage();
    std::fprintf(stderr, "exact method stack space: %zu\theap space:%zu\n", x, y);
    size_t nitems = optind == argc - 1 ? std::strtoull(argv[optind], nullptr, 10): 100000;
    std::vector<uint64_t> items;
    std::mt19937_64 mt;
    while(items.size() < nitems) items.emplace_back(mt());
    for(const auto el: thing.ref()) {
        assert(unsigned(el) == 0);
    }
    //for(size_t i = 0; i < 10;++i)
    for(const auto item: items) thing.addh(item), thingexact.addh(item), thingcs.addh(item);
    for(size_t i = 100; i--;thingcs.addh(137), thingexact.addh(137));
    std::fprintf(stderr, "All inserted\n");
    std::unordered_map<int64_t, uint64_t> histexact, histapprox, histcs;
    size_t missing = 0;
    size_t tot = 0;
    for(const auto j: items) {
        //std::fprintf(stderr, "est count: %zu\n", size_t(thing.est_count(j)));
        ++histapprox[thing.est_count(j)];
        ++histexact[thingexact.est_count(j)];
        ++histcs[thingcs.est_count(j)];
        missing += thingcs.est_count(j) == 0;
        ++tot;
    }
    std::fprintf(stderr, "Missing %zu/%zu\n", missing, tot);
    std::vector<int64_t> hset;
    for(const auto &pair: histexact) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Exact %" PRIi64 "\t%" PRIu64 "\n", k, histexact[k]);
    }
    hset.clear();
    std::fprintf(stderr, "Did th hset thing\n");
    for(const auto &pair: histapprox) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Approx %" PRIi64 "\t%" PRIu64 "\n", k, histapprox[k]);
    }
    hset.clear();
    for(const auto &pair: histcs) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Count sketch %" PRIi64 "\t%" PRIu64 "\n", k, histcs[k]);
    }
    std::fprintf(stderr, "Estimated count for 137: %d\n", thingcs.est_count(137));
    KWiseIndependentPolynomialHash<4> hf; // Just to test compilation
    thingwithnonminmal.l2est();
    thingwithfloats.l2est();
}
