#include "ccm.h"
#include "mh.h"
#include <unordered_map>
#include <getopt.h>


using namespace sketch::cm;

int main(int argc, char *argv[]) {
    int nbits = 10, l2sz = 16, nhashes = 8, c;
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
    pccm_t cms(nbits >> 1, l2sz, nhashes);
    ccm_t cmsexact(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, DefaultCompactVectorType, sketch::common::WangHash, false> cmswithnonminmal(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, std::vector<float, Allocator<float>>, sketch::common::WangHash, false> cmswithfloats(nbits, l2sz, nhashes);
    cs_t cmscs(l2sz, nhashes * 4);
    sketch::mh::RangeMinHash<uint64_t> rm(1 << l2sz);
#if __cplusplus >= 201703L
    auto [x, y] = cms.est_memory_usage();
#else
    auto p = cms.est_memory_usage();
    auto x = p.first; auto y = p.second;
#endif
    std::fprintf(stderr, "probabilistic method stack space: %zu\theap space:%zu\n", x, y);
    std::tie(x, y) = cmsexact.est_memory_usage();
    std::fprintf(stderr, "exact method stack space: %zu\theap space:%zu\n", x, y);
    size_t nitems = optind == argc - 1 ? std::strtoull(argv[optind], nullptr, 10): 100000;
    std::vector<uint64_t> items;
    std::mt19937_64 mt;
    while(items.size() < nitems) items.emplace_back(mt());
    for(const auto el: cms.ref()) {
        assert(unsigned(el) == 0);
    }
    //for(size_t i = 0; i < 10;++i)
    items.emplace_back(137);
    for(const auto item: items) cmsexact.addh(item), cms.addh(item), cmscs.addh(item), cmswithnonminmal.addh(item);
    for(size_t i = 1000; i--;cmscs.addh(137), cmsexact.addh(137), cms.addh(137));
    std::fprintf(stderr, "All inserted\n");
    std::unordered_map<int64_t, uint64_t> histexact, histapprox, histcs;
    size_t missing = 0;
    size_t tot = 0;
    for(const auto j: items) {
        if(j == 137) {
            std::fprintf(stderr, "approx: %i, exact %i, histcs %i\n", int(cms.est_count(137)), int(cmsexact.est_count(137)), int(cmscs.est_count(137)));
        }
        //std::fprintf(stderr, "est count: %zu\n", size_t(cms.est_count(j)));
        ++histapprox[cms.est_count(j)];
        ++histexact[cmsexact.est_count(j)];
        ssize_t csest = cmscs.est_count(j);
        ++histcs[csest];
        missing += csest == 0;
        ++tot;
    }
    std::vector<int64_t> hset;
    for(const auto &pair: histexact) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Exact %" PRIi64 "\t%" PRIu64 "\n", k, histexact[k]);
    }
    hset.clear();
    std::fprintf(stderr, "Did th hset cms\n");
    for(const auto &pair: histapprox) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Approx %" PRIi64 "\t%" PRIu64 "\n", k, histapprox[k]);
    }
    hset.clear();
    for(const auto &pair: histcs) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Count sketch %" PRIi64 "\t%s\n", k, std::to_string(size_t(histcs[k])).data());
    }
    std::fprintf(stderr, "Estimated count for 137: %d\n", cmscs.est_count(137));
    KWiseIndependentPolynomialHash<4> hf; // Just to test compilation
    std::fprintf(stderr, "l2 join size needs further debugging, not doing\n");
    double nonmin = cmswithnonminmal.l2est();
    std::fprintf(stderr, "nonminimal update info l2 join size: %lf\n", nonmin);
#if 0
    double nonmin_man = 0;
    cmswithnonminmal.for_each_register([&](const auto &x) {nonmin_man += x * x;});
    nonmin_man = std::sqrt(nonmin_man);
    double twf = cmswithfloats.l2est();
    nonmin_man = 0;
    cmswithfloats.for_each_register([&](const auto &x) {nonmin_man += x * x;});
    nonmin_man = std::sqrt(nonmin_man);
    std::fprintf(stderr, "float: %lf. man: %lf\n", twf, nonmin_man);
    std::fprintf(stderr, "cmsexact: %lf\n", cmsexact.l2est());
#endif
}
