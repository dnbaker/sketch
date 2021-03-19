#include "sketch/ccm.h"
//#include "sketch/mh.h"
#include <unordered_map>
#include <getopt.h>


using namespace sketch::cm;
using namespace sketch;
using sketch::hash::WangHash;

int main(int argc, char *argv[]) {
    int nbits = 12, l2sz = 18, nhashes = 3, c;
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
    ccm_t cmsexact(nbits, l2sz, nhashes), cmsexact2(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, DefaultCompactVectorType, WangHash, false> cmswithnonminmal(nbits, l2sz, nhashes);
    sketch::cm::ccmbase_t<update::Increment, std::vector<float, Allocator<float>>, WangHash, false> cmswithfloats(nbits, l2sz, nhashes);
    cs_t cmscs(l2sz, nhashes);
    cs4w_t cmscs4w(l2sz, nhashes), cmscs4w2(l2sz, nhashes);
    ccmbase_t<update::Increment, DefaultStaticCompactVectorType<4>> static_cm(nbits, l2sz, nhashes);
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
    std::mt19937_64 mt(nitems ^ (std::mt19937_64(nhashes)()));
    while(items.size() < nitems) items.emplace_back(mt());
    for(const auto el: cms.ref()) {
        assert(unsigned(el) == 0);
    }
    std::fprintf(stderr, "Inserted at first\n");
    //for(size_t i = 0; i < 10;++i)
    items.emplace_back(137);
    int TIMES = 4;
    for(const auto item: items) {
        for(int i = (item & 0xFF) == 0 ? 200: TIMES; i--;) {
            cmsexact.addh(item), cms.addh(item),  cmscs.addh(item), cmswithnonminmal.addh(item), cmscs4w.addh(item), cmsexact2.addh(item);
        }
    }
    std::fprintf(stderr, "Inserted items\n");
    for(size_t i = 1000; i--;cmscs.addh(137), cmsexact.addh(137), cms.addh(137), cmsexact2.addh(137), cmscs4w.addh(137));
    std::fprintf(stderr, "Inserted 137\n");
    //size_t true_is = items.size() + 1000;
    //size_t true_us = items.size() * 2 + 1000;
    std::fprintf(stderr, "All inserted\n");
    std::unordered_map<int64_t, uint64_t> histexact, histapprox, histcs, hist4w;
    size_t missing = 0;
    size_t tot = 0;
    double ssqe4w = 0., ssqe2w = 0.;
    for(const auto j: items) {
        if(j == 137) {
            std::fprintf(stderr, "approx: %i, exact %i, histcs %i, cmscs4w %i\n", int(cms.est_count(137)), int(cmsexact.est_count(137)), int(cmscs.est_count(137)), cmscs4w.addh(137));
            //assert(std::abs(ssize_t(cms.est_count(137)) - (1000 + TIMES)) <= 20 || nitems > 200000);
            assert(std::abs(ssize_t(cmsexact.est_count(137)) - (1000 + TIMES)) < 10 || nitems != 100000);
            assert(std::abs(ssize_t(cmscs.est_count(137)) - (1000 + TIMES)) < 10 || nitems != 100000);
            assert(std::abs(ssize_t(cmscs4w.est_count(137)) - (1000 + TIMES)) < 10 || nitems != 100000);
        }
        //std::fprintf(stderr, "est count: %zu\n", size_t(cms.est_count(j)));
        auto exact_count = j == 137 ? 1000 + TIMES: (j & 0xFF) == 0 ? 200: TIMES;
        ++histapprox[cms.est_count(j) - exact_count];
        ++histexact[cmsexact.est_count(j) - exact_count];
        ssize_t csest = cmscs.est_count(j);
        ++histcs[csest - exact_count];
        ++hist4w[cmscs4w.est_count(j) - exact_count];
        missing += csest == 0;
        ++tot;
        ssqe4w += std::pow(cmscs4w.est_count(j) - exact_count, 2);
        ssqe2w += std::pow(csest - exact_count, 2);
    }
    ssqe4w = std::sqrt(ssqe4w);
    ssqe2w = std::sqrt(ssqe2w);
    std::fprintf(stderr, "ssqe (2w): %f. ssqe (4w): %f\n", ssqe2w, ssqe4w);
    std::fprintf(stderr, "missing %zu of %zu\n", missing, items.size());
    std::vector<int64_t> hset;
    for(const auto &pair: histexact) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Exact %" PRIi64 "\t%" PRIu64 "\n", k, histexact[k]);
    }
    hset.clear();
    std::fprintf(stderr, "Did the hset cms\n");
    for(const auto &pair: histapprox) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Approx %" PRIi64 "\t%" PRIu64 "\n", k, histapprox[k]);
    }
    std::vector<uint64_t> items2(items.size());
    for(auto &i: items2) {
        i = mt();
    }
    std::fprintf(stderr, "Total of false positives from item2: %zu/%zu\n",
                 std::accumulate(items2.begin(), items2.end(), size_t(0), [&](size_t s, auto x) {return s + (cms.est_count(x) != 0);}),
                 size_t(nitems));
    for(const auto i: items) for(int j = TIMES; j--;cmsexact2.addh(i), cmscs4w2.addh(i));
    hset.clear();
    for(const auto &pair: histcs) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Count sketch %" PRIi64 "\t%s\n", k, std::to_string(size_t(histcs[k])).data());
    }
    std::fprintf(stderr, "Estimated count for 137: %d\n", cmscs.est_count(137));
    hset.clear();
    for(const auto &pair: hist4w) hset.push_back(pair.first);
    std::sort(hset.begin(), hset.end());
    for(const auto k: hset) {
        std::fprintf(stderr, "Count sketch4w %" PRIi64 "\t%s\n", k, std::to_string(size_t(hist4w[k])).data());
    }
    KWiseIndependentPolynomialHash<4> hf; // Just to test compilation
    std::fprintf(stderr, "l2 join size needs further debugging, not doing\n");
    double nonmin = cmswithnonminmal.l2est();
    std::fprintf(stderr, "nonminimal update info l2 join size: %lf\n", nonmin);
    auto composed4w = cmscs4w + cmscs4w2;
    auto folded_composed1 = composed4w.fold(1);
    std::fprintf(stderr, "folded with 1\n");
    auto folded_composed2 = composed4w.fold(2);
    std::fprintf(stderr, "folded with 2\n");
}
