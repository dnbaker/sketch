#include "common.h"
#include "hk.h"
#include "vec/welford_sd.h"
#include <map>
using namespace sketch;
using namespace common;
using stats::OnlineSD;

size_t n2v(size_t n) {
    static const hash::WangHash wh;
    //return (((n>>8) & 0xFFu) == 0) ? std::pow((n & 3) + 1, 6): (n & 7) + 1;}
    n = wh(n);
    return n & 63 ? 2: 8;
}

void usage() {
    std::fprintf(stderr, "Test HeavyKeeper structure.\n");
    std::fprintf(stderr, "Usage: hktest <N> <table size> <ntables>\n. All arguments are optional.\n");
    std::exit(1);
}

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argv + argc)
        usage();
    size_t N = argc == 1 ?   15000: std::atoi(argv[1]);
    size_t hksz = argc <= 2 ? 1000: std::atoi(argv[2]);
    size_t ntables = argc <= 3 ? 4: std::atoi(argv[3]);
    HeavyKeeper<32,32> hk1(hksz, ntables);
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = 2; j--;) {
            size_t old = hk1.query(i);
            size_t newnew = hk1.addh(i);
            size_t newv = hk1.query(i);
            assert(newnew == newv || !std::fprintf(stderr, "newnew: %zu. newv: %zu. old: %zu\n", newnew, newv, old));
            std::fprintf(stderr, "Adding. Old : %zu. New: %zu\n", old, hk1.query(i));
        }
        std::fprintf(stderr, "Adding. After adding: %zu\n", hk1.query(i));
    }
#if 0
    size_t negeqpos [] {0, 0, 0};
    OnlineSD<double> sd;
    std::map<float, uint32_t> ics;
    size_t missing = 0, found = 0;
    for(size_t i = 0; i < N; ++i) {
        auto expect = n2v(i);
        auto find = hk1.query(i);
        ++negeqpos[expect == find ? 1: expect > find ? 0: 2];
        if(expect > 5) {
            sd.add(((double(find) - expect) / expect));
            if(!find) ++missing;
            else      ++found;
        }
        ++ics[expect - find];
    }
    size_t exact_right = negeqpos[1];
    std::fprintf(stderr, "64s missing: %zu. Found (nonzero): %zu\n", missing, found);
    std::fprintf(stderr, "Exactly correct: %zu (%f %% correct)\n", exact_right, 100. * double(exact_right) / N);
    std::fprintf(stderr, "Less than: %zu (%f %% less than)\n", negeqpos[0], 100. * double(negeqpos[0]) / N);
    assert(negeqpos[2] == 0);
    //std::fprintf(stderr, "Gt: %zu (%f %% greater than)\n", negeqpos[2], 100. * double(negeqpos[2]) / N);
    std::fprintf(stderr, "Difference stats: mu %f, sigma %f\n", sd.mean(), sd.stdev());
    for(const auto &i: ics) {
        auto &x = i.first;
        auto &y = i.second;
        std::fprintf(stderr, "diff: %d. count: %d/%f\n", int(x), int(y), float(y) / N);
    }
#endif
}
