#include "common.h"
#include "hk.h"
#include "vec/welford_sd.h"
#include <map>
using namespace sketch;
using namespace common;
using stats::OnlineSD;

size_t n2v(size_t n) {
    //return (((n>>8) & 0xFFu) == 0) ? std::pow((n & 3) + 1, 6): (n & 7) + 1;}
    return n & 1 ? 8: n & 2 ? 3: 1;
}

void usage() {
    std::fprintf(stderr, "Test HeavyKeeper structure.\n");
    std::fprintf(stderr, "Usage: hktest <N> <table size> <ntables>\n. All arguments are optional.\n");
    std::exit(1);
}

int main(int argc, char *argv[]) {
    if(std::find_if(argv, argv + argc, [](auto x) {return std::strcmp(x, "-h") == 0;}) != argv + argc)
        usage();
    size_t N = argc == 1 ?  10000: std::atoi(argv[1]);
    size_t hksz = argc <= 2 ?  20000: std::atoi(argv[2]);
    size_t ntables = argc <= 3 ? 4: std::atoi(argv[3]);
    HeavyKeeper<8,8> hk1(hksz, ntables);
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = n2v(i); j--;) {
          //size_t old = hk.query(i);
          hk.addh(i);
          //std::fprintf(stderr, "Adding. Old : %zu. New: %zu\n", old, hk.query(i));
        }
    }
    size_t negeqpos [] {0, 0, 0};
    OnlineSD<double> sd;
    std::map<float, uint32_t> ics;
    for(size_t i = 0; i < N; ++i) {
        auto expect = n2v(i);
        auto find = hk.query(i);
        ++negeqpos[expect == find ? 1: expect > find ? 0: 2];
        if(expect > 10)
            sd.add(((double(find) - expect) / expect));
        ++ics[expect - find];
    }
    size_t exact_right = negeqpos[1];
    std::fprintf(stderr, "Exactly correct: %zu (%f %% correct)\n", exact_right, 100. * double(exact_right) / N);
    std::fprintf(stderr, "Less than: %zu (%f %% less than)\n", negeqpos[0], 100. * double(negeqpos[0]) / N);
    assert(negeqpos[2] == 0);
    //std::fprintf(stderr, "Gt: %zu (%f %% greater than)\n", negeqpos[2], 100. * double(negeqpos[2]) / N);
    std::fprintf(stderr, "Difference stats: mu %f, sigma %f\n", sd.mean(), sd.stdev());
    for(const auto &[x, y]: ics) {
        std::fprintf(stderr, "diff: %d. count: %d/%f\n", int(x), int(y), float(y) / N);
    }
}
