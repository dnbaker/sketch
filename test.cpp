#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "hll.h"

using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;


/*
 * If no arguments are provided, runs test with 1 << 22 elements. 
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    hll::hll_t t(20);
    size_t i(0), lim(1 << 22);
    if(argc > 1) lim = strtoull(argv[1], 0, 10);
    for(; i < lim; ++i) t.addh(i);
    fprintf(stderr, "Quantity expected: %u. Quantity estimated: %lf. Error bounds: %lf.\n",
            i, t.report(), t.est_err());
#ifdef TEST_MANUAL_VS_HARDWARE
    std::vector<uint64_t> rands;
    std::vector<int> lzs;
    const size_t nelem(1<<25);
    rands.reserve(nelem);
    lzs.reserve(nelem);
    while(rands.size() < rands.capacity()) rands.push_back((uint64_t)rand() | rand());
    size_t index(0);
    tp start(system_clock::now());
    for(auto el: rands) {
        lzs.push_back(hll::clz_manual(el));
    }
    tp end(system_clock::now());
    std::chrono::duration<double> elapsed_seconds = end-start;
    fprintf(stderr, "Time for manual: %lf\n", elapsed_seconds.count());
    fprintf(stderr, "Summed clzs: %llu\n", std::accumulate(std::begin(lzs), std::end(lzs), 0ull));
    lzs.resize(0);
    start = system_clock::now();
    for(auto el: rands) {
        lzs.push_back(hll::clz(el));
    }
    end = system_clock::now();
    elapsed_seconds = end-start;
    fprintf(stderr, "Time for intrinsic: %lf\n", elapsed_seconds.count());
    fprintf(stderr, "Summed clzs: %llu\n", std::accumulate(std::begin(lzs), std::end(lzs), 0ull));
#endif
	return EXIT_SUCCESS;
}
