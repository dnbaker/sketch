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
    fprintf(stderr, "Within bounds? %s\n", t.est_err() <= std::abs(lim - t.report()) ? "true": "false"); 
	return EXIT_SUCCESS;
}
