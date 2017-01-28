#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "hll.h"

using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;

static const size_t BITS = 20;


bool test_qty(size_t lim) {
    hll::hll_t t(BITS);
    size_t i(0);
    while(i < lim) t.addh(i++);
    return std::abs(t.report() - lim) <= t.est_err();
}


/*
 * If no arguments are provided, runs test with 1 << 22 elements. 
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    hll::hll_t t(BITS);
    size_t i(0), lim(1 << 22);
    if(argc == 1) test_qty(1 << 22);
    for(char **p(argv + 1); *p; ++p) if(!test_qty(strtoull(*p, 0, 10))) fprintf(stderr, "Failed test with %s\n", *p);
    while(i < lim) t.addh(i++);
    fprintf(stderr, "Quantity expected: %zu. Quantity estimated: %lf. Error bounds: %lf.\n",
            i, t.report(), t.est_err());
    fprintf(stderr, "Within bounds? %s\n", t.est_err() <= std::abs(lim - t.report()) ? "true": "false"); 
	return EXIT_SUCCESS;
}
