#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "hll.h"


using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;

static const size_t BITS = 22;


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
    std::vector<std::uint64_t> vals;
    for(char **p(argv + 1); *p; ++p) vals.push_back(strtoull(*p, 0, 10));
    if(vals.empty()) vals.push_back(1<<BITS);
    for(auto val: vals) {
        hll::hll_t t(BITS);
        for(size_t i(0); i < val; t.addh(i++));
        fprintf(stderr, "Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s\n",
                val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false");
    }
	return EXIT_SUCCESS;
}
