#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cinttypes>
#include "hll.h"
#include "kthread.h"


using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;

static const size_t BITS = 24;


bool test_qty(size_t lim) {
    hll::hll_t t(BITS);
    size_t i(0);
    while(i < lim) t.addh(i++);
    return std::abs(t.report() - lim) <= t.est_err();
}

struct kt_data {
    hll::hll_t &hll_;
    const std::uint64_t n_;
    const int nt_;
};

void kt_helper(void *data, long index, int tid) {
    hll::hll_t &hll(((kt_data *)data)->hll_);
    const std::uint64_t todo((((kt_data *)data)->n_ + ((kt_data *)data)->nt_ - 1) / ((kt_data *)data)->nt_);
    for(std::uint64_t i(index * todo), e(std::min(((kt_data *)data)->n_, (index + 1) * todo)); i < e; hll.addh(i++));
}


/*
 * If no arguments are provided, runs test with 1 << 22 elements.
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    const int nt(8);
    std::vector<std::uint64_t> vals;
    for(char **p(argv + 1); *p; ++p) vals.push_back(strtoull(*p, 0, 10));
    if(vals.empty()) vals.push_back(1ull<<(BITS+1));
    for(const auto val: vals) {
        std::fprintf(stderr, "Processing val = %" PRIu64 "\n", val);
        using hll::detail::ertl_ml_estimate;
        hll::hll_t t(BITS, hll::ORIGINAL);
#ifdef NOT_THREADSAFE
        for(size_t i(0); i < val; t.addh(i++));
#else
        kt_data data {t, val, nt};
        kt_for(nt, &kt_helper, &data, (val + nt - 1) / nt);
#endif
        std::fprintf(stderr, "Calculating for val = %" PRIu64 "\n", val);
        fprintf(stderr, "Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s. Ertl ML estimate: %lf. Error ertl ML: %lf\n",
                val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false", ertl_ml_estimate(t), std::abs(ertl_ml_estimate(t) - val));
    }
	return EXIT_SUCCESS;
}
