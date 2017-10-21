#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <getopt.h>
#include "hll.h"
#include "kthread.h"


using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;

static const size_t BITS = 25;


bool test_qty(size_t lim) {
    hll::hll_t t(BITS);
    for(size_t i(0); i < lim; t.addh(++i));
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

void usage() {
    std::fprintf(stderr, "Usage: ./test <flags>\nFlags:\n-p\tSet number of threads. [8].\n-b\tSet size of sketch. [1 << 18]\n");
    std::exit(EXIT_FAILURE);
}


/*
 * If no arguments are provided, runs test with 1 << 22 elements.
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    if(argc < 2) usage();
    using clock_t = std::chrono::system_clock;
    unsigned nt(8), pb(1 << 18);
    std::vector<std::uint64_t> vals;
    int c;
    while((c = getopt(argc, argv, "p:b:h")) >= 0) {
        switch(c) {
            case 'p': nt = atoi(optarg); break;
            case 'b': pb = atoi(optarg); break;
            case 'h': case '?': usage();
        }
    }
    for(c = optind; c < argc; ++c) vals.push_back(strtoull(argv[c], 0, 10));
    if(vals.empty()) vals.push_back(1ull<<(BITS+1));
    for(const auto val: vals) {
        hll::hll_t t(BITS);
#ifndef THREADSAFE
        for(size_t i(0); i < val; t.addh(i++));
#else
        kt_data data {t, val, (int)nt};
        kt_for(nt, &kt_helper, &data, (val + nt - 1) / nt);
#endif
        auto start(clock_t::now());
        t.parsum(nt, pb);
        auto end(clock_t::now());
        std::chrono::duration<double> timediff(end - start);
        fprintf(stderr, "Time diff: %lf\n", timediff.count());
        fprintf(stderr, "Quantity: %lf\n", t.report());
        auto startsum(clock_t::now());
        t.sum();
        auto endsum(clock_t::now());
        std::chrono::duration<double> timediffsum(endsum - startsum);
        fprintf(stderr, "Time diff not parallel: %lf\n", timediffsum.count());
        fprintf(stderr, "Using %i threads is %4lf%% as fast as 1.\n", nt, timediffsum.count() / timediff.count() * 100.);
        fprintf(stderr, "Quantity: %lf\n", t.report());
        fprintf(stderr, "Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s\n",
                val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false");
    }
	return EXIT_SUCCESS;
}
