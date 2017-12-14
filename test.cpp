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

void estimate(hll::hll_t &h1, hll::hll_t &h2, hll::dhll_t &h3, std::uint64_t expected) {
    h1.sum(); h2.sum(); h3.sum();
    std::fprintf(stderr, "Values\t%lf\t%lf\t%lf\t%" PRIu64 "\n", h1.report(), h2.report(), h3.report(), expected);
    std::fprintf(stderr, "EstErr\t%lf\t%lf\t%lf\t%" PRIu64 "\n", h1.est_err(), h2.est_err(), h3.est_err(), expected);
    std::fprintf(stderr, "Err\t%lf\t%lf\t%lf\t%" PRIu64 "\n", std::abs(h1.report() - expected), std::abs(h2.report() - expected), std::abs(h3.report() - expected), expected);
}


/*
 * If no arguments are provided, runs test with 1 << 22 elements.
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    if(argc < 2) usage();
    using clock_t = std::chrono::system_clock;
    unsigned nt(8), pb(1 << 18);
    double eps(1e-10);
    std::vector<std::uint64_t> vals;
    int c;
    while((c = getopt(argc, argv, "e:p:b:h")) >= 0) {
        switch(c) {
            case 'e': eps = atof(optarg); break;
            case 'p': nt = atoi(optarg); break;
            case 'b': pb = atoi(optarg); break;
            case 'h': case '?': usage();
        }
    }
    for(c = optind; c < argc; ++c) vals.push_back(strtoull(argv[c], 0, 10));
    if(vals.empty()) vals.push_back(1ull<<(BITS+1));
    std::fprintf(stderr, "#Label\th1\th2\th3\n");
    for(const auto val: vals) {
        std::fprintf(stderr, "#Value = %" PRIu64 "\n", val);
        hll::hll_t t(BITS), t2(BITS + 1);
        hll::dhll_t t3(BITS);
#ifndef THREADSAFE
        for(size_t i(0); i < val; t.addh(i), t2.addh(i), t3.addh(i), ++i);
#else
        kt_data data {t, val, (int)nt};
        kt_data data2{t2, val, (int)nt};
        kt_data data3{t3, val, (int)nt};
        kt_for(nt, &kt_helper, &data, (val + nt - 1) / nt);
        kt_for(nt, &kt_helper, &data2, (val + nt - 1) / nt);
        kt_for(nt, &kt_helper, &data3, (val + nt - 1) / nt);
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
        t2.sum();
        t3.sum();
        std::chrono::duration<double> timediffsum(endsum - startsum);
        //fprintf(stderr, "Time diff not parallel: %lf\n", timediffsum.count());
        //fprintf(stderr, "Using %i threads is %4lf%% as fast as 1.\n", nt, timediffsum.count() / timediff.count() * 100.);
        estimate(t, t2, t3, val);
    }
	return EXIT_SUCCESS;
}
