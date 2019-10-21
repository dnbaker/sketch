#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cinttypes>
#include <getopt.h>
#include "hll.h"
#include "kthread.h"


using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;
using namespace sketch;
using namespace sketch::hll;

static const size_t BITS = 15;


bool test_qty(size_t lim) {
    hll_t t(BITS);
    for(size_t i(0); i < lim; t.addh(++i));
    return std::abs(t.report() - lim) <= t.est_err();
}

struct kt_data {
    hll_t &hll_;
    const std::uint64_t n_;
    const int nt_;
    const std::vector<std::uint64_t> &inputs_;
};

void kt_helper(void *data, long index, int tid) {
    hll_t &hll(((kt_data *)data)->hll_);
    const std::vector<std::uint64_t> &inputs(((kt_data *)data)->inputs_);
    const std::uint64_t todo((((kt_data *)data)->n_ + ((kt_data *)data)->nt_ - 1) / ((kt_data *)data)->nt_);
    const std::uint64_t end(std::min(((kt_data *)data)->n_, (index + 1) * todo));
    for(std::uint64_t i(index * todo); i < end; hll.addh(inputs[i++]));
}

void usage() {
    std::fprintf(stderr, "Usage: ./test <flags>\nFlags:\n-p\tSet number of threads. [8].\n-b\tSet size of sketch. [1 << 18]\n");
    std::exit(EXIT_FAILURE);
}

void estimate(hll_t &h1, hll_t &h2, dhll_t &h3, hlldub_t &h4, chlf_t<> &ch, std::uint64_t expected) {
    h1.sum(); h2.sum(); h3.sum(); h4.sum();
    std::fprintf(stderr, "Values\t%lf\t%lf\t%lf\t%lf\t%lf\t%" PRIu64 "\n", h1.report(), h2.report(), h3.report(), h4.report(), ch.chunk_report(), expected);
    std::fprintf(stderr, "EstErr\t%lf\t%lf\t%lf\t%lf\t%lf\t%" PRIu64 "\n", h1.est_err(), h2.est_err(), h3.est_err(), h4.est_err(), -1337., expected);
    std::fprintf(stderr, "Err\t%lf\t%lf\t%lf\t%lf\t%lf\t%" PRIu64 "\n", std::abs(h1.report() - expected), std::abs(h2.report() - expected), std::abs(h3.report() - expected), std::abs(h4.report() - expected), std::abs(ch.chunk_report() - expected), expected);
}


/*
 * If no arguments are provided, runs test with 1 << 22 elements.
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
    using clock_t = std::chrono::system_clock;
    unsigned nt(8), pb(1 << 15);
    std::vector<std::uint64_t> vals;
    int c;
    while((c = getopt(argc, argv, "e:p:b:h")) >= 0) {
        switch(c) {
            case 'p': nt = atoi(optarg); break;
            case 'b': pb = atoi(optarg); break;
            case 'h': case '?': usage();
        }
    }
    //if(argc < 3) usage();
    for(c = optind; c < argc; ++c) vals.push_back(strtoull(argv[c], 0, 10));
    if(vals.empty()) vals.push_back(1ull<<(BITS+1));
    std::vector<std::uint64_t> inputs;
    std::mt19937_64 gen(13);
    if(vals.size() == 1) {
        std::uniform_int_distribution<uint64_t> dist(0, 1ull << 28ull);
        while(vals.size() < 16) vals.push_back(dist(gen));
    }
    std::fprintf(stdout, "#Label\th1\th2\th3\n");
    for(const auto val: vals) {
        std::fprintf(stdout, "#Value = %" PRIu64 "\n", val);
        hll_t t(BITS), t2(BITS + 1), tcp1(BITS), tcp(BITS);
        dhll_t t3(BITS);
        hlldub_t t4(BITS);
        chlf_t<> contigt(3, EstimationMethod::ERTL_MLE, JointEstimationMethod::ERTL_JOINT_MLE, BITS);
        inputs.resize(val);
        for(auto &el: inputs) el = gen();
//#ifndef THREADSAFE
        for(const auto el: inputs) t.addh(el), t2.addh(el), t3.addh(el), t4.addh(el), tcp1.addh(el + 1), tcp.addh(el), contigt.addh(el);
#if 0
//#else
        kt_data data {t, val, (int)nt, inputs};
        kt_data data2{t2, val, (int)nt, inputs};
        kt_data data3{t3, val, (int)nt, inputs};
        kt_data data4{t4, val, (int)nt, inputs};
        kt_for(nt, &kt_helper, &data, (val + nt - 1) / nt);
        kt_for(nt, &kt_helper, &data2, (val + nt - 1) / nt);
        kt_for(nt, &kt_helper, &data3, (val + nt - 1) / nt);
        kt_for(nt, &kt_helper, &data4, (val + nt - 1) / nt);
//#endif
#endif
        auto start(clock_t::now());
        t.parsum(nt, pb);
        auto end(clock_t::now());
        std::chrono::duration<double> timediff(end - start);
        fprintf(stdout, "Time diff: %lf\n", timediff.count());
        fprintf(stdout, "Quantity: %lf\n", t.report());
        auto startsum(clock_t::now());
        t.sum();
        auto endsum(clock_t::now());
        t2.sum();
        t3.sum();
        std::chrono::duration<double> timediffsum(endsum - startsum);
        //fprintf(stderr, "Time diff not parallel: %lf\n", timediffsum.count());
        //fprintf(stderr, "Using %i threads is %4lf%% as fast as 1.\n", nt, timediffsum.count() / timediff.count() * 100.);
        estimate(t, t2, t3, t4, contigt, val);
        t.set_jestim(hll::ERTL_JOINT_MLE);
        tcp1.csum();
        tcp.csum();
        t.csum();
        auto cmp = ertl_joint(t, tcp1);
        std::fprintf(stderr, "Joint: %lf|%lf|%lf\n", cmp[0], cmp[1], cmp[2]);
        t.set_jestim(hll::ERTL_MLE);
        cmp = ertl_joint(t, tcp1);
        t.set_jestim(hll::ERTL_JOINT_MLE);
        std::fprintf(stderr, "Joint: %lf|%lf|%lf\n", cmp[0], cmp[1], cmp[2]);
        cmp = ertl_joint(t, tcp);
        assert(!cmp[0] && !cmp[1]);
    }
	return EXIT_SUCCESS;
}
