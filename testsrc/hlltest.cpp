#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cinttypes>
#include "hll.h"
#include "mh.h"
#include "hbb.h"
#include "kthread.h"


using namespace std::chrono;

using tp = std::chrono::system_clock::time_point;

static const size_t DEFAULT_BITS = 10;

using namespace sketch;

bool test_qty(size_t lim) {
    hll::hll_t t(DEFAULT_BITS);
    for(size_t i = 0;i < lim; t.addh(i++));
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

//using hll = namespace sketch::hll;

/*
 * If no arguments are provided, runs test with 1 << 22 elements.
 * Otherwise, it parses the first argument and tests that integer.
 */

int main(int argc, char *argv[]) {
#ifdef SKETCH_HMH2_H__
    sketch::HyperMinHash mh(10, 16), mh2(12, 16);
    sketch::HyperMinHash mh3(10, 16); mh3 += mh;
    mh.addh(uint64_t(1337));
#endif
    std::vector<std::uint64_t> vals;
    for(char **p(argv + 1); *p; ++p) vals.push_back(strtoull(*p, 0, 10));
    if(vals.empty()) vals.push_back(1000000);
    std::fprintf(stderr, "Using %s vectorization\n",
#if VEC_DISABLED__
     "no"
#else
    "native"
#endif
    );
    for(const int32_t nbits: {10, 12, 14, 16, 20}) {
        for(const auto val: vals) {
#if VERBOSE_AF
            std::fprintf(stderr, "Processing val = %" PRIu64 "\n", val);
#endif
            using hll::detail::ertl_ml_estimate;
            hll::hll_t t(nbits, hll::ORIGINAL);
            hll::hllbase_t<hll::MurFinHash> tmf(nbits, hll::ORIGINAL);
            for(size_t i(0); i < val; t.addh(i), tmf.addh(i++));
            std::fprintf(stderr, "Calculating for val = %" PRIu64 " and nbits = %d\n", val, nbits);
            t.csum();
            tmf.csum();
            fprintf(stderr, "Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s. Ertl ML estimate: %lf. Error ertl ML: %lf\n",
                    val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false", ertl_ml_estimate(t), std::abs(ertl_ml_estimate(t) - val));
            t.not_ready();
            t.set_estim(hll::ORIGINAL);
            t.csum();
            assert(t.est_err() * 2. >= std::abs(val - t.report()));
            fprintf(stderr, "ORIGINAL Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s. Ertl ML estimate: %lf. Error ertl ML: %lf\n",
                    val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false", ertl_ml_estimate(t), std::abs(ertl_ml_estimate(t) - val));
            t.set_estim(hll::ERTL_JOINT_MLE);
            t.not_ready();
            t.csum();
            fprintf(stderr, "JMLE Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s. Ertl ML estimate: %lf. Error ertl ML: %lf\n",
                    val, t.report(), t.est_err(), std::abs(val - t.report()), t.est_err() >= std::abs(val - t.report()) ? "true": "false", ertl_ml_estimate(t), std::abs(ertl_ml_estimate(t) - val));
            assert(t.est_err() * 2. >= std::abs(val - t.report()));
            // assert(t.est_err() >= /*2. * */std::abs(val - t.report()));
            fprintf(stderr, "Quantity expected: %" PRIu64 ". Quantity estimated: %lf. Error bounds: %lf. Error: %lf. Within bounds? %s. Ertl ML estimate: %lf. Error ertl ML: %lf\n",
                    val, tmf.report(), tmf.est_err(), std::abs(val - tmf.report()), tmf.est_err() >= std::abs(val - tmf.report()) ? "true": "false", ertl_ml_estimate(tmf), std::abs(ertl_ml_estimate(tmf) - val));
#ifndef VEC_DISABLED__
            hll::VType tmpv = static_cast<uint64_t>(1337);
            t.addh(tmpv);
            tmf.addh(tmpv);
#endif
            auto mini = t.compress(4);
        }
    }
	return EXIT_SUCCESS;
}
