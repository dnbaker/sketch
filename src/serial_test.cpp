#ifndef HLL_HEADER_ONLY
#define HLL_HEADER_ONLY
#endif
#include "hll.h"
using namespace sketch;
using namespace hll;

int main(int argc, char *argv[]) {
    unsigned nelem(argc > 1 ? std::atoi(argv[1]): 1000000);
    hll_t h(18);
    for(unsigned i(0); i < nelem; h.addh(++i));
    h.sum();
    std::fprintf(stderr, "h count: %lf. String: %s\n", h.report(), h.to_string().data());
    h.not_ready();
    h.set_estim(hll::ORIGINAL);
    h.write("SaveSketch.hll");
    hll_t h2("SaveSketch.hll");
    std::fprintf(stderr, "h2 count: %lf. String: %s\n", h2.report(), h2.to_string().data());
    h2.sum();
    std::fprintf(stderr, "After resumming: h2 count: %lf. String: %s\n", h2.report(), h2.to_string().data());
}
