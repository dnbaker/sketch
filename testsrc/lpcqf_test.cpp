#include "sketch/lpcqf.h"

int main() {
    size_t nentered = 132;
    size_t ss = 128;
    sketch::LPCQF<uint32_t, 5, sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2> lp(ss);
    for(size_t i = 0; i < nentered; ++i) {
        lp.update(i, i + 1);
        std::fprintf(stderr, "Current estimate for %zu: %zu\n", i, size_t(lp.count_estimate(i)));
        assert(lp.count_estimate(i) <= (i + 1) + 128 * 4);
    }
    auto ip = lp.inner_product(lp);
    std::fprintf(stderr, "Inner product: %zu\n", size_t(ip));
    sketch::LPCQF<uint32_t, 0> lp_noquot(ss);
    for(size_t i = 0; i < nentered; ++i) {
        lp_noquot.update(i, i + 1);
        std::fprintf(stderr, "NoQuotientEstimate for %zu: %zu\n", i, size_t(lp_noquot.count_estimate(i)));
    }
}
