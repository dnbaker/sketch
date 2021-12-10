#include "sketch/lpcqf.h"

int main() {
    size_t nentered = 132;
    size_t ss = 128;
    sketch::LPCQF<uint32_t, 5, sketch::IS_QUADRATIC_PROBING | sketch::IS_POW2> lp(ss);
    for(size_t i = 0; i < nentered; ++i) {
        lp.update(i, i + 1);
    }
    for(size_t i = 0; i < nentered; ++i) {
        std::fprintf(stderr, "Current estimate for %zu: %zu\n", i, size_t(lp.count_estimate(i)));
    }
    auto ip = lp.inner_product(lp);
    std::fprintf(stderr, "Inner product: %zu\n", size_t(ip));
    sketch::LPCQF<double, 32, sketch::IS_QUADRATIC_PROBING> lpf(ss);
    for(size_t i = 0; i < 10; ++i) {
        lpf.update(i, i + 1);
#if 0
        auto out = lpf.count_estimate(i);
        //f<decltype(out)> d;
        std::fprintf(stderr, "Input: %zu. Out: %g\n", i + 1, out);
        assert(lp.count_estimate(i) == i + 1);
#endif
    }
}
