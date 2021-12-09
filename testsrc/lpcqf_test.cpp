#include "lpcqf.h"

int main() {
    size_t nentered = 150;
    size_t ss = 128;
    sketch::LPCQF<uint32_t, 5, sketch::IS_QUADRATIC_PROBING> lp(ss);
    for(size_t i = 0; i < nentered; ++i) {
        lp.update(i, i + 1);
    }
    for(size_t i = 0; i < nentered; ++i) {
        std::fprintf(stderr, "Current estimate for %zu: %zu\n", i, lp.count_estimate(i));
    }
    auto ip = lp.inner_product(lp);
    std::fprintf(stderr, "Inner product: %zu\n", size_t(ip));
}
