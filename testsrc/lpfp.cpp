#include "sketch/lpcqf.h"

template<typename FT>
int submain() {
    size_t nentered = 128;
    size_t ss = 128;
    sketch::LPCQF<FT, sizeof(FT) * 4, sketch::IS_QUADRATIC_PROBING> lpf(ss);
    for(size_t i = 0; i < 10; ++i) {
        lpf.update(i, i + 1);
        std::fprintf(stderr, "Inserted %zu, got %g as estimate back\n", i + 1, lpf.count_estimate(i));
    }
    auto ip = lpf.inner_product(lpf);
    std::fprintf(stderr, "ip: %g\n", ip);
    return 0;
}

int main() {
    return submain<float>() || submain<double>();
}
