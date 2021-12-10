#include "sketch/lpcqf.h"

template<typename FT>
int submain() {
    size_t nentered = 32;
    size_t ss = 32;
    sketch::LPCQF<FT, sizeof(FT) * 4, sketch::IS_POW2> lpf(ss);
    for(size_t i = 0; i < nentered; ++i) {
        size_t inserted_key = nentered - i - 1;
        lpf.update(inserted_key, i + 1);
        std::fprintf(stderr, "%c:Inserted %zu, got %g as estimate back\n", sizeof(FT) == 4 ? 'f': 'd',
        i + 1,
        double(lpf.count_estimate(inserted_key)));
    }
    auto ip = lpf.inner_product(lpf);
    std::fprintf(stderr, "ip: %g\n", ip);
    return 0;
}

int main() {
    return submain<float>()
        || submain<double>();
}
