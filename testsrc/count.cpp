#include "sketch/count_eq.h"

template<typename T>
int do_main() {
    std::vector<T> rhs(100), lhs(100);
    std::iota(rhs.begin(), rhs.end(), 0u);
    std::iota(lhs.begin(), lhs.end(), 0u);
    lhs[17] = 1;
    auto nm = sketch::eq::count_eq(lhs.data(), rhs.data(), 100);
    std::fprintf(stderr, "nmatched: %zu\n", nm);
    if(sizeof(T) == 1) {
        std::memset(&rhs[0], 0, 100);
        std::memset(&lhs[0], 0, 100);
        auto nm = sketch::eq::count_eq_nibbles((const uint8_t *)lhs.data(), (const uint8_t *)rhs.data(), 200);
        std::fprintf(stderr, "nm: %zu\n", nm);
        assert(nm == 200);
    }
    return nm != 99;
}

int main() {
    return do_main<uint16_t>() || do_main<uint32_t>() || do_main<uint64_t>() || do_main<uint8_t>();
}
