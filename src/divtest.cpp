#include "common.h"
#include "hk.h"
using namespace sketch;
using namespace common;

int main() {
    for(size_t i = 10; i < 100000; i *= 10) {
        schism::Schismatic<uint64_t> div_(i);
        for(size_t j = 1000000; j < 1200000; j += (std::rand() & 0xFFU)) {
            auto div = div_.div(j);
            auto mod = div_.mod(j);
            assert(div == j / i);
            assert(mod == j % i);
        }
    }
    policy::SizePow2Policy<uint64_t> pol(1000);
    for(size_t j = 10; j < 10000000; j *= 10) {
        policy::SizePow2Policy<uint64_t> p1(j);
        policy::SizeDivPolicy<uint64_t> p2(j);
        std::fprintf(stderr, "input: %zu. power of two: %zu. div: %zu.\n", j, p1.nelem(), p2.nelem());
    }
    HeavyKeeper<16,48> hk(1000, 8);
    constexpr size_t N = 20000;
    for(size_t i = 0; i < N; ++i) {
        for(size_t j = (i & 15u) + 1; j--;hk.addh(i));
    }
    size_t exact_right = N;
    for(size_t i = 0; i < N; ++i) {
        if((i & 15u) + 1 != hk.query(i)) {
            --exact_right;
        }
    }
}
