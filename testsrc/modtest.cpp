#include "sketch/mod.h"
#include <iostream>

using namespace sketch;
int main() {
    modsketch_t<hash::WangHash, uint64_t,
                policy::SizeDivPolicy<uint64_t>,
                std::allocator<uint64_t>
                > mod(16);
    std::cerr << "div: " << mod.mod_.nelem() << '\n';
    assert(mod.add(1 << 16));
    assert(mod.finalize().size() == 1);
    std::mt19937_64 mt;
    for(size_t i = 0; i < 10000; ++i) {
        mod.addh(mt());
    }
    auto f = mod.finalize();
    std::cerr << "Now, out of 10001 items, f has size " << f.size() << '\n';
    auto modfocus = mod.reduce(2);
    std::cerr << "Now, out of 10001 items, f (reduced by 2) has size " << modfocus.finalize().size() << '\n';
}
