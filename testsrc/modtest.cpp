#include "sketch/mod.h"

using namespace sketch;
int main() {
    modsketch_t<> mod(16);
    assert(mod.addh(1 << 16));
    auto f = mod.finalize();
    assert(f.size() == 1);
}
