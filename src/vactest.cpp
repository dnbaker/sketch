#include "vac.h"

int main() {
    sketch::vac::HVAC h(10, 10), h2(10, 10);
    sketch::vac::PowerHVAC ph(1.08, 10, 10), ph2(1.08, 10, 10);
    for(size_t i = 0; i < 10000; ++i)
        h.addh(i), h2.addh(std::rand()), ph.addh(i), ph2.addh(std::rand());
    auto h3 = h + h2;
    for(auto &sub: h3.sketches_)
        sub.sum(), std::fprintf(stderr, "Union card %lf\n", sub.report());
}
