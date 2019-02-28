#include "mult.h"
using namespace sketch;
using namespace cws;
int main () {
    common::DefaultRNGType gen;
    CWSamples<> zomg(100, 1000);
    realccm_t<> rc(0.999, 10, 20, 8);
    nt::VecCard<uint16_t> vc(13, 10), vc2(13, 10);
    for(size_t i = 0; i < 100000; ++i)
        vc.addh(gen()), vc2.addh(gen());
    auto vc3 = vc + vc2;
    auto zomg2 = vc.report();
    auto zomg3 = vc3.report();
}
