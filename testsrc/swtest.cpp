#include "sketch/ccm.h"
#include "circularqueue/cq.h"
using namespace sketch;
using CT = cm::cs4wbase_t<int32_t>;

int main() {
    cm::SlidingWindow<CT> sw(10000, CT(10, 4));
    cm::SlidingWindow<CT, circ::deque, unsigned> swd(10000, CT(10, 4));
    for(size_t i = 0; i < 100000; ++i)
        sw.addh(i), swd.addh(i);
}
