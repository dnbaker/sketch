#include <cstdlib>
#include <cstdio>
#include "hll.h"

int main() {
    hll::hll_t t(20);
    size_t i(0);
    for(; i < 1 << 22; ++i) t.addh(i);
    fprintf(stderr, "Do stuff\n");
    fprintf(stderr, "Quantity expected: %u. Quantity estimated: %lf. Error bounds: %lf.\n",
            i, t.report(), t.est_err());
}
