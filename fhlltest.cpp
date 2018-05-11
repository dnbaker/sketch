#include "filterhll.h"

using namespace sketch;
using namespace fhll;
using namespace bf;

int main() {
    fhll_t h(20, 8, 18, 1, 1337, 20);
    pcbf_t h2(20, 8, 4, 13337, 0);
    pcfhll_t h3(16, 10, 10, 20, 1, 1337, 32);
}
