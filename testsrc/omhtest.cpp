#include "include/sketch/omh.h"

using namespace sketch;
int main() {
    OrderedMinHash<> omh(10, 5);
    FinalRMinHash<uint64_t, Allocator<uint64_t>> f = omh.finalize();
}
