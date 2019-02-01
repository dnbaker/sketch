#include "bbmh.h"
using namespace sketch;

int main() {
    mh::BBitMinHasher<uint64_t> tmp(4, 14);
    auto f = tmp.finalize();
}
