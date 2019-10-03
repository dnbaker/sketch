#include "common.h"

int main() {
    std::fprintf(stderr, "version: %s. major: %d. minor: %d\n", SKETCH_VERSION_STR, SKETCH_MAJOR, SKETCH_MINOR);
}
