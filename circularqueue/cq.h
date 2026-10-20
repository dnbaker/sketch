// Compatibility path: before 0.20 this directory was the circularqueue
// submodule, and downstream code (bonsai's test/cbuf.cpp, via its hll
// submodule) includes "hll/circularqueue/cq.h". The maintained header is
// include/circularqueue/cq.h.
#include "../include/circularqueue/cq.h"
