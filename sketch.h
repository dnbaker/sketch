#ifndef SKETCH_SINGLE_HEADER_H__
#define SKETCH_SINGLE_HEADER_H__
#include "hll.h"
#include "bf.h"
#include "cbf.h"
#include "mh.h"
#include "bbmh.h"
#include "ccm.h"
#include "cbf.h"
#include "mult.h"
#include "heap.h"
#include "filterhll.h"
#include "mult.h"
#include "sparse.h"

namespace sketch {
    // Flatten all classes to global sketch namespace.
    // Subnamespaces can still be subsampled
    using namespace hll;
    using namespace bf;
    using namespace minhash;
    using namespace fhll;
    using namespace common;
    using namespace cws;
    using namespace nt;
}

#endif /* SKETCH_SINGLE_HEADER_H__ */
