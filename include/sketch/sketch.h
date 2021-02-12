#ifndef SKETCH_SINGLE_HEADER_H__
#define SKETCH_SINGLE_HEADER_H__
#include "./hll.h"
#include "./bf.h"
#include "./mh.h"
#include "./bbmh.h"
#include "./ccm.h"
#include "./cbf.h"
#include "./mult.h"
#include "./heap.h"
#include "./filterhll.h"
#include "./mult.h"
#include "./sparse.h"
#include "./dd.h"
#include "./hk.h"
#include "./vac.h"
#include "./hbb.h"
#include "./mod.h"
#include "./setsketch.h"

#ifdef __CUDACC__
#include "hllgpu.h"
#endif

namespace sketch {
    // Flatten all classes to global sketch namespace.
    // Subnamespaces can still be subsampled

    // Set representations
    using namespace hll; // HyperLogLog
    using namespace bf;  // Bloom Filters
    using namespace minhash; // Minhash
    using namespace fhll;    // Filtered HLLs

    // Multiplicities
    using namespace cws; // Consistent Weighted Sampling
    using namespace nt;  // ntcard
    using namespace wj;  // Weighted Jaccard adapters

    // Count point estimators
    using namespace hk;  // Heavy-Keeper
    using namespace cm; // Count/Count-Min

    // Utilities
    using namespace heap; // Heap maintainers for multisets based on a variety of criteria
    using namespace vac;  // Approximate Multiplicity samplers for streams
}

namespace sk = sketch;

#endif /* SKETCH_SINGLE_HEADER_H__ */
