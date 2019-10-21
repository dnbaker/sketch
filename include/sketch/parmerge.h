#pragma once
#include <cstdint>


#if __CUDACC__
#include "parmerge.cu.h"
#else
#ifndef MERGE_SORTED_PARALLEL_CPU_H__
#define MERGE_SORTED_PARALLEL_CPU_H__
namespace sketch {
// https://www.cc.gatech.edu/sites/default/files/images/fast_and_adaptive_list_intersections_on_the_gpu.pdf
// http://www.adms-conf.org/p1-SCHLEGEL.pdf
}

#endif /* MERGE_SORTED_PARALLEL_CPU_H__ */

#endif /* #ifndef  __CUDACC__ */
