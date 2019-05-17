#ifndef MERGE_SORTED_PARALLEL_CUDA_H__
#define MERGE_SORTED_PARALLEL_CUDA_H__
#include "thrust/merge.h"

namespace sketch {

template<typename CountType>
struct Incrementer {
    CountType *ptr_;
    Incrementer(CountType *ptr): ptr_(ptr) {
    }
    void operator++(int) {
        ++*ptr_;
    }
    void operator++() {
        ++*ptr_;
    }
};

template<typename It1, typename It2, typename CountType=std::uint32_t>
__host__ __device__
void count_intersection(It1 start1, It1 end1, It2 start2, It2 end2, CountType *ptr=nullptr) {
    CountType val = 0;
    thrust::merge(start1, end1, start2, end2, Incrementer<CountType>(&val));
    if(!ptr) std::fprintf(stderr, "Count: %zu\n", size_t(val));
    else *ptr = val;
}

} // namespace sketch

#endif /* MERGE_SORTED_PARALLEL_H__ */
