#ifndef OMH_H
#define OMH_H
#include "circularqueue/cq.h"
#include "sketch/mh.h"

namespace sketch {
template<typename Hash=hash::WangHash>
struct OrderedMinHash {
    // Assuming that the sequence is sketched in order
    // This performs a minhash over contiguous t-tuples of windows,
    // and the bottom-k sketches provide
    // estimators for the edit distance LSH (OMH) by Guillaume Marcais
    // https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6612865/
    using MinHasher = minhash::BottomKHasher<Hash>;
    MinHasher h_;
    circ::deque<uint64_t, uint32_t> q_;
    int t_;
    OrderedMinHash(int t, size_t k): h_(k), q_(t), t_(t) {
    }
    void update(uint64_t key) {
        q_.push_back(key);
        if(q_.size() > t_) q_.pop_front();
        else if(q_.size() < t_) return;
        Hash hash;
        auto it = q_.begin();
        uint64_t hv = hash(*it);
        for(int i = t_; i--;) {
            ++it;
            hv ^= hash(*it);
        }
        h_.add(hv);
    }
    using final_type = typename MinHasher::final_type;
    operator const std::vector<uint64_t, Allocator<uint64_t>> & () const {
        return this->h_.mpq_.getq();
    }
    final_type finalize() const {
        return h_.finalize();
    }
};

}
#endif
