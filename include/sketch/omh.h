#include <vector>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include "sketch/fy.h"
#include "xxHash/xxh3.h"

#include "flat_hash_map/flat_hash_map.hpp"


namespace sketch {

namespace omh {

using std::size_t;
using std::uint64_t;
using std::uint32_t;

template<typename FT=double>
struct mvt_t {
    // https://arxiv.org/pdf/1802.03914v2.pdf, algorithm 5,
    // and https://arxiv.org/pdf/1911.00675.pdf, algorithm 4
    std::vector<FT> data_;
    mvt_t(size_t m, const FT maxv=std::numeric_limits<FT>::max()): data_((m << 1) - 1, maxv)
    {
    }


    FT *data() {return data_.data();}
    const FT *data() const {return data_.data();}
    // Check size and max
    size_t getm() const {return (data_.size() >> 1) + 1;}
    FT max() const {return data_.back();}
    FT operator[](size_t i) const {return data_[i];}
    void reset() {
        std::fill(data_.begin(), data_.end(), std::numeric_limits<FT>::max());
    }

    bool update(size_t index, FT x) {
        const auto sz = data_.size();
        const auto mv = getm();
        if(x < data_[index]) {
            do {
                data_[index] = x;
                index = mv + (index >> 1);
                if(index >= sz) break;
                size_t lhi = (index - mv) << 1;
                size_t rhi = lhi + 1;
                x = std::max(data_[lhi], data_[rhi]);
            } while(x < data_[index]);
            return true;
        }
        return false;
    }
};

template<typename FT=double> 
struct OMHasher {
private:
    size_t m_, l_;
    std::vector<uint64_t> indices;
    std::vector<FT>      vals;
    mvt_t<FT>             mvt;
    ska::flat_hash_map<uint64_t, uint32_t> counter;
    fy::LazyShuffler ls_;
    
    bool sub_update(const uint64_t pos, const FT value, const uint64_t element_idx) {
        if(value >= vals[l_ * (pos + 1) - 1]) return false;
        auto start = l_ * pos, stop = start + l_ - 1;
        uint64_t ix;
        for(ix = stop;ix > start && value < vals[ix - 1]; --ix) {
            vals[ix] = vals[ix - 1];
            indices[ix] = indices[ix - 1];
        }
        vals[ix] = value;
        indices[ix] = element_idx;
        return mvt.update(pos, vals[stop]);
    }

    void update(const uint64_t item, const uint64_t item_index) {
        uint64_t rng = item;
        uint64_t hv = wy::wyhash64_stateless(&rng);
        auto it = counter.find(hv);
        if(it == counter.end()) it = counter.emplace(hv, 1).first;
        else ++it->second;
        rng ^= it->second;
        uint64_t rv = wy::wyhash64_stateless(&rng); // RNG with both item and count

        FT f = std::log((rv >> 12) * 2.220446049250313e-16);
        ls_.reset();
        ls_.seed(rv);
        uint32_t n = 0;
        for(;f < mvt.max();) {
            uint32_t idx = ls_.step();
            if(sub_update(idx, f, item_index)) {
                if(f >= mvt.max()) break;
            }
            if(++n == m_) break;
            f += std::log((wy::wyhash64_stateless(&rng) >> 12) * 2.220446049250313e-16)
                 * (m_ / (m_ - n));
            // Sample from exponential distribution, then divide by number
        }
    }
public:
    OMHasher(size_t m, size_t l, const FT maxv=std::numeric_limits<FT>::max())
        : m_(m), l_(l), indices(m_ * l_), vals(m_ * l_), mvt(m), ls_(m)
    {   
        reset();
    }

    template<typename T>
    std::vector<uint64_t> hash(const T *ptr, size_t n) {
        reset();
        for(size_t i = 0; i < n; ++i) {
            update(ptr[i], i);
        }
        return finalize(ptr);
    }

    size_t m() const {return m_;}
    size_t l() const {return l_;}

    template<typename T>
    std::vector<uint64_t> finalize(const T *data) {
        std::vector<uint64_t> ret(m_);
        std::vector<T> tmpdata(l_);
        for(size_t i = 0; i < m_; ++i) {
            auto ptr = &indices[l_ * i];
            std::sort(ptr, ptr + l_);
            std::transform(ptr, ptr + l_, tmpdata.data(), [data](auto x) {return data[x];});
            ret[i] = XXH3_64bits(tmpdata.data(), l_ * sizeof(T));
        }
        return ret;
    }

    void reset() {
        std::fill(vals.begin(), vals.end(), std::numeric_limits<FT>::max());
        std::fill(indices.begin(), indices.end(), uint64_t(-1));
        mvt.reset();
        counter.clear();
    }   

};

} // omh

} // sketch
