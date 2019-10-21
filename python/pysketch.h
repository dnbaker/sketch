#ifndef PYSKETCH_H__
#define PYSKETCH_H__
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "sketch/bbmh.h"
#include <omp.h>
#include "aesctr/wy.h"
namespace py = pybind11;
using namespace sketch;
using namespace hll;
using namespace hll;

static size_t nchoose2(size_t n) {return n * (n - 1) / 2;}

static size_t flat2fullsz(size_t n) {
    n <<= 1;
    size_t i;
    for(i = std::sqrt(n);i * (i - 1) < n; ++i);
    if(i * (i - 1) != n) throw std::runtime_error("Failed to extract correct size");
    return i;
}

template<typename Sketch>
struct AsymmetricCmpFunc {
    template<typename Func>
    static py::array_t<float> apply(py::list l, const Func &func) {
        std::vector<Sketch *> ptrs(l.size(), nullptr);
        size_t i = 0;
        for(py::handle ob: l) {
            auto lp = ob.cast<Sketch *>();
            if(!lp) throw std::runtime_error("Failed to cast to Sketch *");
            ptrs[i++] = lp;
        }
        const size_t lsz = l.size();
        py::array_t<float> ret({lsz, lsz});
        float *ptr = static_cast<float *>(ret.request().ptr);
        for(size_t i = 0; i < lsz; ++i) {
            OMP_PRAGMA("omp parallel for")
            for(size_t j = 0; j < lsz; ++j) {
			    ptr[i * lsz + j] = func(*ptrs[i], *ptrs[j]);
			    ptr[j * lsz + i] = func(*ptrs[j], *ptrs[i]);
            }
        }
        return ret;
    }
};

struct CmpFunc {
    template<typename Func>
    static py::array_t<float> apply(py::list l, const Func &func) {
        std::vector<hll::hll_t *> ptrs(l.size(), nullptr);
        size_t i = 0;
        for(py::handle ob: l) {
            auto lp = ob.cast<hll_t *>();
            if(!lp) throw std::runtime_error("Note: I die");
            ptrs[i++] = lp;
        }
        const size_t lsz = l.size(), nc2 = nchoose2(lsz);
        py::array_t<float> ret({nc2});
        float *ptr = static_cast<float *>(ret.request().ptr);
        for(size_t i = 0; i < lsz; ++i) {
            OMP_PRAGMA("omp parallel for")
            for(size_t j = i + 1; j < lsz; ++j) {
                size_t access_index = ((i * (lsz * 2 - i - 1)) / 2 + j - (i + 1));
			    ptr[access_index] = func(*ptrs[i], *ptrs[j]);
            }
        }
        return ret;
    }
};

struct JIF {
    template<typename T>
    auto operator()(T &x, T &y) const {
        return x.jaccard_index(y);
    }
};
struct USF {
    template<typename T>
    auto operator()(T &x, T &y) const {
        return x.union_size(y);
    }
};
struct ISF {
    template<typename T>
    auto operator()(T &x, T &y) const {
        return intersection_size(x, y);
    }
};
struct SCF {
    template<typename T>
    auto operator()(T &x, T &y) const {
        return intersection_size(x, y) / std::min(x.report(), y.report());
    }
};
struct CSF {
    template<typename T>
    auto operator()(T &x, T &y) const {
        return x.containment_index(y);
    }
};

#endif
