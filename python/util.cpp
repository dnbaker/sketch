#include "pysketch.h"
#include "sketch/isz.h"

template<typename T>
struct dumbrange {
    T beg, e_;
    dumbrange(T beg, T end): beg(beg), e_(end) {}
    auto begin() const {return beg;}
    auto end()   const {return e_;}
};

template<typename T>
inline dumbrange<T> make_dumbrange(T beg, T end) {return dumbrange<T>(beg, end);}

PYBIND11_MODULE(sketch_util, m) {
    m.doc() = "General utilities: shs_isz, which performs fast set intersections\n"
              "fast{div/mod}, which performs fast mod and division operations\n"
              "tri2full, which takes packed distance matrices and unpacks them\n"
              "ij2ind, which computes the index into a packed matrix from unpacked coordinates\n"
              "randset, which generates a random set of 64-bit integers\n";
    m.def("shs_isz", [](py::array lhs, py::array rhs) {
        py::buffer_info lhinfo = lhs.request(), rhinfo = rhs.request();
        if(lhinfo.format != rhinfo.format) throw std::runtime_error("dtypes for lhs and rhs array are not the same");
        if(lhinfo.ndim != rhinfo.ndim || lhinfo.ndim != 1) throw std::runtime_error("Wrong number of dimensions");
        
        size_t ret;
#define PERF_ISZ__(type) \
        if(py::isinstance<py::array_t<type>>(lhs)) { \
            auto lhrange = make_dumbrange((const type *)lhinfo.ptr, (const type *)lhinfo.ptr + lhinfo.size);\
            auto rhrange = make_dumbrange((const type *)rhinfo.ptr, (const type *)rhinfo.ptr + rhinfo.size);\
            ret = sketch::common::intersection_size(lhrange, rhrange); \
            goto end; \
        }
        PERF_ISZ__(uint64_t)
        PERF_ISZ__(int64_t)
        PERF_ISZ__(uint32_t)
        PERF_ISZ__(int32_t)
        PERF_ISZ__(double)
        PERF_ISZ__(float)
#undef PERF_ISZ__
        else throw std::runtime_error("Unexpected type for shs_isz");
        end:
        return ret;
    }, "shs_isz: computes the intersection size of two sorted hash set lists as numpy arrays. These must be 1-dimensional contiguous and of the same dtype");
    m.def("fastmod", [](py::array lhs, int64_t v) {
        auto inf = lhs.request();
        std::fprintf(stderr, "found format: %s\n", inf.format.data());

#define PERF_DUMB_RANGE(type, func) \
        if(py::isinstance<py::array_t<type>>(lhs)) { \
            schism::Schismatic<type> div(v); \
            for(auto &i: make_dumbrange((type *)inf.ptr, (type *)inf.ptr + inf.size)) i = div.func(i);\
            return; \
        }
        PERF_DUMB_RANGE(uint64_t, mod)
        PERF_DUMB_RANGE(int64_t, mod)
        PERF_DUMB_RANGE(uint32_t, mod)
        PERF_DUMB_RANGE(int32_t, mod)
        throw std::runtime_error("Invalid type for fastmod");
    });
    m.def("fastdiv", [](py::array lhs, int64_t v) {
        auto inf = lhs.request();
        PERF_DUMB_RANGE(uint64_t, div)
        PERF_DUMB_RANGE(int64_t, div)
        PERF_DUMB_RANGE(uint32_t, div)
        PERF_DUMB_RANGE(int32_t, div)
        throw std::runtime_error("Invalid type for fastmod");
#undef PERF_DUMB_RANGE
    });
    m.def("tri2full", [](py::array_t<float> arr) {
        size_t dim = flat2fullsz(arr.size());
        py::array_t<float> ret({dim, dim});
        auto retptr = static_cast<float *>(ret.request().ptr), aptr = static_cast<float *>(arr.request().ptr);
        for(size_t i = 0; i < dim; ++i) {
            for(size_t j = 0; j < i; ++j) {
                size_t ind = (((j) * (dim * 2 - j - 1)) / 2 + i - (j + 1));
                retptr[dim * i + j] = aptr[ind];
            }
            retptr[dim * i + i] = 1.; // Jaccard index is 1 for this case.
            for(size_t j = i + 1; j < dim; ++j) {
                size_t ind = ((i * (dim * 2 - i - 1)) / 2 + j - (i + 1));
                retptr[dim * i + j] = aptr[ind];
            }
        }
        return ret;
    }, py::return_value_policy::take_ownership);
    m.def("ij2ind", [](size_t i, size_t j, size_t n) {return i < j ? (((i) * (n * 2 - i - 1)) / 2 + j - (i + 1)): (((j) * (n * 2 - j - 1)) / 2 + i - (j + 1));});
    m.def("randset", [](size_t i) {
        thread_local wy::WyHash<uint64_t, 4> gen(1337 + std::hash<std::thread::id>()(std::this_thread::get_id()));
        py::array_t<uint64_t> ret({i});
        auto ptr = static_cast<uint64_t *>(ret.request().ptr);
        OMP_PFOR
        for(size_t j = 0; j < i; ++j) {
            ptr[j] = gen();
        }
        return ret;
    }, py::return_value_policy::take_ownership, "Generate a 1d random numpy array")
    .def("jaccard_matrix", [](py::list l) {return CmpFunc::apply(l, JIF());}, py::return_value_policy::take_ownership,
         "Compare sketches in parallel. Input: list of sketches. Output: numpy array of n-choose-2 flat matrix of JIs, float")
    .def("intersection_matrix", [](py::list l) {return CmpFunc::apply(l, ISF());}, "Compare sketches in parallel. Input: list of sketches. Output: n-choose-2 flat matrix of intersection_size, float")
    .def("containment_matrix", [](py::list l) {return AsymmetricCmpFunc::apply(l, CSF());}, py::return_value_policy::take_ownership, "Compare sketches in parallel. Input: list of sketches. Output: n-choose-2 flat matrix of intersection_size, float")
    .def("union_size_matrix", [](py::list l) {return CmpFunc::apply(l, USF());},
         "Compare sketches in parallel. Input: list of sketches.")
    .def("symmetric_containment_matrix", [](py::list l) {return CmpFunc::apply(l, SCF());},
         "Compare sketches in parallel. Input: list of sketches.");
} // pybind11 module
