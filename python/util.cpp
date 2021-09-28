#include "pysketch.h"
#include "sketch/isz.h"
#include "xxHash/xxh3.h"

template<typename T>
struct dumbrange {
    T beg, e_;
    dumbrange(T beg, T end): beg(beg), e_(end) {}
    auto begin() const {return beg;}
    auto end()   const {return e_;}
};

template<typename T>
inline dumbrange<T> make_dumbrange(T beg, T end) {return dumbrange<T>(beg, end);}

inline uint64_t xxhash(py::str x, uint64_t seed) {
    Py_ssize_t sz;
    auto cstr = PyUnicode_AsUTF8AndSize(x.ptr(), &sz);
    if(!cstr) throw std::invalid_argument("hash has no c string?");
    return XXH3_64bits_withSeed(static_cast<const void *>(cstr), sz, seed);
}

XXH3_state_t fromseed(uint64_t seed) {
    XXH3_state_t state;
    if(seed)
        XXH3_64bits_reset_withSeed(&state, seed);
    else
        XXH3_64bits_reset(&state);
    return state;
}

inline uint64_t xxhash(py::str x) {
    Py_ssize_t sz;
    auto cstr = PyUnicode_AsUTF8AndSize(x.ptr(), &sz);
    if(!cstr) throw std::invalid_argument("hash has no c string?");
    return XXH3_64bits(static_cast<const void *>(cstr), sz);
}

inline uint64_t xxhash(py::list x, uint64_t seed=0) {
    XXH3_state_t state = fromseed(seed);
    Py_ssize_t sz;
    for(auto obj: x) {
        auto s = py::cast<py::str>(obj);
        auto p = PyUnicode_AsUTF8AndSize(s.ptr(), &sz);
        XXH3_64bits_update(&state, p, sz);
    }
    return XXH3_64bits_digest(&state);
}

py::array_t<uint64_t> xxhash_ngrams(py::list x, Py_ssize_t n, uint64_t seed) {
    auto lx = len(x);
    py::array_t<uint64_t> ret(std::max(Py_ssize_t(lx - n + 1), Py_ssize_t(0)));
    if(!ret.size()) {
        return ret;
    }
    Py_ssize_t sz;
    auto hashrange = [&](auto start, auto end) {
        XXH3_state_t state = fromseed(seed);
        for(auto i = start; i < end; ++i) {
            auto s = py::cast<py::str>(x[i]);
            auto p = PyUnicode_AsUTF8AndSize(s.ptr(), &sz);
            XXH3_64bits_update(&state, p, sz);
        }
        return XXH3_64bits_digest(&state);
    };
    auto rp = (uint64_t *)ret.request().ptr;
    for(size_t i = 0; i < lx - n + 1; ++i) {
        rp[i] = hashrange(i, i + n);
    }
    return ret;
}

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
            ret = sketch::isz::intersection_size(lhrange, rhrange); \
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
#define PERF_FM(TYPE, NAME, OP) do {\
    m.def(NAME, [](py::array_t<TYPE> lhs, Py_ssize_t v) { \
        schism::Schismatic<TYPE> div(v);\
        auto inf = lhs.request();\
        auto ptr = (const TYPE *)inf.ptr;\
        py::array_t<TYPE> ret(inf.size);\
        auto retptr = (TYPE *)ret.request().ptr;\
        std::transform(ptr, ptr + inf.size, retptr, [&](auto x) {return div.OP(x);});\
        return ret;\
    });\
    m.def(NAME "_", [](py::array_t<TYPE> lhs, Py_ssize_t v) { \
        schism::Schismatic<TYPE> div(v);\
        auto inf = lhs.request();\
        auto ptr = (TYPE *)inf.ptr;\
        py::array_t<TYPE> ret(inf.size);\
        auto retptr = (TYPE *)ret.request().ptr;\
        std::transform(ptr, ptr + inf.size, retptr, [&](auto x) {return div.OP(x);});\
        return lhs;\
    }); } while(0);
    PERF_FM(int32_t, "fastdiv", div);
    PERF_FM(int32_t, "fastmod", mod);
    PERF_FM(int64_t, "fastdiv", div);
    PERF_FM(int64_t, "fastmod", mod);
    PERF_FM(uint32_t, "fastdiv", div);
    PERF_FM(uint32_t, "fastmod", mod);
    PERF_FM(uint64_t, "fastdiv", div);
    PERF_FM(uint64_t, "fastmod", mod);
#undef PERF_FM
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
        thread_local wy::WyHash<uint64_t, 1> gen(1337 + std::hash<std::thread::id>()(std::this_thread::get_id()));
        py::array_t<uint64_t> ret(i);
        auto ptr = static_cast<uint64_t *>(ret.request().ptr);
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
         "Compare sketches in parallel. Input: list of sketches.")
    .def("hash", [](py::str x, uint64_t seed) {
        return xxhash(x, seed);
    }, py::arg("x"), py::arg("seed") = 0)
    .def("hash", [](py::list x, uint64_t seed) {
        return xxhash(x, seed);
    }, py::arg("x"), py::arg("seed") = 0)
    .def("hash_ngrams", [](py::list x, int n, Py_ssize_t seed) {
        return xxhash_ngrams(x, n, seed);
    }, py::arg("x"), py::arg("n") = 3, py::arg("seed") = 0)
#define COUNT_EQ_TYPE(TYPE) \
    .def("count_eq", [](py::array_t<TYPE, py::array::c_style> &lhs, py::array_t<TYPE, py::array::c_style> &rhs) {\
        if(lhs.size() != rhs.size()) throw std::invalid_argument("Mismatched sizes");\
        return sketch::eq::count_eq(lhs.data(), rhs.data(), lhs.size());\
    }, py::arg("lhs"), py::arg("rhs"))\
    .def("ccount_eq", [](py::array_t<TYPE, py::array::c_style> &lhs, py::array_t<TYPE, py::array::c_style> &rhs, py::int_ nthreads) {\
        {py::ssize_t nt = nthreads.cast<py::ssize_t>(); OMP_ONLY(if(nt >= 1) omp_set_num_threads(nt);)}\
        py::buffer_info lhi = lhs.request(), rhi = rhs.request(), retinf;\
        py::object retarr = py::none();\
        if(lhi.shape.at(1) != rhi.shape.at(1)) throw std::invalid_argument("Mismatched sizes");\
        if(lhi.ndim != 2 || rhi.ndim != 2) throw std::invalid_argument("Wrong dimensions: require 2-d arrays in both cases.");\
        const auto ndim = lhi.shape.at(1);\
        const std::vector<py::ssize_t> shape{{lhi.shape[0], rhi.shape[0]}};\
        if(ndim < 256)\
            retarr = py::array_t<uint8_t>(shape), retinf = retarr.cast<py::array_t<uint8_t>>().request();\
        else if(ndim < 65536) \
            retarr = py::array_t<uint16_t>(shape), retinf = retarr.cast<py::array_t<uint16_t>>().request();\
        else if(ndim < (1LL << 32))\
            retarr = py::array_t<uint32_t>(shape), retinf = retarr.cast<py::array_t<uint32_t>>().request();\
        else\
            retarr = py::array_t<uint64_t>(shape), retinf = retarr.cast<py::array_t<uint64_t>>().request();\
        OMP_PFOR\
        for(py::ssize_t i = 0; i < lhi.shape[0]; ++i) {\
            for(py::ssize_t j = 0; j < rhi.shape[0]; ++j) {\
                auto neq = sketch::eq::count_eq((TYPE *)lhi.ptr + i * lhi.shape[1], (TYPE *)rhi.ptr + j * rhi.shape[1], lhi.shape[1]);\
                if(retinf.itemsize == 1) {\
                    ((uint8_t *)retinf.ptr)[i * retinf.shape[1] + j] = neq;\
                } else if(retinf.itemsize == 2) {\
                    ((uint16_t *)retinf.ptr)[i * retinf.shape[1] + j] = neq;\
                } else if(retinf.itemsize == 4) {\
                    ((uint32_t *)retinf.ptr)[i * retinf.shape[1] + j] = neq;\
                } else {\
                    ((uint64_t *)retinf.ptr)[i * retinf.shape[1] + j] = neq;\
                }\
            }\
        }\
        return retarr;\
    }, py::arg("lhs"), py::arg("rhs"), py::arg("nthreads") = -1)\
    .def("pcount_eq", [](py::array_t<TYPE, py::array::c_style> &lhs, py::int_ nthreads) {\
        py::ssize_t nt = nthreads.cast<py::ssize_t>();\
        if(nt < 1) nt = 1;\
        py::buffer_info lhi = lhs.request(), retinf;\
        py::object retarr = py::none();\
        if(lhi.ndim != 2) throw std::invalid_argument("Wrong dimensions: require 2-d array.");\
        const auto ndim = lhi.shape.at(1);\
        const auto nelem = lhi.shape.at(0);\
        const auto nret = (nelem * (nelem - 1)) >> 1;\
        auto getind = [nelem](size_t x, size_t y) {return (x * (nelem * 2 - x - 1)) / 2 + y - (x + 1);};\
        const std::vector<py::ssize_t> retshape{{nret}};\
        if(ndim < 256)\
            retarr = py::array_t<uint8_t>(retshape), retinf = retarr.cast<py::array_t<uint8_t>>().request();\
        else if(ndim < 65536) \
            retarr = py::array_t<uint16_t>(retshape), retinf = retarr.cast<py::array_t<uint16_t>>().request();\
        else if(ndim < (1LL << 32))\
            retarr = py::array_t<uint32_t>(retshape), retinf = retarr.cast<py::array_t<uint32_t>>().request();\
        else\
            retarr = py::array_t<uint64_t>(retshape), retinf = retarr.cast<py::array_t<uint64_t>>().request();\
        OMP_PRAGMA("omp parallel for schedule(dynamic) num_threads(nt)")\
        for(py::ssize_t i = 0; i < lhi.shape[0]; ++i) {\
            const auto lhp = (TYPE *)lhi.ptr + i * lhi.shape[1];\
            for(py::ssize_t j = i + 1; j < lhi.shape[0]; ++j) {\
                auto neq = sketch::eq::count_eq(lhp, (TYPE *)lhi.ptr + j * lhi.shape[1], lhi.shape[1]);\
                const auto ind = getind(i, j);\
                if(retinf.itemsize == 1) {\
                    ((uint8_t *)retinf.ptr)[ind] = neq;\
                } else if(retinf.itemsize == 2) {\
                    ((uint16_t *)retinf.ptr)[ind] = neq;\
                } else if(retinf.itemsize == 4) {\
                    ((uint32_t *)retinf.ptr)[ind] = neq;\
                } else {\
                    ((uint64_t *)retinf.ptr)[ind] = neq;\
                }\
            }\
        }\
        return retarr;\
    }, py::arg("lhs"), py::arg("nthreads") = -1)
    COUNT_EQ_TYPE(uint64_t)
    COUNT_EQ_TYPE(uint32_t)
    COUNT_EQ_TYPE(uint16_t)
    COUNT_EQ_TYPE(uint8_t)
    COUNT_EQ_TYPE(double)
    COUNT_EQ_TYPE(float)
    COUNT_EQ_TYPE(int64_t)
    COUNT_EQ_TYPE(int32_t)
    COUNT_EQ_TYPE(int16_t)
    COUNT_EQ_TYPE(int8_t)
#undef COUNT_EQ_TYPE
    ;
} // pybind11 module
