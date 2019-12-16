#include "pysketch.h"


PYBIND11_MODULE(bbmh, m) {
    m.doc() = "BBitMinHash support";
    py::class_<mh::BBitMinHasher<uint64_t>> (m, "BBitMinHasher")
        .def(py::init<size_t, unsigned>())
        .def("clear", &mh::BBitMinHasher<uint64_t>::clear, "Clear all entries.")
        //.def("report", &mh::BBitMinHasher<uint64_t>::report, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", [](mh::BBitMinHasher<uint64_t> &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh", [](mh::BBitMinHasher<uint64_t> &h1, uint64_t v) {h1.addh(WangHash()(v));}, "Hash an integer value and then add that to the sketch..")
        .def("jaccard_index", [](mh::BBitMinHasher<uint64_t> &h1, mh::BBitMinHasher<uint64_t> &h2) {return jaccard_index(h2, h2);});
        //.def("sprintf", &mh::BBitMinHasher<uint64_t>::sprintf)
        //.def("union", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1 + h2;})
        //.def("union_size", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1.union_size(h2);})
    m.def("jaccard_index", [](mh::BBitMinHasher<uint64_t> &h1, mh::BBitMinHasher<uint64_t> &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10, unsigned b=32) {
         mh::BBitMinHasher<uint64_t> ret(ss, b);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.")
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         mh::BBitMinHasher<uint64_t> ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers")
    //.def("union_size", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1.union_size(h2);}, "Calculate union size")
    .def("ij2ind", [](size_t i, size_t j, size_t n) {return i < j ? (((i) * (n * 2 - i - 1)) / 2 + j - (i + 1)): (((j) * (n * 2 - j - 1)) / 2 + i - (j + 1));})
    .def("randset", [](size_t i) {
        static wy::WyHash<uint64_t, 4> gen(1337);
        py::array_t<uint64_t> ret({i});
        auto ptr = static_cast<uint64_t *>(ret.request().ptr);
        for(size_t j = 0; j < i; ptr[j++] = gen());
        return ret;
    }, py::return_value_policy::take_ownership, "Generate a 1d random numpy array")
    .def("tri2full", [](py::array_t<float> arr) {
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
} // pybind11 module
