#include "python/pysketch.h"
using sketch::bf_t;

PYBIND11_MODULE(sketch_bf, m) {
    m.doc() = "Bloom Filter support"; // optional module docstring
    py::class_<bf_t> (m, "bf")
        .def(py::init<size_t, unsigned, uint64_t>())
        .def(py::init<size_t, unsigned>())
        .def(py::init<std::string>())
        .def("clear", &bf_t::clear, "Clear all entries.")
        .def("resize", &bf_t::resize, "Change old size to a new size.")
        .def("add", [](bf_t &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh", [](bf_t &h1, py::object p) {
            auto hv = py::hash(p);
            h1.add(hv);
        }, "Hash a python object and add it to the sketch.")
        .def("jaccard_index", [](bf_t &h1, bf_t &h2) {return jaccard_index(h1, h2);})
        .def("union", [](const bf_t &h1, const bf_t &h2) {return h1 + h2;})
        .def("union_size", [](const bf_t &h1, const bf_t &h2) {return h1.union_size(h2);})
        .def("__ior__", [](bf_t &lh, const bf_t &rh) {
            lh |= rh;
            return lh;
        })
        .def("getcard", &bf_t::cardinality_estimate, "Estimate cardinality of items insrted into Bloom Filter")
        .def("__or__", [](bf_t &lh, const bf_t &rh) {
            return lh | rh;
        }).def("__contains__", [](bf_t &lh, py::object obj) {
            return lh.may_contain(py::hash(obj));
        }).def("__contains__", [](bf_t &lh, uint64_t v) {
            return lh.may_contain(v);
        });
    m.def("jaccard_index", [](bf_t &h1, bf_t &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         bf_t ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.")
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         bf_t ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers");
} // pybind11 module
