#include "python/pysketch.h"
using sketch::bf_t;

std::string bf2str(const bf_t &h) {
    return std::string("BloomFilter{.p=") + std::to_string(h.p()) + ",.nhashes=" + std::to_string(h.nhashes()) + '}';
}

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
        .def_property_readonly("nhashes", [](const bf_t &h1) {return h1.nhashes();})
        .def_property_readonly("tablesize", [](const bf_t &h1) {return 1ull << h1.p();})
        .def("__ior__", [](bf_t &lh, const bf_t &rh) {
            lh |= rh;
            return lh;
        })
        .def("getcard", &bf_t::cardinality_estimate, "Estimate cardinality of items inserted into Bloom Filter")
        .def("__or__", [](bf_t &lh, const bf_t &rh) {
            return lh | rh;
        }).def("__contains__", [](bf_t &lh, py::object obj) {
            return lh.may_contain(py::hash(obj));
        }).def("__contains__", [](bf_t &lh, uint64_t v) {
            return lh.may_contain(v);
        }).def("__str__", [](const bf_t &h) {return bf2str(h);
        }).def("__repr__", [](const bf_t &h) {
            return bf2str(h) + ':' + std::to_string(reinterpret_cast<uint64_t>(&h));
        }).def("__eq__", [](const sketch::bf_t &h, const sketch::bf_t &h2) {
            return h == h2;
        }).def("__neq__", [](const sketch::bf_t &h, const sketch::bf_t &h2) {
            return h != h2;
        }).def("write", [](const sketch::bf_t &h, std::string path) {
            h.write(path);
        }).def("to_numpy", [](const sketch::bf_t &h) {
            py::array_t<uint64_t> ret(py::ssize_t(h.core().size()));
            std::copy(h.core().data(), h.core().data() + h.core().size(), (uint64_t *)ret.request().ptr);
            return py::make_tuple(ret, py::int_(h.m()), py::int_(h.nhashes()));
        });
    m.def("jaccard_index", [](bf_t &h1, bf_t &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss, int nhashes) {
         bf_t ret(ss, nhashes);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.",
        py::arg("a"), py::arg("ss") = 10, py::arg("nhashes") = 4)
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss, int nhashes) {
         bf_t ret(ss, nhashes);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers", py::arg("a"), py::arg("ss") = 10, py::arg("nhashes") = 4);
} // pybind11 module
