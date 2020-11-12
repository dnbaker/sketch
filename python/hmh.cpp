#include "python/pysketch.h"

PYBIND11_MODULE(sketch_hmh, m) {
    m.doc() = "HyperMinHash support";
    py::class_<sketch::HyperMinHash> (m, "hmh")
        .def(py::init<unsigned, unsigned>(), py::arg("p"), py::arg("rsize") = 8)
        .def(py::init<std::string>(), py::arg("path"))
        .def("getcard", &sketch::HyperMinHash::getcard, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", [](sketch::HyperMinHash &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh", [](sketch::HyperMinHash &h1, py::object p) {
            h1.addh(py::hash(p));
        }
             , "Hash a python object and add it to the sketch.")
        .def("jaccard_index", [](sketch::HyperMinHash &h1, sketch::HyperMinHash &h2) {return jaccard_index(h1, h2);})
        .def("union", [](const sketch::HyperMinHash &h1, const sketch::HyperMinHash &h2) {return h1 + h2;})
        .def("union_size", [](const sketch::HyperMinHash &h1, const sketch::HyperMinHash &h2) {return h1.union_size(h2);})
        .def("unset_card", &sketch::HyperMinHash::unset_card, "Reset cardinality (for the case of continued updates.")
        .def("__str__", [](const sketch::HyperMinHash &h) {
            char buf[256];
            int l = std::sprintf(buf, "HyperMinHash{.p = %d, .r = %d, .register_size = %d}", 64 - h.max_lremainder(), h.regsize() - 6, h.regsize());
            return std::string(buf, l);
        }).def("__repr__", [](const sketch::HyperMinHash &h) {
            char buf[256];
            int l = std::sprintf(buf, "HyperMinHash{.p = %d, .r = %d, .register_size = %d, .cardest=%0.12g, .address=%p}", 64 - h.max_lremainder(), h.regsize() - 6, h.regsize(), h.getcard(), static_cast<const void *>(&h));
            return std::string(buf, l);
        }).def("__eq__", [](const sketch::HyperMinHash &h, const sketch::HyperMinHash &h2) {
            return h == h2;
        }).def("__neq__", [](const sketch::HyperMinHash &h, const sketch::HyperMinHash &h2) {
            return h != h2;
        }).def("write", [](const sketch::HyperMinHash &h, std::string path) {
            h.write(path);
        });

    m.def("jaccard_index", [](sketch::HyperMinHash &h1, sketch::HyperMinHash &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10, unsigned remsize=16) {
         sketch::HyperMinHash ret(ss, remsize);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.")
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10, unsigned remsize=16) {
         sketch::HyperMinHash ret(ss, remsize);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers")
    .def("union_size", [](const sketch::HyperMinHash &h1, const sketch::HyperMinHash &h2) {return h1.union_size(h2);}, "Calculate union size");
} // pybind11 module
