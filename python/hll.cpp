#include "python/pysketch.h"

PYBIND11_MODULE(sketch_hll, m) {
    m.doc() = "HyperLogLog support"; // optional module docstring
    py::class_<hll_t> (m, "hll")
        .def(py::init<size_t>())
        .def(py::init<std::string>())
        .def("clear", &hll_t::clear, "Clear all entries.")
        .def("resize", &hll_t::resize, "Change old size to a new size.")
        .def("sum", &hll_t::sum, "Add up results.")
        .def("csum", &hll_t::csum, "Only add up results if no cached value is present.")
        .def("report", &hll_t::report, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", [](hll_t &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh_", [](hll_t &h1, uint64_t v) {h1.addh(h1.hash(v));}, "Hash an integer value and then add that to the sketch..")
        .def("addh", [](hll_t &h1, py::object p) {
            auto hv = py::hash(p);
            h1.addh(hv);
        }, "Hash a python object and add it to the sketch.")
        .def("jaccard_index", [](hll_t &h1, hll_t &h2) {return jaccard_index(h1, h2);})
        .def("sprintf", &hll_t::sprintf)
        .def("union", [](const hll_t &h1, const hll_t &h2) {return h1 + h2;})
        .def("union_size", [](const hll_t &h1, const hll_t &h2) {return h1.union_size(h2);})
        .def("est_err", &hll_t::est_err, "Estimate error")
        .def("relative_error", &hll_t::relative_error, "Expected error for sketch")
        .def("compress", [](const hll_t &h1, unsigned newnp) {return h1.compress(newnp);},
             py::return_value_policy::take_ownership,
             "Compress an HLL sketch from a previous prefix length to a smaller one.")
        .def("__str__", [](const hll_t &h) {return h.desc_string();})
        .def("__repr__", [](const hll_t &h) {return h.desc_string() + ':' + std::to_string(reinterpret_cast<uint64_t>(&h));})
        .def("__ior__", [](hll_t &lh, const hll_t &rh) {lh += rh; return lh;})
        .def("__or__", [](const hll_t &lh, const hll_t &rh) {return lh + rh;})
        .def("__eq__", [](const sketch::hll_t &h, const sketch::hll_t &h2) {
            return h == h2;
        }).def("__neq__", [](const sketch::hll_t &h, const sketch::hll_t &h2) {
            return h != h2;
        }).def("write", [](const sketch::hll_t &h, std::string path) {
            h.write(path);
        }).def("to_numpy", [](const sketch::hll_t &h) {
            py::array_t<uint8_t> ret(py::ssize_t(h.size()));
            std::copy(h.core().data(), h.core().data() + h.size(), (uint8_t *)ret.request().ptr);
            return ret;
        });
    m.def("jaccard_index", [](hll_t &h1, hll_t &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         hll_t ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         ret.sum();
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.")
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         hll_t ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         ret.sum();
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers")
    .def("union_size", [](const hll_t &h1, const hll_t &h2) {return h1.union_size(h2);}, "Calculate union size");
} // pybind11 module
