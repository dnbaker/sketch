#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "hll.h"
namespace py = pybind11;
using namespace sketch;
using namespace hll;

PYBIND11_MODULE(_hll, m) {
    m.doc() = "HyperLogLog"; // optional module docstring
    py::class_<hll_t> (m, "_hll")
        .def(py::init<size_t>())
        .def(py::init<std::string>())
        .def("clear", &hll_t::clear, "Clear all entries.")
        .def("resize", &hll_t::resize, "Change old size to a new size.")
        .def("sum", &hll_t::sum, "Add up results.")
        .def("report", &hll_t::report, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", [](hll_t &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh_", [](hll_t &h1, uint64_t v) {h1.addh(h1.hash(v));}, "Hash an integer value and then add that to the sketch..")
        .def("jaccard_index", [](hll_t &h1, hll_t &h2) {return jaccard_index(h2, h2);})
        .def("sprintf", &hll_t::sprintf)
        .def("union", [](const hll_t &h1, const hll_t &h2) {return h1 + h2;});
    m.def("jaccard_index", [](hll_t &h1, hll_t &h2) {
            return jaccard_index(h1, h2);
        }
    ).def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         hll_t ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
    }, "Creates an HLL sketch from a numpy array of 64-bit integers.");
}
