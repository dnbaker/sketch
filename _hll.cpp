#include "pybind11/pybind11.h"
#ifndef HLL_HEADER_ONLY
#define HLL_HEADER_ONLY
#endif
#include "hll.h"
namespace py = pybind11;
using namespace hll;

PYBIND11_MODULE(_hll, m) {
    m.doc() = "pybind11-powered HyperLogLog"; // optional module docstring
    py::class_<hll_t> (m, "hll")
        .def(py::init<size_t>())
        .def("clear", &hll_t::clear, "Clear all entries.")
        .def("resize", &hll_t::resize, "Change old size to a new size.")
        .def("sum", &hll_t::sum, "Add up results.")
        .def("report", &hll_t::report, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", &hll_t::add, "Add a (hashed) value to the sketch.")
        .def("addh_", &hll_t::addh, "Hash an integer value and then add that to the sketch.");
}
