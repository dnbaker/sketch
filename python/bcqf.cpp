#include "sketch/cqf.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
PYBIND11_MODULE(sketch_bcqf, m) {
    m.doc() = "Block-confined counting quotient filter (round 16n): displacement-free exact counting filter";
    using bcqf_t = sketch::cqf::bcqf48_16;
    py::class_<bcqf_t> (m, "bcqf")
        .def(py::init([](uint32_t q, double max_load) {
            if(max_load == 1.0) return new bcqf_t(q);
            return new bcqf_t(q, max_load);
        }), py::arg("q"), py::arg("max_load") = 1.0, "New filter with 2^q slots; max_load <= 0.9 grows proactively.")
        .def("clear", &bcqf_t::clear, "Remove all entries.")
        .def("q", &bcqf_t::q, "log2 of slot count.")
        .def("size", &bcqf_t::size, "Distinct keys.")
        .def("empty", &bcqf_t::empty, "True if no live slots.")
        .def("load_factor", &bcqf_t::load_factor, "Live keys / slots.")
        .def("max_load", &bcqf_t::max_load, "Proactive grow threshold.")
        .def("count", &bcqf_t::count, "Count of a (hashed, 48-bit reduced) value.")
        .def("contains", &bcqf_t::contains, "Membership of a (hashed) value.")
        .def("add", [](bcqf_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value.")
        .def("addh", [](bcqf_t &h, uint64_t v) {h.addh(v);}, "Hash an integer key and add it.")
        .def("addh_np", [](bcqf_t &h, const py::array_t<uint64_t> &a) {
            auto r = a.unchecked<1>();
            for(py::ssize_t i = 0; i < r.shape(0); ++i) h.addh(r(i));
        }, "Vectorized: hash and add every element of a uint64 array.")
        .def("dec", &bcqf_t::dec, "Decrement one occurrence; erases at zero.")
        .def("erase", &bcqf_t::erase, "Erase a key entirely.")
        .def("write", [](const bcqf_t &h, std::string path) {h.write(path);}, "Serialize (CQF6).")
        .def_static("from_path", [](std::string path) { return new bcqf_t(bcqf_t::from_path, path); }, "Deserialize from a CQF6/CQF5 file.")
        .def("__repr__", [](const bcqf_t &h) {
            return std::string("bcqf(q=") + std::to_string(h.q()) + ", size=" + std::to_string(h.size())
                 + ", load=" + std::to_string(h.load_factor()) + ")";
        });
}
