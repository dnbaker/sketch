#include "python/pysketch.h"
#include "sketch/cqf.h"

PYBIND11_MODULE(sketch_cqf, m) {
    m.doc() = "Counting quotient filter (occupied/runend rank-select filter, Pandey et al. 2017 style)";
    using cqf_t = sketch::cqf::cqf64_64;
    py::class_<cqf_t> (m, "cqf")
        .def(py::init([](uint32_t q, uint32_t r) {
            if(r != 0 && r != 64 - q) throw std::invalid_argument("r must equal 64 - q for the cqf64_64 binding (pass r=0 to accept the default)");
            return new cqf_t(q);
        }), py::arg("q") = 8, py::arg("r") = 0)
        .def("clear", &cqf_t::clear, "Remove all entries.")
        .def("q", &cqf_t::q, "Quotient bits.")
        .def("r", &cqf_t::r, "Remainder bits.")
        .def("size", &cqf_t::size, "Live slots (distinct keys).")
        .def("empty", &cqf_t::empty, "True if no live slots.")
        .def("load_factor", &cqf_t::load_factor, "Load factor.")
        .def("fpr_bound", &cqf_t::fpr_bound, "False-positive bound (0 for exact-mode instantiations).")
        .def("add", [](cqf_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value.")
        .def("addh", [](cqf_t &h, uint64_t v) {h.addh(v);}, "Hash an integer value and add it.")
        .def("addh_np", [](cqf_t &h, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); ++i) h.addh(ptr[i]);
        }, "Hash and add a 1-D numpy uint64 array of values.")
        .def("hash", &cqf_t::hash, "Hash a value with the filter's hash function (count/contains/dec/erase take hashed values).")
        .def("dec", &cqf_t::dec, "Decrement one occurrence; erases the key at zero. Returns False if absent.")
        .def("erase", &cqf_t::erase, "Erase a key entirely. Returns False if absent.")
        .def("contains", &cqf_t::contains, "Membership of a (hashed) value.")
        .def("count", &cqf_t::count, "Count of a (hashed) value.")
        .def("write", [](const cqf_t &h, std::string path) {h.write(path);}, "Serialize to a gzipped file.")
        .def("__repr__", [](const cqf_t &h) {
            return std::string("cqf(q=") + std::to_string(h.q()) + ", r=" + std::to_string(h.r())
                 + ", load=" + std::to_string(h.load_factor()) + ")";
        });
    m.def("from_np", [](const py::array_t<uint64_t> &input, uint32_t q, uint32_t r) {
         if(r != 0 && r != 64 - q) throw std::invalid_argument("r must equal 64 - q for the cqf64_64 binding (pass r=0 to accept the default)");
         cqf_t ret(q);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ++i) ret.addh(ptr[i]);
         return ret;
     }, py::arg("input"), py::arg("q") = 8, py::arg("r") = 0,
        py::return_value_policy::take_ownership, "Creates a CQF from a numpy array of (unhashed) 64-bit integers");
}
