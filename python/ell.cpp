#include "python/pysketch.h"
#include "sketch/exaloglog.h"

PYBIND11_MODULE(sketch_ell, m) {
    m.doc() = "ExaLogLog support";
    py::class_<ell::exaloglog_t> (m, "exaloglog")
        .def(py::init<uint32_t>(), py::arg("p") = 10)
        .def(py::init([](uint8_t t, uint8_t d, uint32_t p) {return new ell::exaloglog_t(t, d, p);}),
             py::arg("t") = 2, py::arg("d") = 20, py::arg("p") = 10)
        .def(py::init<std::string>())
        .def("clear", &ell::exaloglog_t::clear, "Clear all entries.")
        .def("t", &ell::exaloglog_t::t, "t parameter.")
        .def("d", &ell::exaloglog_t::d, "d parameter.")
        .def("p", &ell::exaloglog_t::p, "Precision parameter.")
        .def("size", &ell::exaloglog_t::m, "Number of registers.")
        .def("report", &ell::exaloglog_t::report, "Estimated cardinality (bias-reduced ML).")
        .def("add", [](ell::exaloglog_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value to the sketch.")
        .def("add", [](ell::exaloglog_t &h, const py::array_t<uint64_t> &a) {h.add(a.data(), size_t(a.size()));},
             "Add a batch of (hashed) values from a 1-D numpy uint64 array.")
        .def("addh_np", [](ell::exaloglog_t &h1, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h1.addh(ptr[i++]));
        }, "Hash and add a 1-D numpy uint64 array of values.")
        .def("addh", [](ell::exaloglog_t &h1, uint64_t v) {h1.addh(v);}, "Hash an integer value and then add that to the sketch.")
        .def("addh", [](ell::exaloglog_t &h1, py::object p) {
            h1.addh(py::hash(p));
        }, "Hash a python object and add it to the sketch.")
        .def("union", [](const ell::exaloglog_t &h1, const ell::exaloglog_t &h2) {return h1 + h2;})
        .def("union_size", [](const ell::exaloglog_t &h1, const ell::exaloglog_t &h2) {
            auto tmp = h1 + h2;
            return tmp.report();
        }, "Estimate the size of the union of two sketches.")
        .def("downsize", [](const ell::exaloglog_t &h, uint8_t d, uint32_t p) {return h.downsize(d, p);},
             py::return_value_policy::take_ownership,
             "Return a copy of this sketch reduced to the given d and precision.")
        .def("merge", [](ell::exaloglog_t &h, const ell::exaloglog_t &o) {h.add(o);}, "In-place merge of a sketch with same t, >= d and >= p.")
        .def("relative_error", &ell::exaloglog_t::relative_error, "Theoretical relative standard error.")
        .def("__ior__", [](ell::exaloglog_t &lh, const ell::exaloglog_t &rh) {lh += rh; return lh;})
        .def("__or__", [](const ell::exaloglog_t &lh, const ell::exaloglog_t &rh) {return lh + rh;})
        .def("__eq__", [](const ell::exaloglog_t &h, const ell::exaloglog_t &o) {return h == o;})
        .def("__str__", [](const ell::exaloglog_t &h) {return h.desc_string();})
        .def("__repr__", [](const ell::exaloglog_t &h) {
            return std::string("exaloglog(t=") + std::to_string(h.t()) + ", d=" + std::to_string(h.d())
                 + ", p=" + std::to_string(h.p()) + ", est=" + std::to_string(h.report()) + ")";
        })
        .def("to_bytes", [](const ell::exaloglog_t &h) {
            std::string out;
            auto put = [&out](const void *src, size_t n) {
                const size_t off = out.size();
                out.resize(off + n);
                std::memcpy(out.data() + off, src, n);
            };
            const uint32_t magic = 0x454C4C31u, ver = 1, t = h.t(), d = h.d(), np = h.p();
            put(&magic, 4); put(&ver, 4); put(&t, 4); put(&d, 4); put(&np, 4);
            const uint32_t pad = 0;
            put(&pad, 4);
            put(h.data(), h.packed_size());
            return py::bytes(out);
        })
        .def_static("from_bytes", [](const py::bytes &b) {
            std::string s(b);
            if(s.size() < 24) throw std::invalid_argument("[exaloglog] truncated serialized sketch");
            const char *p = s.data();
            uint32_t magic, ver, t, d, np, pad;
            std::memcpy(&magic, p, 4); p += 4;
            std::memcpy(&ver, p, 4); p += 4;
            std::memcpy(&t, p, 4); p += 4;
            std::memcpy(&d, p, 4); p += 4;
            std::memcpy(&np, p, 4); p += 4;
            std::memcpy(&pad, p, 4); p += 4;
            (void)pad;
            if(magic != 0x454C4C31u || ver != 1) throw std::invalid_argument("[exaloglog] bad magic/version");
            ell::exaloglog_t ret(uint8_t(t), uint8_t(d), np);
            if(s.size() < 24 + ret.packed_size()) throw std::invalid_argument("[exaloglog] truncated serialized sketch");
            std::memcpy(ret.mutable_data(), p, ret.packed_size());
            return ret;
        })
        .def("write", [](const ell::exaloglog_t &h, std::string path) {h.write(path);})
        .def("martingale_report", [](ell::exaloglog_t &h, const py::array_t<uint64_t> &input) {
            mle::detail::MartingaleEstimator mart;
            auto ptr = input.data();
            for(ssize_t i = 0; i < input.size(); ++i) h.addh(ptr[i], mart);
            return mart.estimate();
        }, "Consume an array of integers with martingale (HIP) estimation and return the estimate.");
    m.def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         ell::exaloglog_t ret{uint32_t(ss)};
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an ExaLogLog sketch from a numpy array of (unhashed) 64-bit integers");
}
