#include "python/pysketch.h"
#include "sketch/ultraloglog.h"
#include "sketch/exaloglog.h"

PYBIND11_MODULE(sketch_ull, m) {
    m.doc() = "UltraLogLog support"; // optional module docstring
    py::class_<ull::ultraloglog_t> (m, "ultraloglog")
        .def(py::init<uint32_t>())
        .def(py::init<std::string>())
        .def("clear", &ull::ultraloglog_t::clear, "Clear all entries.")
        .def("p", &ull::ultraloglog_t::p, "Precision parameter.")
        .def("size", &ull::ultraloglog_t::size, "Number of registers.")
        .def("report", &ull::ultraloglog_t::report, "Estimated cardinality.")
        .def("mle_estimate", &ull::ultraloglog_t::mle_estimate, "Maximum-likelihood estimate (lower variance, slower).")
        .def("fgra_estimate", &ull::ultraloglog_t::fgra_estimate, "Optimal-FGRA estimate (faster, default).")
        .def("set_estim", [](ull::ultraloglog_t &h, bool use_mle) {h.set_estim(use_mle ? ull::ULL_MLE: ull::ULL_FGRA);})
        .def("add", [](ull::ultraloglog_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value to the sketch.")
        .def("add", [](ull::ultraloglog_t &h, const py::array_t<uint64_t> &a) {h.add(a.data(), size_t(a.size()));},
             "Add a batch of (hashed) values from a 1-D numpy uint64 array.")
        .def("addh_np", [](ull::ultraloglog_t &h1, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h1.addh(ptr[i++]));
        }, "Hash and add a 1-D numpy uint64 array of values.")
        .def("addh", [](ull::ultraloglog_t &h1, uint64_t v) {h1.addh(v);}, "Hash an integer value and then add that to the sketch.")
        .def("addh", [](ull::ultraloglog_t &h1, py::object p) {
            h1.addh(py::hash(p));
        }, "Hash a python object and add it to the sketch.")
        .def("union", [](const ull::ultraloglog_t &h1, const ull::ultraloglog_t &h2) {return h1 + h2;})
        .def("union_size", [](const ull::ultraloglog_t &h1, const ull::ultraloglog_t &h2) {
            auto tmp = h1 + h2;
            return tmp.report();
        }, "Estimate the size of the union of two sketches.")
        .def("downsize", [](const ull::ultraloglog_t &h, uint32_t newp) {return h.downsize(newp);},
             py::return_value_policy::take_ownership,
             "Return a copy of this sketch reduced to the given precision.")
        .def("merge", [](ull::ultraloglog_t &h, const ull::ultraloglog_t &o) {h.add(o);}, "In-place merge of a higher- or equal-precision sketch.")
        .def("relative_error", &ull::ultraloglog_t::relative_error, "Theoretical relative standard error.")
        .def("__ior__", [](ull::ultraloglog_t &lh, const ull::ultraloglog_t &rh) {lh += rh; return lh;})
        .def("__or__", [](const ull::ultraloglog_t &lh, const ull::ultraloglog_t &rh) {return lh + rh;})
        .def("__eq__", [](const ull::ultraloglog_t &h, const ull::ultraloglog_t &o) {return h == o;})
        .def("__str__", [](const ull::ultraloglog_t &h) {return h.desc_string();})
        .def("__repr__", [](const ull::ultraloglog_t &h) {
            return std::string("ultraloglog(p=") + std::to_string(h.p()) + ", est=" + std::to_string(h.report()) + ")";
        })
        .def("to_bytes", [](const ull::ultraloglog_t &h) {
            std::string out;
            auto put = [&out](const void *src, size_t n) {
                const size_t off = out.size();
                out.resize(off + n);
                std::memcpy(out.data() + off, src, n);
            };
            const uint32_t magic = 0x554C4C31u, ver = 1, np = h.p(), estim = h.get_estim();
            put(&magic, 4); put(&ver, 4); put(&np, 4); put(&estim, 4);
            put(h.core().data(), h.size());
            return py::bytes(out);
        })
        .def_static("from_bytes", [](const py::bytes &b) {
            std::string s(b);
            if(s.size() < 16) throw std::invalid_argument("[ultraloglog] truncated serialized sketch");
            const char *p = s.data();
            uint32_t magic, ver, np, estim;
            std::memcpy(&magic, p, 4); p += 4;
            std::memcpy(&ver, p, 4); p += 4;
            std::memcpy(&np, p, 4); p += 4;
            std::memcpy(&estim, p, 4); p += 4;
            if(magic != 0x554C4C31u || ver != 1) throw std::invalid_argument("[ultraloglog] bad magic/version");
            if(s.size() < 16 + (size_t(1) << np)) throw std::invalid_argument("[ultraloglog] truncated serialized sketch");
            ull::ultraloglog_t ret(np);
            ret.set_estim(ull::UllEstimationMethod(estim));
            std::memcpy(ret.mutable_core().data(), p, size_t(1) << np);
            return ret;
        })
        .def("write", [](const ull::ultraloglog_t &h, std::string path) {h.write(path);})
        .def("to_numpy", [](const ull::ultraloglog_t &h) {
            py::array_t<uint8_t> ret(py::ssize_t(h.size()));
            std::copy(h.core().data(), h.core().data() + h.size(), (uint8_t *)ret.request().ptr);
            return ret;
        })
        .def("martingale_report", [](ull::ultraloglog_t &h, const py::array_t<uint64_t> &input) {
            // Feeds the array through the sketch while running the martingale
            // estimator; returns the martingale estimate. The sketch is modified.
            mle::detail::MartingaleEstimator mart;
            auto ptr = input.data();
            for(ssize_t i = 0; i < input.size(); ++i) h.addh(ptr[i], mart);
            return mart.estimate();
        }, "Consume an array of integers with martingale (HIP) estimation and return the estimate.");
    m.def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         ull::ultraloglog_t ret{uint32_t(ss)};
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an UltraLogLog sketch from a numpy array of (unhashed) 64-bit integers");
}
