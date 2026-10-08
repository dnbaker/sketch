#include "python/pysketch.h"
#include "sketch/fracminhash.h"

PYBIND11_MODULE(sketch_fmh, m) {
    m.doc() = "FracMinHash support (sourmash-style fractional sampling)";
    py::class_<fmh::fracminhash_t> (m, "fracminhash")
        .def(py::init([](double scale) {return new fmh::fracminhash_t(scale);}), py::arg("scale") = 1000.)
        .def(py::init([](uint64_t max_hash) {return new fmh::fracminhash_t(max_hash);}), py::arg("max_hash"))
        .def(py::init<std::string>())
        .def("clear", &fmh::fracminhash_t::clear, "Remove all kept samples.")
        .def("scale", &fmh::fracminhash_t::scale, "Sampling scale (1/kept fraction).")
        .def("max_hash", &fmh::fracminhash_t::max_hash, "Hash threshold.")
        .def("size", &fmh::fracminhash_t::size, "Number of kept samples.")
        .def("report", &fmh::fracminhash_t::report, "Estimated cardinality.")
        .def("cardinality", &fmh::fracminhash_t::report, "Estimated cardinality.")
        .def("est_err", &fmh::fracminhash_t::est_err, "Standard error of the cardinality estimate.")
        .def("add", [](fmh::fracminhash_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value to the sketch.")
        .def("add", [](fmh::fracminhash_t &h, const py::array_t<uint64_t> &a) {
            h.add(a.data(), size_t(a.size()));
        }, "Add a batch of (hashed) values from a 1-D numpy uint64 array.")
        .def("addh", [](fmh::fracminhash_t &h1, uint64_t v) {h1.addh(v);}, "Hash an integer value and then add that to the sketch.")
        .def("addh", [](fmh::fracminhash_t &h1, py::object p) {h1.addh(py::hash(p));},
             "Hash a python object and add it to the sketch.")
        .def("addh_np", [](fmh::fracminhash_t &h1, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h1.addh(ptr[i++]));
        }, "Hash and add a 1-D numpy uint64 array of values.")
        .def("containment_index", &fmh::fracminhash_t::containment_index,
             "Estimated fraction of the argument's elements contained in this sketch.")
        .def("jaccard_index", &fmh::fracminhash_t::jaccard_index, "Jaccard estimate over kept samples.")
        .def("union_size", &fmh::fracminhash_t::union_size, "Estimate the size of the union of two sketches.")
        .def("union", [](const fmh::fracminhash_t &h1, const fmh::fracminhash_t &h2) {return h1 + h2;})
        .def("__ior__", [](fmh::fracminhash_t &lh, const fmh::fracminhash_t &rh) {lh += rh; return lh;})
        .def("__or__", [](const fmh::fracminhash_t &lh, const fmh::fracminhash_t &rh) {return lh + rh;})
        .def("__eq__", [](const fmh::fracminhash_t &h, const fmh::fracminhash_t &o) {return h == o;})
        .def("__str__", [](const fmh::fracminhash_t &h) {return h.desc_string();})
        .def("__repr__", [](const fmh::fracminhash_t &h) {
            return std::string("fracminhash(scale=") + std::to_string(h.scale()) + ", kept="
                 + std::to_string(h.size()) + ", est=" + std::to_string(h.report()) + ")";
        })
        .def("write", [](const fmh::fracminhash_t &h, std::string path) {h.write(path);}, "Serialize to a gzipped file.")
        .def("to_bytes", [](const fmh::fracminhash_t &h) {
            // magic, version, max_hash, min_kept, count, keys (set order)
            std::string out;
            auto put = [&out](const void *src, size_t n) {
                const size_t off = out.size();
                out.resize(off + n);
                std::memcpy(out.data() + off, src, n);
            };
            const uint32_t magic = 0x464D4831u, ver = 1;
            const uint64_t mh = h.max_hash(), mn = h.min_hash(), n = h.size();
            put(&magic, 4); put(&ver, 4);
            put(&mh, 8); put(&mn, 8); put(&n, 8);
            h.for_each_kept([&](uint64_t v) {put(&v, 8);});
            return py::bytes(out);
        })
        .def_static("from_bytes", [](const py::bytes &b) {
            std::string s(b);
            if(s.size() < 28) throw std::invalid_argument("[fracminhash] truncated serialized sketch");
            const char *p = s.data();
            uint32_t magic, ver;
            std::memcpy(&magic, p, 4); p += 4;
            std::memcpy(&ver, p, 4); p += 4;
            if(magic != 0x464D4831u) throw std::invalid_argument("[fracminhash] bad magic in serialized sketch");
            if(ver != 1) throw std::invalid_argument("[fracminhash] unsupported serialized version");
            uint64_t mh, mn, n;
            std::memcpy(&mh, p, 8); p += 8;
            std::memcpy(&mn, p, 8); p += 8;
            std::memcpy(&n, p, 8); p += 8;
            if(s.size() < 28 + n * 8) throw std::invalid_argument("[fracminhash] truncated serialized sketch");
            fmh::fracminhash_t ret(mh);
            for(uint64_t i = 0; i < n; ++i, p += 8) {
                uint64_t v;
                std::memcpy(&v, p, 8);
                ret.add(v);
            }
            return ret;
        })
        .def("to_numpy", [](const fmh::fracminhash_t &h) {
            py::array_t<uint64_t> ret(py::ssize_t(h.size()));
            uint64_t *ptr = (uint64_t *)ret.request().ptr;
            h.for_each_kept([&](uint64_t v) {*ptr++ = v;});
            return ret;
        }, "Kept hashes as a 1-D numpy uint64 array.");
    m.def("from_np", [](const py::array_t<uint64_t> &input, double scale=1000.) {
         fmh::fracminhash_t ret(scale);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates a FracMinHash sketch from a numpy array of (unhashed) 64-bit integers");
}
