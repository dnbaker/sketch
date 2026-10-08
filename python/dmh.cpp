#include "python/pysketch.h"
#include "sketch/dartminhash.h"

PYBIND11_MODULE(sketch_dmh, m) {
    m.doc() = "DartMinHash support (dart-throwing minhash)";
    py::class_<dmh::dartminhash_t> (m, "dartminhash")
        .def(py::init([](uint32_t k, uint64_t seed) {return new dmh::dartminhash_t(k, seed);}),
             py::arg("k") = 256, py::arg("seed") = 137)
        .def(py::init<std::string>())
        .def("clear", &dmh::dartminhash_t::clear, "Clear all registers.")
        .def("size", &dmh::dartminhash_t::size, "Number of registers (buckets).")
        .def("filled", &dmh::dartminhash_t::filled, "Number of filled registers.")
        .def("report", &dmh::dartminhash_t::report, "Estimated cardinality.")
        .def("cardinality", &dmh::dartminhash_t::report, "Estimated cardinality.")
        .def("add", [](dmh::dartminhash_t &h, uint64_t v) {h.add(v);}, "Add a (hashed) value to the sketch.")
        .def("add", [](dmh::dartminhash_t &h, const py::array_t<uint64_t> &a) {
            h.add(a.data(), size_t(a.size()));
        }, "Add a batch of (hashed) values from a 1-D numpy uint64 array.")
        .def("addh", [](dmh::dartminhash_t &h1, uint64_t v) {h1.addh(v);}, "Hash an integer value and then add that to the sketch.")
        .def("addh", [](dmh::dartminhash_t &h1, py::object p) {h1.addh(py::hash(p));},
             "Hash a python object and add it to the sketch.")
        .def("addh_np", [](dmh::dartminhash_t &h1, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h1.addh(ptr[i++]));
        }, "Hash and add a 1-D numpy uint64 array of values.")
        .def("jaccard_index", &dmh::dartminhash_t::jaccard_index,
             "Unbiased Jaccard estimate; k*J^ ~ Binomial(k, J).")
        .def("containment_index", &dmh::dartminhash_t::containment_index,
             "Estimated fraction of the argument's elements contained in this sketch.")
        .def("union_size", &dmh::dartminhash_t::union_size, "Estimate the size of the union of two sketches.")
        .def("union", [](const dmh::dartminhash_t &h1, const dmh::dartminhash_t &h2) {return h1 + h2;})
        .def("__ior__", [](dmh::dartminhash_t &lh, const dmh::dartminhash_t &rh) {lh += rh; return lh;})
        .def("__or__", [](const dmh::dartminhash_t &lh, const dmh::dartminhash_t &rh) {return lh + rh;})
        .def("__eq__", [](const dmh::dartminhash_t &h, const dmh::dartminhash_t &o) {return h == o;})
        .def("__str__", [](const dmh::dartminhash_t &h) {return h.desc_string();})
        .def("__repr__", [](const dmh::dartminhash_t &h) {
            return std::string("dartminhash(k=") + std::to_string(h.size()) + ", filled="
                 + std::to_string(h.filled()) + ", est=" + std::to_string(h.report()) + ")";
        })
        .def("write", [](const dmh::dartminhash_t &h, std::string path) {h.write(path);}, "Serialize to a gzipped file.")
        .def("to_bytes", [](const dmh::dartminhash_t &h) {
            // magic, version, k, seed, ranks, ids
            std::string out;
            auto put = [&out](const void *src, size_t n) {
                const size_t off = out.size();
                out.resize(off + n);
                std::memcpy(out.data() + off, src, n);
            };
            const uint32_t magic = 0x444D4831u, ver = 1, k = h.size();
            const uint64_t seed = h.seed();
            put(&magic, 4); put(&ver, 4); put(&k, 4); put(&seed, 8);
            put(h.ranks().data(), k * 8);
            put(h.ids().data(), k * 8);
            return py::bytes(out);
        })
        .def_static("from_bytes", [](const py::bytes &b) {
            std::string s(b);
            if(s.size() < 20) throw std::invalid_argument("[dartminhash] truncated serialized sketch");
            const char *p = s.data();
            uint32_t magic, ver, k;
            uint64_t seed;
            std::memcpy(&magic, p, 4); p += 4;
            std::memcpy(&ver, p, 4); p += 4;
            std::memcpy(&k, p, 4); p += 4;
            std::memcpy(&seed, p, 8); p += 8;
            if(magic != 0x444D4831u) throw std::invalid_argument("[dartminhash] bad magic in serialized sketch");
            if(ver != 1) throw std::invalid_argument("[dartminhash] unsupported serialized version");
            if(s.size() < 20 + size_t(k) * 16) throw std::invalid_argument("[dartminhash] truncated serialized sketch");
            dmh::dartminhash_t ret(k, seed);
            std::memcpy(ret.mutable_ranks().data(), p, k * 8); p += k * 8;
            std::memcpy(ret.mutable_ids().data(), p, k * 8);
            return ret;
        })
        .def("to_numpy", [](const dmh::dartminhash_t &h) {
            py::array_t<uint64_t> ranks(py::ssize_t(h.size())), ids(py::ssize_t(h.size()));
            std::copy(h.ranks().begin(), h.ranks().begin() + h.size(), (uint64_t *)ranks.request().ptr);
            std::copy(h.ids().begin(), h.ids().begin() + h.size(), (uint64_t *)ids.request().ptr);
            return py::make_tuple(ranks, ids);
        }, "Register (ranks, ids) as a pair of 1-D numpy uint64 arrays.");
    m.def("from_np", [](const py::array_t<uint64_t> &input, uint32_t k=256, uint64_t seed=137) {
         dmh::dartminhash_t ret(k, seed);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates a DartMinHash sketch from a numpy array of (unhashed) 64-bit integers");
}
