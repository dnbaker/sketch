#include "pysketch.h"


PYBIND11_MODULE(sketch_bbmh, m) {
    m.doc() = "BBitMinHash support";
    py::class_<mh::FinalBBitMinHash> (m, "FinalBBitMinHash")
        .def(py::init<std::string>())
        .def("jaccard_index", [](mh::FinalBBitMinHash &lhs, mh::FinalBBitMinHash &rhs) {
            return lhs.jaccard_index(rhs);
        })
        .def("containment_index", [](mh::FinalBBitMinHash &lhs, mh::FinalBBitMinHash &rhs) {
            return lhs.containment_index(rhs);
        })
        .def("popcnt", [](const mh::FinalBBitMinHash &lhs) {
            return lhs.popcnt();
        })
        .def("size", [](const mh::FinalBBitMinHash &lhs) {return lhs.nmin();})
        .def("equal_blocks", [](mh::FinalBBitMinHash &lhs, mh::FinalBBitMinHash &rhs) {
            return lhs.equal_bblocks(rhs);
        }).def("write", [](mh::FinalBBitMinHash &lhs, std::string path) {
            lhs.write(path);
        })
        .def("compress", [](const mh::BBitMinHasher<uint64_t> &h1, unsigned newnp) {return h1.compress(newnp);},
             py::return_value_policy::take_ownership,
            "Compress an b-bit minhash sketch from a previous prefix length to a smaller one.");

    py::class_<mh::BBitMinHasher<uint64_t>> (m, "BBitMinHasher")
        .def(py::init<size_t, unsigned>())
        .def("clear", &mh::BBitMinHasher<uint64_t>::clear, "Clear all entries.")
        //.def("report", &mh::BBitMinHasher<uint64_t>::report, "Emit estimated cardinality. Performs sum if not performed, but sum must be recalculated if further entries are added.")
        .def("add", [](mh::BBitMinHasher<uint64_t> &h1, uint64_t v) {h1.add(v);}, "Add a (hashed) value to the sketch.")
        .def("addh", [](mh::BBitMinHasher<uint64_t> &h1, uint64_t v) {h1.addh(WangHash()(v));}, "Hash an integer value and then add that to the sketch..")
        .def("jaccard_index", [](mh::BBitMinHasher<uint64_t> &h1, mh::BBitMinHasher<uint64_t> &h2) {return jaccard_index(h2, h2);})
        .def("write", [](mh::BBitMinHasher<uint64_t> &lhs, std::string path) {
                lhs.write(path);
        }, "Write sketch to disk at path")
        .def("finalize", [](mh::BBitMinHasher<uint64_t> &lhs) {
            mh::FinalBBitMinHash ret = lhs.finalize();
            return ret;
        }, py::return_value_policy::take_ownership);
        //.def("sprintf", &mh::BBitMinHasher<uint64_t>::sprintf)
        //.def("union", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1 + h2;})
        //.def("union_size", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1.union_size(h2);})

    m.def("jaccard_index", [](mh::BBitMinHasher<uint64_t> &h1, mh::BBitMinHasher<uint64_t> &h2) {
            return jaccard_index(h1, h2);
        }, "Calculates jaccard indexes between two sketches")
    .def("from_shs", [](const py::array_t<uint64_t> &input, size_t ss=10, unsigned b=32) {
         mh::BBitMinHasher<uint64_t> ret(ss, b);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.add(ptr[i++]));
         return ret;
    }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of 64-bit hashes.")
    .def("from_np", [](const py::array_t<uint64_t> &input, size_t ss=10) {
         mh::BBitMinHasher<uint64_t> ret(ss);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size();ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates an HLL sketch from a numpy array of (unhashed) 64-bit integers")
    .def("union_size", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1.union_size(h2);}, "Calculate union size");
    //.def("union_size", [](const mh::BBitMinHasher<uint64_t> &h1, const mh::BBitMinHasher<uint64_t> &h2) {return h1.union_size(h2);}, "Calculate union size")
} // pybind11 module
