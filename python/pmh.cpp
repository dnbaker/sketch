#include "python/pysketch.h"
#include "sketch/bmh.h"

PYBIND11_MODULE(sketch_pmh, m) {
    m.doc() = "Permutation MinHash (PMH1/PMH2, Ertl; weighted minhash sketches)";
    py::class_<sketch::wmh::pmh2_t<>> (m, "pmh2")
        .def(py::init<size_t>(), py::arg("m") = 1024)
        .def("add", [](sketch::wmh::pmh2_t<> &h, uint64_t v) {h.add(v);},
             "Add a pre-hashed value at weight 1 (unweighted).")
        .def("add", [](sketch::wmh::pmh2_t<> &h, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h.add(ptr[i++]));
        }, "Add a batch of pre-hashed values (weight 1) from a 1-D numpy uint64 array.")
        .def("addh", [](sketch::wmh::pmh2_t<> &h, uint64_t v) {h.addh(v);},
             "Hash an integer value and add it at weight 1.")
        .def("addh_np", [](sketch::wmh::pmh2_t<> &h, const py::array_t<uint64_t> &a) {
            auto ptr = a.data();
            for(ssize_t i = 0; i < a.size(); h.addh(ptr[i++]));
        }, "Hash and add a 1-D numpy uint64 array of values (weight 1).")
        .def("update", [](sketch::wmh::pmh2_t<> &h, uint64_t id, double w) {h.update(id, w);},
             "Weighted update: add id with weight w.")
        .def("jaccard_index", &sketch::wmh::pmh2_t<>::jaccard_index,
             "Jaccard estimate from matching register winners (exact for unweighted sets; weighted is approximate).")
        .def("report", &sketch::wmh::pmh2_t<>::report, "Total weight added (exact).")
        .def("total_weight", &sketch::wmh::pmh2_t<>::total_weight, "Total weight added (exact).")
        .def("size", &sketch::wmh::pmh2_t<>::size, "Number of registers.")
        .def("__ior__", [](sketch::wmh::pmh2_t<> &lh, const sketch::wmh::pmh2_t<> &rh) {lh += rh; return lh;})
        .def("__or__", [](const sketch::wmh::pmh2_t<> &lh, const sketch::wmh::pmh2_t<> &rh) {
            auto ret = lh;
            ret += rh;
            return ret;
        })
        .def("__repr__", [](const sketch::wmh::pmh2_t<> &h) {
            return std::string("pmh2(m=") + std::to_string(h.size()) + ", weight=" + std::to_string(h.report()) + ")";
        });
    m.def("from_np", [](const py::array_t<uint64_t> &input, size_t m=1024) {
         sketch::wmh::pmh2_t<> ret(m);
         auto ptr = input.data();
         for(ssize_t i = 0; i < input.size(); ret.addh(ptr[i++]));
         return ret;
     }, py::return_value_policy::take_ownership, "Creates a PMH2 sketch from a numpy array of (unhashed) 64-bit integers");
}
