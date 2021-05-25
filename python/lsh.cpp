#include "python/pysketch.h"
#include "sketch/ssi.h"

using sketch::lsh::SetSketchIndex;
using namespace pybind11::literals;

#define DEC_SSI(name, T1, T2) \
class name: public SetSketchIndex<T1, T2> {\
    public:\
    template<typename...Args>\
    name(Args &&...args): SetSketchIndex<T1, T2>(std::forward<Args>(args)...) {}\
    name(size_t m, const py::array_t<int64_t, py::array::forcecast> lhs):\
        SetSketchIndex<T1, T2>(m, std::vector<int64_t>((const int64_t *)lhs.data(), lhs.data() + lhs.size())) {}\
    name(size_t m, const py::array_t<int64_t, py::array::forcecast> lhs, const py::array_t<int64_t, py::array::forcecast> rhs):\
        SetSketchIndex<T1, T2>(m, std::vector<int64_t>((const int64_t *)lhs.data(), lhs.data() + lhs.size()), std::vector<int64_t>((const int64_t *)rhs.data(), rhs.data() + rhs.size())) {}\
};
DEC_SSI(SSI_6464, uint64_t, uint64_t);
DEC_SSI(SSI_3264, uint32_t, uint64_t);
DEC_SSI(SSI_1664, uint16_t, uint64_t);
DEC_SSI(SSI_6432, uint64_t, uint32_t);
DEC_SSI(SSI_3232, uint32_t, uint32_t);
DEC_SSI(SSI_1632, uint16_t, uint32_t);
#undef DEC_SSI

template<typename T>
struct minispan {
    T *ptr_;
    size_t n_;
    minispan(T *ptr, size_t n): ptr_(ptr), n_(n) {}
    minispan(const T *ptr, size_t n): ptr_(const_cast<T *>(ptr)), n_(n) {}
    T *begin() {return ptr_;}
    const T *begin() const {return ptr_;}
    T *end() {return ptr_ + n_;}
    const T *end() const {return ptr_ + n_;}
    size_t size() const {return n_;}
    T *data() {return ptr_;}
    const T *data() const {return ptr_;}
    T &operator[](size_t idx) {return ptr_[idx];}
    const T &operator[](size_t idx) const {return ptr_[idx];}
};

template<typename SSI, typename TYPE, int FLAGS>
void update_one(SSI &index, py::array_t<TYPE, FLAGS> &arr) {
    auto arrinf = arr.request();
    const py::ssize_t nc = arrinf.size;
    minispan<TYPE> myspan(arr.data(), nc);
    index.update(myspan);
}
template<typename SSI, typename TYPE, int FLAGS>
void update_all(SSI &index, py::array_t<TYPE, FLAGS> &arr) {
    auto arrinf = arr.request();
    const py::ssize_t nc = arrinf.shape[1];
    for(py::ssize_t i = 0; i < arrinf.shape[0]; ++i) {
        minispan<TYPE> myspan(arr.data(i, 0), nc);
        index.update(myspan);
    }
}

template<typename SSI>
void declare_lsh_table(py::class_<SSI> &cls) {
    cls.def(py::init<size_t, bool>(), py::arg("m"), py::arg("densify") = false)
    .def(py::init<size_t>(), py::arg("m"))
    .def(py::init<size_t, py::array_t<int64_t, py::array::forcecast>>(), py::arg("m"), py::arg("persig"))
    .def(py::init<size_t, py::array_t<int64_t, py::array::forcecast>, py::array_t<int64_t, py::array::forcecast>>(), py::arg("m"), py::arg("persig"), py::arg("persigsize"))
    .def("m", &SSI::m)
    .def("size", &SSI::size)
    .def("add", [](SSI &index, py::object item) {
        if(py::isinstance<py::array>(item)) {
            auto arr = py::cast<py::array>(item);
            auto inf = arr.request();
            if(inf.format.size() > 1) throw std::invalid_argument(std::string("Required: simple dtype of one character length. Found: ") + inf.format);
            if(inf.ndim == 1) {
                if(inf.size != py::ssize_t(index.m())) throw std::invalid_argument("Wrong dimension");
                switch(inf.format[0]) {
                    case 'L': case 'l': {py::array_t<uint64_t, py::array::forcecast> arr64(arr); update_one(index, arr64);} break;
                    case 'I': case 'i': {py::array_t<uint32_t, py::array::forcecast> arr32(arr); update_one(index, arr32);} break;
                    case 'H': case 'h': {py::array_t<uint16_t, py::array::forcecast> arr16(arr); update_one(index, arr16);} break;
                    case 'B': case 'b': {py::array_t<uint8_t, py::array::forcecast> arr8(arr); update_one(index, arr8);} break;
                    case 'd': {py::array_t<double, py::array::forcecast> arrd(arr); update_one(index, arrd);} break;
                    case 'f': {py::array_t<float, py::array::forcecast> arrf(arr); update_one(index, arrf);} break;
                    default: throw std::invalid_argument(std::string("Unexpected dtype: ") + inf.format);
                }
            } else if(inf.ndim == 2) {
                if(inf.shape[1] != py::ssize_t(index.m())) throw std::invalid_argument("Wrong dimension on 2-D array");
                switch(inf.format[0]) {
                    case 'L': case 'l': {py::array_t<uint64_t, py::array::forcecast> arr64(arr); update_all(index, arr64);} break;
                    case 'I': case 'i': {py::array_t<uint32_t, py::array::forcecast> arr32(arr); update_all(index, arr32);} break;
                    case 'H': case 'h': {py::array_t<uint16_t, py::array::forcecast> arr16(arr); update_all(index, arr16);} break;
                    case 'B': case 'b': {py::array_t<uint8_t, py::array::forcecast> arr8(arr); update_all(index, arr8);} break;
                    case 'd': {py::array_t<double, py::array::forcecast> arrd(arr); update_all(index, arrd);} break;
                    case 'f': {py::array_t<float, py::array::forcecast> arrf(arr); update_all(index, arrf);} break;
                    default: throw std::invalid_argument(std::string("Unexpected dtype: ") + inf.format);
                }
            } else throw std::invalid_argument("Cannot process arrays with > 2 dimensions");
        } else throw std::invalid_argument("Can only add numpy arrays to the sketch");
    }, py::arg("item"))
    .def("query", [](SSI &index, py::array arr, py::ssize_t maxcand, py::ssize_t startidx) {
        auto inf = arr.request();
        std::tuple<std::vector<typename SSI::id_type>, std::vector<uint32_t>, std::vector<uint32_t>> ret;
        if(inf.format.size() > 1) throw std::invalid_argument(std::string("Required: simple dtype of one character length. Found: ") + inf.format);
        if(inf.ndim > 2) throw std::invalid_argument("too many (> 2) dimensions");
        else if(inf.ndim == 2) {
           throw std::invalid_argument("Currently not supported: 2-d queries");
        } else {
            const py::ssize_t nc = inf.size;
            switch(inf.format[0]) {
                case 'L': case 'l': {py::array_t<uint64_t, py::array::forcecast> arr64(arr); ret = std::move(index.query_candidates(minispan<uint64_t>(arr64.data(), nc), maxcand, startidx));} break;
                case 'I': case 'i': {py::array_t<uint32_t, py::array::forcecast> arr32(arr); ret = std::move(index.query_candidates(minispan<uint32_t>(arr32.data(), nc), maxcand, startidx));} break;
                case 'H': case 'h': {py::array_t<uint16_t, py::array::forcecast> arr16(arr); ret = std::move(index.query_candidates(minispan<uint16_t>(arr16.data(), nc), maxcand, startidx));} break;
                case 'B': case 'b': {py::array_t<uint8_t, py::array::forcecast> arr8(arr); ret = std::move(index.query_candidates(minispan<uint8_t>(arr8.data(), nc), maxcand, startidx));} break;
                case 'f': {py::array_t<float, py::array::forcecast> arrf(arr); ret = std::move(index.query_candidates(minispan<float>(arrf.data(), nc), maxcand, startidx));} break;
                case 'd': {py::array_t<double, py::array::forcecast> arrd(arr); ret = std::move(index.query_candidates(minispan<double>(arrd.data(), nc), maxcand, startidx));} break;
                default: throw std::invalid_argument(std::string("Unexpected dtype: ") + inf.format);
            }
        }
        auto &idref = std::get<0>(ret);
        auto &countsref = std::get<1>(ret);
        auto &iprref = std::get<2>(ret);
        const size_t nids = idref.size(), nipr = iprref.size();
        py::array_t<typename SSI::id_type> ids(nids);
        py::array_t<uint32_t> items_per_row(nipr);
        py::array_t<uint32_t> counts(nids);
        std::copy(idref.data(), idref.data() + idref.size(), ids.mutable_data());
        std::copy(countsref.data(), countsref.data() + countsref.size(), counts.mutable_data());
        std::copy(iprref.data(), iprref.data() + iprref.size(), items_per_row.mutable_data());
        return py::dict("ids"_a = ids, "per_row"_a = items_per_row, "counts"_a = counts);
    }, py::arg("item"), py::arg("maxcand") = 50, py::arg("start") = py::ssize_t(-1));
}

void init_lsh_table(py::module &m) {
    py::class_<SSI_6464> l64_64(m, "LSHTable64_64");
    py::class_<SSI_6432> l64_32(m, "LSHTable64_32");
    py::class_<SSI_3264> l32_64(m, "LSHTable32_64");
    py::class_<SSI_3232> l32_32(m, "LSHTable32_32");
    py::class_<SSI_1664> l16_64(m, "LSHTable16_64");
    py::class_<SSI_1632> l16_32(m, "LSHTable16_32");
    declare_lsh_table(l64_64); declare_lsh_table(l32_64); declare_lsh_table(l16_64);
    declare_lsh_table(l64_32); declare_lsh_table(l32_32); declare_lsh_table(l16_32);
}

PYBIND11_MODULE(sketch_lsh, m) {
    m.doc() = "Python bindings LSH-table building. Supports keys of 16, 32, or 64 bits and values of 32 or 64 bits for index hits.";
    init_lsh_table(m);
}
