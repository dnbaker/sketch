#ifndef FIXED_VECTOR_H__
#define FIXED_VECTOR_H__
#include <stdexcept>
#include <cstdlib>
#include <type_traits>
#include <memory>

#ifndef CONST_IF
#if __cplusplus >= 201703L
#define CONST_IF if constexpr
#else
#define CONST_IF if
#endif
#endif

namespace fixed {

template<typename T, size_t aln=0>
class vector {
    static_assert(std::is_trivial<T>::value, "T must be a trivial type");
    T *data_;
    const size_t n_;
public:
    static T *allocate(size_t nelem) {
        T *ret;
        CONST_IF(aln) {
            ret = nullptr;
            (void)posix_memalign((void **)&ret, aln, nelem * sizeof(T));
        } else ret = static_cast<T *>(std::malloc(sizeof(T) * nelem));
        if(ret == nullptr) throw std::bad_alloc();
        return ret;
    }
    template<typename It>
    vector(It i1, It i2): data_(allocate(std::distance(i1, i2))), n_(std::distance(i1, i2)) {
        std::copy(i1, i2, data_);
    }
    vector(size_t n): data_(allocate(n)), n_(n) {
        
    }
    ~vector() {std::free(data_);}
    vector &operator=(const vector &o) = delete;
    vector &operator=(vector &&o) {
        if(data_) {
            std::free(data_);
        }
        data_ = o.data_; n = o.n_;
    }
    vector(vector &&o): n_(o.n_) {
        if(this == std::addressof(o)) return;
        data_ = o.data_;
        o.data_ = nullptr;
    }
    vector(const vector &o): vector(o.size()) {
        std::copy(o.begin(), o.end(), begin());
    }
    auto begin() {return data_;}
    auto begin() const {return data_;}
    auto end() {return data_ + n_;}
    auto end() const {return data_ + n_;}
    auto size() const {return n_;}
    T &operator[](size_t k) {return data_[k];}
    const T &operator[](size_t k) const {return data_[k];}
    const T *data() const {return data_;}
    T       *data()       {return data_;}

    bool operator<(const vector &o) const {
        return std::lexicographic_compare<T>(begin(), end(), o.begin(), o.end());
    }
    bool operator>(const vector &o) const {
        return std::lexicographic_compare<T>(begin(), end(), o.begin(), o.end(), std::greater<T>());
    }
};

}

#endif
