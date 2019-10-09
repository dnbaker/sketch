#ifndef FIXED_VECTOR_H__
#define FIXED_VECTOR_H__
#include <stdexcept>
#include <cstdlib>
#include <type_traits>
#include <memory>

#ifndef CONST_IF
#if __cplusplus >= 201703L
#define CONST_IF(...) if constexpr(__VA_ARGS__)
#else
#define CONST_IF(...) if(__VA_ARGS__)
#endif
#endif

namespace fixed {

template<typename T, size_t aln=0>
class vector {
    static_assert(std::is_trivially_destructible<T>::value, "T must not have a destructor to call");
    T *data_;
    size_t n_;
public:
    static T *allocate(size_t nelem) {
        T *ret;
        const size_t nb = nelem * sizeof(T);
        CONST_IF(aln) {
#if _GLIBCXX_HAVE_ALIGNED_ALLOC
            ret = std::aligned_alloc(aln, nb);
#else
            ret = nullptr;
            (void)posix_memalign((void **)&ret, aln, nb);
#endif
        } else {
            ret = static_cast<T *>(std::malloc(nb));
        }
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
    vector &operator=(const vector &o) {
        auto tmp = static_cast<T *>(std::realloc(data_, o.n_ * sizeof(T)));
        if(tmp == nullptr) throw std::bad_alloc();
        data_ = tmp;
        n_ = o.n_;
        std::copy(o.data_, o.data_ + n_, data_);
        return *this;
    }
    vector &operator=(vector &&o) {
        std::free(data_);
        data_ = o.data_;
        n_ = o.n_;
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
        return std::lexicographical_compare(begin(), end(), o.begin(), o.end());
    }
    bool operator>(const vector &o) const {
        return std::lexicographical_compare(begin(), end(), o.begin(), o.end(), std::greater<T>());
    }
};

}

#endif
