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
    using value_type = T;
    static T *allocate(size_t nelem) {
        void *ret;
        const size_t nb = nelem * sizeof(T);
        CONST_IF(aln) {
            if(posix_memalign(&ret, aln, nb)) {
                throw std::bad_alloc();
            }
        } else {
            if((ret = std::malloc(nb)) == nullptr) {
                throw std::bad_alloc();
            }
        }
        return static_cast<T *>(ret);
    }
    template<typename It>
    vector(It i1, It i2): data_(allocate(std::distance(i1, i2))), n_(std::distance(i1, i2)) {
        std::copy(i1, i2, data_);
    }
    vector(size_t n, T initial_value=T()): data_(allocate(n)), n_(n) {
        std::fill_n(data_, n_, initial_value);
    }
    ~vector() {std::free(data_);}
    vector(): data_(nullptr), n_(0) {}
    vector &operator=(const vector &o) {
        auto tmp = static_cast<T *>(std::realloc(data_, o.n_ * sizeof(T)));
        if(tmp == nullptr) throw std::bad_alloc();
        data_ = tmp;
        n_ = o.n_;
        std::copy(o.data_, o.data_ + n_, data_);
        return *this;
    }
    void resize(size_t newsize, const T initial_value=T()) {
        if(newsize <= n_) {
            n_ = newsize;
            return;
        }
        auto tmp = allocate(newsize);
        CONST_IF(std::is_trivially_destructible<T>::value) {
            std::copy(data_, data_ + n_, tmp);
        } else {
            std::move(data_, data_ + n_, tmp);
        }
        std::fill_n(tmp + n_, newsize - n_, initial_value);
        data_ = tmp;
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
    T &front() {return data_[0];}
    const T &front() const {return data_[0];}
    T &back() {return data_[n_ - 1];}
    const T &back() const {return data_[n_ - 1];}
    void fill(const T val) {
        std::fill(this->begin(), this->end(), val);
    }
    bool operator<(const vector &o) const {
        return std::lexicographical_compare(begin(), end(), o.begin(), o.end());
    }
    bool operator>(const vector &o) const {
        return std::lexicographical_compare(begin(), end(), o.begin(), o.end(), std::greater<T>());
    }
};

}

#endif
