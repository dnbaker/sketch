#pragma once
#ifndef CIRCULAR_QUEUE_H__
#define CIRCULAR_QUEUE_H__
#include <new>         // For placement new
#include <cassert>     // For assert
#include <cstdlib>     // For std::size_t
#include <cstdint>     // For std::uint*_ts
#include <stdexcept>   // For std::bad_alloc
#include <string>      // for std::string [for exception handling]
#include <type_traits> // For std::enable_if_t/std::is_unsigned_v
#include <cstring>     // For std::memcpy
#include <vector>      // For converting to vector
#include <climits>     // CHAR_BIT
#include <iostream>
#include <algorithm>

namespace circ {
using std::size_t;
using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;

#if __cplusplus >= 201703L
#define CIRC_CONSTIF if constexpr
#else
#define CIRC_CONSTIF if
#endif

template<typename T, typename SizeType=uint32_t>
class deque;


template<typename T>
static inline T roundup(T x) {
    // With -O2 or higher, this full unrolls for all types
    // This also eliminates warnings from other attempts to generalize the approach.
    --x;
    unsigned i = 1;
    do {x |= (x >> i);} while((i <<= 1) < sizeof(T) * CHAR_BIT);
    return ++x;
}

template<typename T, typename SizeType>
class circular_iterator {
    using size_type = SizeType;
    // TODO: increment by an integral quantity.
    using deque_type = deque<T, SizeType>;
    deque_type *ref_;
    deque_type &ref() {return *ref_;}
    const deque_type &ref() const {return *ref_;}
    SizeType           pos_;
public:
    circular_iterator(deque_type &reff, SizeType pos): ref_(std::addressof(reff)), pos_(pos) {}
    circular_iterator(const circular_iterator &other): ref_(other.ref_), pos_(other.pos_) {}
    std::ptrdiff_t operator-(circular_iterator o) const {
        return (o.pos_ - pos_) & ref().mask();
    }
    std::ptrdiff_t operator+(circular_iterator o) const {
        return (o.pos_ + pos_) & ref().mask();
    }
    circular_iterator &operator=(const circular_iterator &o) {
        this->ref_ = o.ref_;
        this->pos_ = o.pos_;
        return *this;
    }
    circular_iterator operator-(std::ptrdiff_t i) const {
        circular_iterator tmp(*this);
        tmp -= i;
        return tmp;
    }
    circular_iterator operator+(std::ptrdiff_t i) const {
        circular_iterator tmp(*this);
        tmp += i;
        return tmp;
    }
    circular_iterator &operator-=(std::ptrdiff_t i) {
        pos_ -= i;
        pos_ &= ref().mask();
        return *this;
    }
    circular_iterator &operator+=(std::ptrdiff_t i) {
        pos_ += i;
        pos_ &= ref().mask();
        return *this;
    }
    T &operator*() {
        return ref().data()[pos_];
    }
    const T &operator*() const noexcept {
        return ref().data()[pos_];
    }
    T *operator->() {
        return &ref().data()[pos_];
    }
    const T *operator->() const {
        return &ref().data()[pos_];
    }
    circular_iterator &operator++() noexcept {
        ++pos_;
        pos_ &= ref().mask();
        return *this;
    }
    circular_iterator operator++(int) noexcept {
        circular_iterator copy(*this);
        this->operator++();
        return copy;
    }
    bool operator==(const circular_iterator &other) const noexcept {
        return pos_ == other.pos_;
    }
    bool operator!=(const circular_iterator &other) const noexcept {
        return pos_ != other.pos_;
    }
    bool operator<(const circular_iterator &other) const noexcept {
        return pos_ < other.pos_;
    }
    bool operator<=(const circular_iterator &other) const noexcept {
        return pos_ <= other.pos_;
    }
    bool operator>(const circular_iterator &other) const noexcept {
        return pos_ > other.pos_;
    }
    bool operator>=(const circular_iterator &other) const noexcept {
        return pos_ >= other.pos_;
    }
};
template<typename T, typename SizeType>
class const_circular_iterator {
    using size_type = SizeType;
    using deque_type = deque<T, SizeType>;
    const deque_type *ref_;
    SizeType          pos_;
    auto &ref() {return *ref_;}
public:
    const_circular_iterator(const deque_type &ref, SizeType pos): ref_(&ref), pos_(pos) {}
    const_circular_iterator(const const_circular_iterator &other): ref_(&other.ref_), pos_(other.pos_) {}
    std::ptrdiff_t operator-(const_circular_iterator o) const {
        return (o.pos_ - pos_) & ref().mask();
    }
    std::ptrdiff_t operator+(const_circular_iterator o) const {
        return (o.pos_ + pos_) & ref().mask();
    }
    const_circular_iterator &operator=(const const_circular_iterator &o) {
        this->ref_ = o.ref_;
        this->pos_ = o.pos_;
        return *this;
    }
    const_circular_iterator operator-(std::ptrdiff_t i) const {
        const_circular_iterator tmp(*this);
        tmp -= i;
        return tmp;
    }
    const_circular_iterator operator+(std::ptrdiff_t i) const {
        const_circular_iterator tmp(*this);
        tmp += i;
        return tmp;
    }
    const_circular_iterator &operator-=(std::ptrdiff_t i) {
        pos_ -= i;
        pos_ &= ref().mask();
        return *this;
    }
    const_circular_iterator &operator+=(std::ptrdiff_t i) {
        pos_ += i;
        pos_ &= ref().mask();
        return *this;
    }
    const T &operator*() const noexcept {
        return ref().data()[pos_];
    }
    const_circular_iterator &operator++() noexcept {
        ++pos_;
        pos_ &= ref().mask();
        return *this;
    }
    const_circular_iterator operator++(int) noexcept {
        const_circular_iterator copy(*this);
        this->operator++();
        return copy;
    }
    bool operator==(const const_circular_iterator &other) const noexcept {
        return pos_ == other.pos_;
    }
    bool operator!=(const const_circular_iterator &other) const noexcept {
        return pos_ != other.pos_;
    }
    bool operator<(const const_circular_iterator &other) const noexcept {
        return pos_ < other.pos_;
    }
    bool operator<=(const const_circular_iterator &other) const noexcept {
        return pos_ <= other.pos_;
    }
    bool operator>(const const_circular_iterator &other) const noexcept {
        return pos_ > other.pos_;
    }
    bool operator>=(const const_circular_iterator &other) const noexcept {
        return pos_ >= other.pos_;
    }
};

template<typename T, typename SizeType>
class deque {
    // A circular queue in which extra memory has been allocated up to a power of two.
    // This allows us to use bitmasks instead of modulus operations.
    // This circular queue is NOT threadsafe. Its purpose is creating a double-ended queue without
    // the overhead of a doubly-linked list.
    SizeType  mask_;
    SizeType start_;
    SizeType  stop_;
    T        *data_;
    static_assert(std::is_unsigned<SizeType>::value, "Must be unsigned");

public:
    using size_type = SizeType;
    using iterator = circular_iterator<T, size_type>;
    using const_iterator = const_circular_iterator<T, size_type>;
    deque(SizeType size=3):
            mask_(roundup(size + 1) - 1),
            start_(0), stop_(0),
            data_(static_cast<T *>(std::malloc((mask_ + 1) * sizeof(T))))
    {
        assert((mask_ & (mask_ + 1)) == 0);
        if(data_ == nullptr) {
            throw std::bad_alloc();
        }
    }
    deque(deque &&other) {
        if(&other == this) return;
        std::memcpy(this, &other, sizeof(*this));
        other.mask_ = other.start_ = other.stop_ = SizeType(0);
        other.data_ = nullptr;
    }
    deque(const deque &other) {
        if(&other == this) return;
        start_ = other.start_;
        stop_  = other.stop_;
        mask_  = other.mask_;
        auto tmp = static_cast<T *>(std::malloc(sizeof(T) * (mask_ + 1)));
        if(__builtin_expect(tmp == nullptr, 0)) throw std::bad_alloc();
        data_ = tmp;
        for(auto i(other.start_); i != other.stop_; data_[i] = other.data_[i], i = (i+1) & mask_);
    }
    iterator begin() noexcept {
        return iterator(*this, start_);
    }
    iterator end() noexcept {
        return iterator(*this, stop_);
    }
    const_iterator cbegin() const noexcept {
        return const_iterator(*this, start_);
    }
    const_iterator cend()   const noexcept {
        return const_iterator(*this, stop_);
    }
    auto start() const {return start_;}
    auto stop() const {return stop_;}
    auto mask() const {return mask_;}
    auto data() const {return data_;}
    auto data()       {return data_;}
    void resize(size_type new_size) {
        // TODO: this will cause a leak if the later malloc fails.
        // Fix this by only copying the temporary buffer to data_ in case of success in all allocations.
        if(__builtin_expect(new_size < mask_, 0)) throw std::runtime_error("Attempting to resize to value smaller than queue's size, either from user error or overflowing the size_type. Abort!");
        new_size = roundup(new_size); // Is this necessary? We can hide resize from the user and then cut out this call.
        new_size = std::max(size_type(4), new_size);
        auto tmp = std::realloc(data_, new_size * sizeof(T));
        if(tmp == nullptr) throw std::bad_alloc();
        data_ = static_cast<T *>(tmp);
        if(start_ == stop_) {
            if(start_) {
                stop_ = mask_ + 1;
                auto tmp = static_cast<T *>(std::malloc((stop_) * sizeof(T)));
                if(tmp == nullptr) throw std::bad_alloc();
                std::memcpy(tmp, data_ + start_, (stop_ - start_) * sizeof(T));
                std::memcpy(tmp + (stop_ - start_), data_, stop_ * sizeof(T));
                std::memcpy(data_, tmp, (stop_) * sizeof(T));
                std::free(tmp);
                start_ = 0;
            }
        } else if(stop_ < start_) {
            auto tmp = size();
            std::rotate(this->begin(), this->end(), iterator(*this, mask_));
            start_ = 0;
            stop_ = tmp;
        }
        mask_ = new_size - 1;
    }
    // Does not yet implement push_front.
    template<typename... Args>
    T &push_back(Args &&... args) {
        if(__builtin_expect(((stop_ + 1) & mask_) == start_, 0)) {
            resize((mask_ + 1) << 1);
        }
        size_type ind = stop_;
        ++stop_; stop_ &= mask_;
        return *(new(data_ + ind) T(std::forward<Args>(args)...));
    }
    template<typename... Args>
    T &push_front(Args &&... args) {
        if(((start_ - 1) & mask_) == stop_) {
            resize((mask_ + 1) << 1);
        }
        start_ = (start_ - 1) & mask_;
        assert(start_ <= mask_);
        assert(start_ > 0);
        return *(new(data_ + start_) T(std::forward<Args>(args)...));
    }
    template<typename... Args>
    T &emplace_back(Args &&... args) {
        return push_back(std::forward<Args>(args)...); // Interface compatibility.
    }
    template<typename... Args>
    T &emplace_front(Args &&... args) {
        return push_front(std::forward<Args>(args)...); // Interface compatibility.
    }
    T pop() {
        if(__builtin_expect(stop_ == start_, 0)) throw std::runtime_error("Popping item from empty buffer. Abort!");
        T ret(std::move(data_[start_++]));
        start_ &= mask_;
        return ret; // If unused, the std::move causes it to leave scope and therefore be destroyed.
    }
    T pop_back() {
        if(__builtin_expect(stop_ == start_, 0)) throw std::runtime_error("Popping item from empty buffer. Abort!");
        T ret(std::move(data_[--stop_]));
        start_ &= mask_;
        return ret; // If unused, the std::move causes it to leave scope and therefore be destroyed.
    }
    T pop_front() {
        return pop(); // Interface compatibility with std::list.
    }
    template<typename... Args>
    T push_pop(Args &&... args) {
        T ret(pop());
        push(std::forward<Args>(args)...);
        return ret;
    }
    template<typename... Args>
    T &push(Args &&... args) {
        return push_back(std::forward<Args>(args)...); // Interface compatibility
    }
    T &back() {
        return data_[(stop_ - 1) & mask_];
    }
    const T &back() const {
        return data_[(stop_ - 1) & mask_];
    }
    T &front() {
        return data_[start_];
    }
    const T &front() const {
        return data_[start_];
    }
    template<typename Functor>
    void for_each(const Functor &func) {
        for(SizeType i = start_; i != stop_; func(data_[i++]), i &= mask_);
    }
    template<typename Functor>
    void for_each(const Functor &func) const {
        for(SizeType i = start_; i != stop_; func(data_[i++]), i &= mask_);
    }
    void show() const {
        assert(data_);
        for_each([](auto x) {std::cerr << x << '\n';});
    }
    ~deque() {this->free();}
    size_type capacity() const noexcept {return mask_;}
    size_type size()     const noexcept {return (stop_ - start_) & mask_;}
    std::vector<T> to_vector() const {
        std::vector<T> ret;
        ret.reserve(this->size());
        for(size_type i(start_); i != stop_; ret.emplace_back(std::move(data_[i])), ++i, i &= mask_);
        return ret;
    }
    void clear() {
        //CIRC_CONSTIF(std::is_destructible<T>::value)
            for(size_type i(start_); i != stop_; data_[i++].~T(), i &= mask_);
        start_ = stop_ = 0;
    }
    void free() {
        clear();
        std::free(data_);
    }
}; // deque


template<typename T, typename S=std::uint32_t>
class FastCircularQueue: public deque<T, S> {
public:
    template<typename...Args>
    FastCircularQueue(Args &&...args): deque<T, S>(std::forward<Args>(args)...) {
        std::fprintf(stderr, "Warning: FastCircularQueue has been deprecated. Call as circ::deque instead.\n");
    }
};

} // namespace circ
namespace std {
template<typename T, typename SizeType>
struct iterator_traits<circ::circular_iterator<T, SizeType>> {
    using difference_type = std::ptrdiff_t;
    using reference_type = T &;
    using pointer        = T *;
    using value_type = T;
    struct iterator_category: public forward_iterator_tag {};
};

template<typename T, typename SizeType>
struct iterator_traits<circ::const_circular_iterator<T, SizeType>> {
    using difference_type = std::ptrdiff_t;
    using reference_type = T &;
    using pointer        = T *;
    using value_type = T;
    struct iterator_category: public forward_iterator_tag {};
};
} // namespace std

#endif /* #ifndef CIRCULAR_QUEUE_H__ */
