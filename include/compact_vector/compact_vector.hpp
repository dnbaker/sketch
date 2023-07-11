#ifndef __COMPACT_VECTOR_H__
#define __COMPACT_VECTOR_H__

#include <new>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#include "compact_iterator.hpp"

namespace compact {

namespace vector_imp {
inline int clz(unsigned int x) { return __builtin_clz(x); }
inline int clz(unsigned long x) { return __builtin_clzl(x); }
inline int clz(unsigned long long x) { return __builtin_clzll(x); }

template<class Derived,
         typename IDX, unsigned BITS, typename W, typename Allocator, unsigned UB, bool TS>
class vector {
  Allocator m_allocator;
  size_t    m_size;             // Size in number of elements
  size_t    m_capacity;         // Capacity in number of elements
  W*        m_mem;

public:
  // Number of bits required for indices/values in the range [0, s).
  static unsigned required_bits(size_t s) {
    unsigned res = bitsof<size_t>::val - 1 - clz(s);
    res += (s > ((size_t)1 << res)) + (std::is_signed<IDX>::value ? 1 : 0);
    return res;
  }

  static size_t elements_to_words(size_t size, unsigned bits) {
    const size_t total_bits = size * bits;
    return total_bits / UB + (total_bits % UB != 0);
  }

  typedef compact::iterator<IDX, BITS, W, TS, UB>   iterator;
  typedef compact::const_iterator<IDX, BITS, W, UB> const_iterator;
  typedef compact::iterator<IDX, BITS, W, true, UB> mt_iterator; // Multi thread safe version
  typedef std::reverse_iterator<iterator>        reverse_iterator;
  typedef std::reverse_iterator<const_iterator>  const_reverse_iterator;

protected:
  static W* allocate_s(size_t capacity, unsigned bits, Allocator& allocator) {
    const auto nb_words = elements_to_words(capacity, bits);
    W* res = allocator.allocate(nb_words);
    if(UB != bitsof<W>::val) // CAS vector, expect high bit of each word to be zero, so zero it all
      std::fill_n(res, nb_words, (W)0);
    return res;
  }

  W* allocate(size_t capacity) {
    return allocate_s(capacity, bits(), m_allocator);
  }

  void deallocate(W* mem, size_t capacity) {
    m_allocator.deallocate(mem, elements_to_words(capacity, bits()));
  }

  // Error messages
  static constexpr const char* EOUTOFRANGE = "Index is out of range";

public:

  vector(vector &&rhs)
    : m_allocator(std::move(rhs.m_allocator))
    , m_size(rhs.m_size)
    , m_capacity(rhs.m_capacity)
    , m_mem(rhs.m_mem)
  {
    rhs.m_size = rhs.m_capacity = 0;
    rhs.m_mem = nullptr;
  }
  vector(const vector &rhs)
    : m_allocator(rhs.m_allocator)
    , m_size(rhs.m_size)
    , m_capacity(rhs.m_capacity)
    , m_mem(allocate_s(m_capacity, rhs.bits(), m_allocator))
  {
    std::memcpy(m_mem, rhs.m_mem, rhs.bytes());
  }

  vector(unsigned b, size_t s, Allocator allocator = Allocator())
    : m_allocator(allocator)
    , m_size(s)
    , m_capacity(s)
    , m_mem(allocate_s(s, b, m_allocator))
  {
    static_assert(UB <= bitsof<W>::val, "used_bits must be less or equal to the number of bits in the word_type");
    static_assert(BITS <= UB, "number of bits larger than usable bits");
  }
  explicit vector(Allocator allocator = Allocator())
    : vector(0, 0, allocator)
  { }
  ~vector() {
    m_allocator.deallocate(m_mem, elements_to_words(m_capacity, bits()));
  }

  vector& operator=(const vector& rhs) {
    m_allocator = rhs.m_allocator;
    if(m_capacity < rhs.size()) {
      deallocate(m_mem, m_capacity);
      m_capacity = rhs.size();
      m_mem = allocate(m_capacity);
    }
    m_size      = rhs.m_size;
    std::memcpy(m_mem, rhs.m_mem, bytes());
    return *this;
  }

  vector& operator=(vector&& rhs) {
    m_allocator = std::move(rhs.m_allocator);
    m_size      = rhs.m_size;
    m_capacity  = rhs.m_capacity;
    m_mem       = rhs.m_mem;

    rhs.m_size = rhs.m_capacity = 0;
    rhs.m_mem  = nullptr;
    return *this;
  }

  const_iterator begin() const { return const_iterator(m_mem, bits(), 0); }
  iterator begin() { return iterator(m_mem, bits(), 0); }
  const_iterator end() const { return begin() + m_size; }
  iterator end() { return begin() + m_size; }
  const_iterator cbegin() const { return begin(); }
  const_iterator cend() const { return end(); }
  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator crbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator crend() const { return const_reverse_iterator(begin()); }

  // Multi thread safe iterator
  mt_iterator mt_begin() { return mt_iterator(m_mem, bits(), 0); }
  mt_iterator mt_end() { return begin() + m_size; }

  IDX operator[](size_t i) const {
    return BITS
      ? *const_iterator(m_mem + (i * BITS) / UB, BITS, (i * BITS) % UB)
      : *const_iterator(m_mem + (i * bits()) / UB, bits(), (i * bits()) % UB);
    // return cbegin()[i];
  }
  IDX at(size_t i) const {
    if(i >= size()) throw std::out_of_range(EOUTOFRANGE);
    return this->operator[](i);
  }
  typename iterator::lhs_setter_type operator[](size_t i) {
    return BITS
      ? typename iterator::lhs_setter_type(m_mem + (i * BITS) / UB, BITS, (i * BITS) % UB)
      : typename iterator::lhs_setter_type(m_mem + (i * bits()) / UB, bits(), (i * bits()) % UB);
    //  return begin()[i];
  }
  typename iterator::lhs_setter_type at(size_t i) {
    if(i >= size()) throw std::out_of_range(EOUTOFRANGE);
    return this->operator[](i);
  }

  template <class InputIterator>
  void assign (InputIterator first, InputIterator last) {
    clear();
    for( ; first != last; ++first)
      push_back(*first);
  }
  void assign (size_t n, const IDX& val) {
    clear();
    for(size_t i = 0; i < n; ++i)
      push_back(val);
  }
  inline void assign (std::initializer_list<IDX> il) {
    assign(il.begin(), il.end());
  }

  void resize (size_t n, const IDX& val) {
    if(n <= size()) {
      m_size = n;
      return;
    }
    if(n > m_capacity)
      enlarge(n);

    auto it = begin() + size();
    for(size_t i = size(); i < n; ++i, ++it)
      *it = val;
    m_size = n;
  }
  inline void resize (size_t n) { resize(n, IDX()); }

  inline iterator erase (const_iterator position) { return erase(position, position + 1); }
  iterator erase (const_iterator first, const_iterator last) {
    const auto length = last - first;
    iterator res(begin() + (first - cbegin()));
    if(length) {
      std::copy(last, cend(), res);
      m_size -= length;
    }
    return res;
  }


  IDX front() const { return *cbegin(); }
  typename iterator::lhs_setter_type front() { return *begin(); }
  IDX back() const { return *(cbegin() + (m_size - 1)); }
  typename iterator::lhs_setter_type back() { return *(begin() + (m_size - 1)); }

  size_t size() const { return m_size; }
  bool empty() const { return m_size == 0; }
  size_t capacity() const { return m_capacity; }

  void push_back(IDX x) {
    if(m_size == m_capacity)
      enlarge();
    *end() = x;
    ++m_size;
  }

  void pop_back() { --m_size; }
  void clear() { m_size = 0; }
  iterator emplace (const_iterator position, IDX x) {
    const ssize_t osize = size();
    const ssize_t distance = position - begin();
    if(distance == osize) {
      push_back(x);
      return begin() + distance;
    }
    push_back(IDX());
    auto res = begin() + distance;
    std::copy_backward(res, begin() + osize, end());
    *res = x;
    return res;
  }
  void emplace_back(IDX x) { push_back(x); }

  W* get() { return m_mem; }
  const W* get() const { return m_mem; }
  size_t bytes() const { return sizeof(W) * elements_to_words(m_capacity, bits()); }
  inline unsigned bits() const { return static_cast<const Derived*>(this)->bits(); }
  static constexpr unsigned static_bits() { return BITS; }
  static constexpr unsigned used_bits() { return UB; }
  static constexpr bool thread_safe() { return TS; }
  // Zero out the entire memory array. Every element is 0 after this call.
  void zero() {
    std::fill_n(get(), elements_to_words(capacity(), bits()), (W)0);
  }

protected:
  void enlarge(size_t given = 0) {
    const size_t new_capacity = !given ? std::max(m_capacity * 2, (size_t)(bitsof<W>::val / bits() + 1)) : given;
    W* new_mem = allocate(new_capacity);
    std::copy(m_mem, m_mem + elements_to_words(m_capacity, bits()), new_mem);
    deallocate(m_mem, m_capacity);
    m_mem      = new_mem;
    m_capacity = new_capacity;
  }
};

template<typename IDX, typename W, typename Allocator, unsigned UB, bool TS>
class vector_dyn
  : public vector_imp::vector<vector_dyn<IDX, W, Allocator, UB, TS>, IDX, 0, W, Allocator, UB, TS>
{
  typedef vector_imp::vector<vector_dyn<IDX, W, Allocator, UB, TS>, IDX, 0, W, Allocator, UB, TS> super;
  const unsigned m_bits;    // Number of bits in an element

public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  vector_dyn(unsigned b, size_t s, Allocator allocator = Allocator())
    : super(b, s, allocator)
    , m_bits(b)
  { }
  vector_dyn(unsigned b, Allocator allocator = Allocator())
    : super(allocator)
    , m_bits(b)
  { }

  vector_dyn(vector_dyn&& rhs)
    : super(std::move(rhs))
    , m_bits(rhs.bits())
  { }

  vector_dyn(const vector_dyn& rhs)
    : super(rhs)
    , m_bits(rhs.bits())
  { }

  inline unsigned bits() const { return m_bits; }

  vector_dyn& operator=(const vector_dyn& rhs) {
    if(bits() != rhs.bits())
      throw std::invalid_argument("Bit length of compacted vector differ");
    static_cast<super*>(this)->operator=(rhs);
    return *this;
  }

  vector_dyn& operator=(vector_dyn&& rhs) {
    if(bits() != rhs.bits())
      throw std::invalid_argument("Bit length of compacted vector differ");
    static_cast<super*>(this)->operator=(std::move(rhs));
    return *this;
  }
};

} // namespace vector_imp

template<typename IDX, unsigned BITS = 0, typename W = uint64_t, typename Allocator = std::allocator<W>>
class vector
  : public vector_imp::vector<vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val, false>
{
  typedef vector_imp::vector<vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val, false> super;

public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  vector(size_t s, Allocator allocator = Allocator())
    : super(BITS, s, allocator)
  { }
  vector(Allocator allocator = Allocator())
    : super(allocator)
  { }

  static constexpr unsigned bits() { return BITS; }
};

template<typename IDX, typename W, typename Allocator>
class vector<IDX, 0, W, Allocator>
  : public vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val, false>
{
  typedef vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val, false> super;

public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  vector(unsigned b, size_t s, Allocator allocator = Allocator())
    : super(b, s, allocator)
  {
    if(b > bitsof<W>::val)
      throw std::out_of_range("Number of bits larger than usable bits");
  }
  vector(unsigned b, Allocator allocator = Allocator())
    : super(b, allocator)
  { }
};

template<typename IDX, unsigned BITS = 0, typename W = uint64_t, typename Allocator = std::allocator<W>>
class ts_vector
  : public vector_imp::vector<ts_vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val, true>
{
  typedef vector_imp::vector<ts_vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val, true> super;

public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  ts_vector(size_t s, Allocator allocator = Allocator())
    : super(BITS, s, allocator)
  { }
  ts_vector(Allocator allocator = Allocator())
    : super(allocator)
  { }

  static constexpr unsigned bits() { return BITS; }
};


template<typename IDX, typename W, typename Allocator>
class ts_vector<IDX, 0, W, Allocator>
  : public vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val, true>
{
  typedef vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val, true> super;
public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  ts_vector(unsigned b, size_t s, Allocator allocator = Allocator())
    : super(b, s, allocator)
  {
    if(b > bitsof<W>::val)
      throw std::out_of_range("Number of bits larger than usable bits");
  }
  ts_vector(unsigned b, Allocator allocator = Allocator())
    : super(b, allocator)
  { }
};

template<typename IDX, unsigned BITS = 0, typename W = uint64_t, typename Allocator = std::allocator<W>>
class cas_vector
  : public vector_imp::vector<cas_vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val-1, true>
{
  typedef vector_imp::vector<cas_vector<IDX, BITS, W, Allocator>, IDX, BITS, W, Allocator, bitsof<W>::val-1, true> super;

public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  cas_vector(size_t s, Allocator allocator = Allocator())
    : super(BITS, s, allocator)
  { }
  cas_vector(Allocator allocator = Allocator())
    : super(allocator)
  { }

  static constexpr unsigned bits() { return BITS; }
};

template<typename IDX, typename W, typename Allocator>
class cas_vector<IDX, 0, W, Allocator>
  : public vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val - 1, true>
{
  typedef vector_imp::vector_dyn<IDX, W, Allocator, bitsof<W>::val - 1, true> super;
public:
  typedef typename super::iterator              iterator;
  typedef typename super::const_iterator        const_iterator;
  typedef IDX                                   value_type;
  typedef Allocator                             allocator_type;
  typedef typename iterator::lhs_setter_type    reference;
  typedef const reference                       const_reference;
  typedef iterator                              pointer;
  typedef const_iterator                        const_pointer;
  typedef std::reverse_iterator<iterator>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef ptrdiff_t                             difference_type;
  typedef size_t                                size_type;
  typedef W                                     word_type;

  cas_vector(unsigned b, size_t s, Allocator allocator = Allocator())
    : super(b, s, allocator)
  {
    if(b > bitsof<W>::val - 1)
      throw std::out_of_range("Number of bits larger than usable bits");
  }
  cas_vector(unsigned b, Allocator allocator = Allocator())
    : super(b, allocator)
  { }
};

} // namespace compact

#endif /* __COMPACT_VECTOR_H__ */
