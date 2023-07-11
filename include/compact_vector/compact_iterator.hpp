#ifndef __COMPACT_ITERATOR_H__
#define __COMPACT_ITERATOR_H__

#if __cplusplus < 201103L
#error C++ versions less than C++11 are not supported.
#endif

#include <iterator>
#include <memory>
#include <type_traits>
#include <cstddef>
#include <climits>
#include <ostream>

#if __cplusplus >= 202002L
#include <compare>
#endif

#include "const_iterator_traits.hpp"
#include "parallel_iterator_traits.hpp"
#include "prefetch_iterator_traits.hpp"


namespace compact {
// Number of bits in type t
template<typename T>
struct bitsof {
  static constexpr size_t val = sizeof(T) * CHAR_BIT;
};

// Compact iterator definition. A 'iterator<int> p' would
// behave identically to an 'int*', except that the underlying storage
// is bit-packed. The actual number of bits used by each element is
// specified at construction.
//
// * IDX is the type of the integral type, i.e., the type of the
//   integer pointed to, as seen from the outside. It behaves like a
//   pointer to IDX.
//
// * BITS is the number of bits used for each word. If BITS==0, the
// * class is specialized to use a number of bits defined at runtime
// * instead of compile time.
//
// * W is the word type used internally. We must have sizeof(IDX) <=
//   sizeof(W).
//
// * TS is true for Thread Safe operations. It is only concerned with
//   basic thread safety: if p1 != p2, than manipulating *p1 and *p2
//   in 2 different thread is safe.
//
// * UB is the number of Used Bits in each word. We must have UB <=
//   bitsof<W>::val. Normally UB == bitsof<W>::val, but for some applications,
//   saving a few bits in each word can be useful (for example to
//   provide the compare and swap operation CAS).
template<typename IDX, unsigned BITS = 0, typename W = uint64_t,
         unsigned UB = bitsof<W>::val>
class const_iterator;
template<typename IDX, unsigned BITS = 0, typename W = uint64_t,
         bool TS = false, unsigned UB = bitsof<W>::val>
class iterator;

namespace iterator_imp {
constexpr bool divides(unsigned a, unsigned b) { return b % a == 0; }

template<typename W, bool TS>
struct mask_store { }; // Store bits within a word masked

// Getter / setter for BITS > 0 (number of bits known at compile time)
template<typename IDX, unsigned BITS, typename W, unsigned UB>
struct gs {
  // Get a value at position p, offset o and b bits. There is no
  // multi-thread guarantee of consistency if another thread writes at
  // the same location. Especially when the value straddles two words,
  // the value read may not be valid because of race conditions. See
  // fetch for that.
  static IDX get(const W* p, unsigned o) {
    static constexpr size_t Wbits  = bitsof<W>::val;
    IDX res;

    constexpr W mask = ~(W)0 >> (Wbits - BITS);
    if(UB == Wbits) {
      res = (*p >> o) & mask;
    } else {
      constexpr W hibit_mask = ~(W)0 >> (Wbits - UB);
      res = ((*p & hibit_mask) >> o) & mask;
    }

    if(!divides(BITS, UB) && o + BITS > UB) {
      const unsigned over  = o + BITS - UB;
      const uint64_t mask  = ~(W)0 >> (Wbits - over);
      res                 |= (*(p + 1) & mask) << (BITS - over);
    }
    if(std::is_signed<IDX>::value && (res & ((IDX)1 << (BITS - 1))))
      res |= ~static_cast<typename std::make_unsigned<IDX>::type>(0) << BITS;

    return res;
  }
  static inline IDX get(const W* p, unsigned b, unsigned o) { return get(p, o); }


  // Set a value at position p, offset o. TS set to true makes it safe
  // for multiple threads to write at different locations (same or
  // adjacent p and different o). It is still unsafe to use set with
  // multi-threads with same p and o. See push.
  template<bool TS>
  static void set(IDX x, W* p, unsigned o) {
    static constexpr size_t Wbits  = bitsof<W>::val;
    static constexpr W      ubmask = ~(W)0 >> (Wbits - UB);
    const W                 y      = x;
    W                       mask   = ((~(W)0 >> (Wbits - BITS)) << o) & ubmask;
    mask_store<W, TS>::store(p, mask, y << o);
    if(!divides(BITS, UB) && o + BITS > UB) {
      unsigned over = o + BITS - UB;
      mask              = ~(W)0 >> (Wbits - over);
      mask_store<W, TS>::store(p + 1, mask, y >> (BITS - over));
    }
  }
  template<bool TS>
  static inline IDX set(IDX x, W* p, unsigned b, unsigned o) { return set(x, p, o); }

  // Do a CAS at position p, offset o and number of bits b. Expect
  // value exp and set value x. It takes care of the tricky case when
  // the value pointed by (p, o, b) straddles 2 words. Then it require
  // 2 CAS and it is technically not lock free anymore: if the current
  // thread dies after setting the MSB to 1 during the first CAS, then
  // the location is "dead" and other threads may be prevented from
  // making progress.
  static bool cas(const IDX x, const IDX exp, W* p, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The CAS operation is valid for a cas_vector (used bits less than bits in word)");
    static constexpr size_t Wbits  = bitsof<W>::val;
    static constexpr W      ubmask = ~(W)0 >> (Wbits - UB);
    static_assert(UB < Wbits, "Used bits must strictly less than bit size of W for CAS.");
    if(divides(BITS, UB) || o + BITS <= UB) {
      const W mask  = (~(W)0 >> (Wbits - BITS)) << o & ubmask;
      return mask_store<W, true>::cas(p, mask, (W)x << o, (W)exp << o);
    }

    // o + BITS > UB. Needs to do a CAS with MSB set to 1, expecting MSB at
    // 0. If failure, then return failure. If success, cas rest of value
    // in next word, then set MSB back to 0.
    static constexpr W msb = (W)1 << (Wbits - 1);
    W mask = (~(W)0 >> (Wbits - BITS)) << o;
    if(!mask_store<W, true>::cas(p, mask, msb | ((W)x << o), ~msb & ((W)exp << o)))
      return false;
    const unsigned over = o + BITS - UB;
    mask                = ~(W)0 >> (Wbits - over);
    const bool     res  = mask_store<W, true>::cas(p + 1, mask, (W)x >> (BITS - over), (W)exp >> (BITS - over));
    mask_store<W, true>::store(p, msb, 0);
    return res;
  }
  static inline bool cas(const IDX x, const IDX exp, W* p, unsigned b, unsigned o) {
    return cas(x, exp, p, o);
  }

  // Fetch a value at position p, offset o and number of bits b. This is
  // used when multiple threads may update the same position (p,o,b) with
  // cas operations. In the case where the value straddles two words,
  // then a CAS operation set the MSB to 1 (to prevent any other thread
  // from changing the value), then reads the second words, finally sets
  // the MSB back to 0 with a CAS operation.
  //
  // Result returned in res. Returns true if fetch is successfull
  // (either value contained within a word, or CAS operations
  // succeeded). Otherwise, it returns false.
  static bool try_fetch(IDX& res, W* p, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The fetch operation is valid for cas_vector (used bits less than bits in word");
    static constexpr size_t Wbits     = bitsof<W>::val;
    static constexpr W      msb       = (W)1 << (Wbits - 1);
    const bool              need_lock = !(divides(BITS, UB) || o + BITS <= UB);

    if(need_lock) {
      if(!mask_store<W, true>::cas(p, msb, msb, 0))
        return false;
    }

    res = get(p, BITS, o);

    if(need_lock)
      mask_store<W, true>::store(p, msb, 0);
    return true;
  }

  static inline IDX fetch(W* p, unsigned o) {
    IDX res;
    while(!try_fetch(res, p, o)) { }
    return res;
  }
  static inline IDX fetch(W* p, unsigned b, unsigned o) { return fetch(p, o); }

  // Push value y at position p, offset o and number of bits b. This
  // can be used when multiple threads may update the same position
  // (p,o,b) with cas operations. See fetch for details
  static bool try_push(IDX y, W* p, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The push operation is valid for cas_vector (used bits less than bits in word");
    static constexpr size_t Wbits     = bitsof<W>::val;
    static constexpr W      msb       = (W)1 << (Wbits - 1);
    const bool              need_lock = !(divides(BITS, UB) || o + BITS <= UB);

    if(need_lock) {
      if(!mask_store<W, true>::cas(p, msb, msb, 0))
        return false;
    }

    set<true>(y, p, o);

    if(need_lock)
      mask_store<W, true>::store(p, msb, 0);
    return true;
  }
  static inline void push(IDX y, W* p, unsigned o) {
    while(!try_push(y, p, o)) { }
  }
};

// gs implementation for number of bits known at runtime (BITS == 0).
//
// XXX: too much code duplication with non-specialized version of gs. Improve!
template<typename IDX, typename W, unsigned UB>
struct gs<IDX, 0, W, UB> {
  static IDX get(const W* p, unsigned b, unsigned o) {
    static constexpr size_t Wbits  = bitsof<W>::val;
    static constexpr W      ubmask = ~(W)0 >> (Wbits - UB);
    W                       mask   = ((~(W)0 >> (Wbits - b)) << o) & ubmask;
    IDX                     res    = (*p & mask) >> o;
    if(o + b > UB) {
      const unsigned over  = o + b - UB;
      mask                     = ~(W)0 >> (Wbits - over);
      res                     |= (*(p + 1) & mask) << (b - over);
    }
    if(std::is_signed<IDX>::value && res & ((IDX)1 << (b - 1)))
      res |= ~(IDX)0 << b;

    return res;
  }

  template<bool TS>
  static void set(IDX x, W* p, unsigned b, unsigned o) {
    static constexpr size_t Wbits  = bitsof<W>::val;
    static constexpr W      ubmask = ~(W)0 >> (Wbits - UB);
    const W                 y      = x;
    W                       mask   = ((~(W)0 >> (Wbits - b)) << o) & ubmask;
    mask_store<W, TS>::store(p, mask, y << o);
    if(o + b > UB) {
      unsigned over = o + b - UB;
      mask              = ~(W)0 >> (Wbits - over);
      mask_store<W, TS>::store(p + 1, mask, y >> (b - over));
    }
  }

  // Do a CAS at position p, offset o and number of bits b. Expect value
  // exp and set value x. It takes care of the tricky case when the
  // value pointed by (p, o, b) straddles 2 words. Then it require 2 CAS
  // and it is technically not lock free anymore: if the current thread
  // dies after setting the MSB to 1 during the first CAS, then the
  // location is "dead" and other threads maybe prevented from making
  // progress.
  static bool cas(const IDX x, const IDX exp, W* p, unsigned b, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The CAS operation is valid for a cas_vector (used bits less than bits in word)");
    static constexpr size_t Wbits  = bitsof<W>::val;
    static constexpr W      ubmask = ~(W)0 >> (Wbits - UB);
    static_assert(UB < Wbits, "Used bits must strictly less than bit size of W for CAS.");
    if(o + b <= UB) {
      const W mask  = (~(W)0 >> (Wbits - b)) << o & ubmask;
      return mask_store<W, true>::cas(p, mask, (W)x << o, (W)exp << o);
    }

    // o + b > UB. Needs to do a CAS with MSB set to 1, expecting MSB at
    // 0. If failure, then return failure. If success, cas rest of value
    // in next word, then set MSB back to 0.
    static constexpr W msb = (W)1 << (Wbits - 1);
    W mask = (~(W)0 >> (Wbits - b)) << o;
    if(!mask_store<W, true>::cas(p, mask, msb | ((W)x << o), ~msb & ((W)exp << o)))
      return false;
    const unsigned over = o + b - UB;
    mask                    = ~(W)0 >> (Wbits - over);
    const bool         res  = mask_store<W, true>::cas(p + 1, mask, (W)x >> (b - over), (W)exp >> (b - over));
    mask_store<W, true>::store(p, msb, 0);
    return res;
  }

  // Fetch a value at position p, offset o and number of bits b. This is
  // used when multiple thread may update the same position (p,o,b) with
  // cas operations. In the case where the value straddles two words,
  // then a CAS operation set the MSB to 1 (to prevent any other thread
  // from changing the value), then reads the second words, finally sets
  // the MSB back to 0 with a CAS operation.
  //
  // Result returned in res. Returns true if fetch is successfull
  // (either value contained within a word, or CAS operations
  // succeeded). Otherwise, it returns false.
  static bool try_fetch(IDX& res, W* p, unsigned b, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The fetch operation is valid for cas_vector (used bits less than bits in word");

    static constexpr size_t Wbits     = bitsof<W>::val;
    static constexpr W      msb       = (W)1 << (Wbits - 1);
    const bool              need_lock = !(divides(b, UB) || o + b <= UB);

    if(need_lock) {
      if(!mask_store<W, true>::cas(p, msb, msb, 0))
        return false;
    }

    res = get(p, b, o);

    if(need_lock)
      mask_store<W, true>::store(p, msb, 0);
    return true;
  }

  static inline IDX fetch(W* p, unsigned b, unsigned o) {
    IDX res;
    while (!try_fetch(res, p, b, o)) { }
    return res;
  }

  // Push value y at position p, offset o and number of bits b. This
  // can be used when multiple threads may update the same position
  // (p,o,b) with cas operations. See fetch for details
  static bool try_push(IDX y, W* p, unsigned b, unsigned o) {
    static_assert(UB < bitsof<W>::val, "The push operation is valid for cas_vector (used bits less than bits in word");
    static constexpr size_t Wbits     = bitsof<W>::val;
    static constexpr W      msb       = (W)1 << (Wbits - 1);
    const bool              need_lock = !(divides(b, UB) || o + b <= UB);

    if(need_lock) {
      if(!mask_store<W, true>::cas(p, msb, msb, 0))
        return false;
    }

    set<true>(y, p, b, o);

    if(need_lock)
      mask_store<W, true>::store(p, msb, 0);
    return true;
  }
  static inline void push(IDX y, W* p, unsigned b, unsigned o) {
    while(!try_push(y, p, b, o)) { }
  }
};

// Helper struct to help choose between get/fetch and set/push. Use
// fetch and push when the number of used bits is less than the number
// of bits in the word. (e.g., cas_vector iterator).
struct gf_sp_helper {
  template<typename IDX, unsigned BITS, typename W, unsigned UB, typename... Args>
  static inline
  typename std::enable_if<UB == bitsof<W>::val, IDX>::type
  getfetch(const W* p, Args&&... args) {
    return gs<IDX, BITS, W, UB>::get(p, std::forward<Args>(args)...);
  }

  template<typename IDX, unsigned BITS, typename W, unsigned UB, typename... Args>
  static inline
  typename std::enable_if<UB != bitsof<W>::val, IDX>::type
  getfetch(const W* p, Args&&... args) {
    return gs<IDX, BITS, W, UB>::fetch(const_cast<W*>(p), std::forward<Args>(args)...);
  }

  template<typename IDX, unsigned BITS, typename W, unsigned UB, typename... Args>
  static inline
  typename std::enable_if<UB == bitsof<W>::val, void>::type
  setpush(IDX y, W* p, Args&&... args) {
    gs<IDX, BITS, W, UB>::template set<true>(y, p, std::forward<Args>(args)...);
  }

  template<typename IDX, unsigned BITS, typename W, unsigned UB, typename... Args>
  static inline
  typename std::enable_if<UB != bitsof<W>::val, void>::type
  setpush(IDX y, W* p, Args&&... args) {
    gs<IDX, BITS, W, UB>::push(y, p, std::forward<Args>(args)...);
  }
};

// Mask store, depending on the thread safety guarantee
template<typename W>
struct mask_store<W, false> {
  static inline void store(W* p, W mask, W val) {
    *p = (*p & ~mask) | (val & mask);
  }
};

template<typename W>
struct mask_store<W, true> {
  static void store(W* p, W mask, W val) {
    W cval = *p, oval;
    do {
      W nval = (cval & ~mask) | (val & mask);
      oval = cval;
      cval = __sync_val_compare_and_swap(p, oval, nval);
    } while(cval != oval);
  }

  // Do a CAS at p and val, only in the bits covered by mask. It
  // retries while bits outside of mask change but those inside the
  // mask are equal to the expected value exp.
  static bool cas(W* p, W mask, W val, W exp) {
    W cval = *p, oval;
    val &= mask;
    exp &= mask;
    if(val == exp)
      return (cval & mask) == exp;
    while((cval & mask) == exp) {
      W nval = (cval & ~mask) | val;
      oval = cval;
      cval = __sync_val_compare_and_swap(p, oval, nval);
      if(cval == oval)
        return true;
    }
    return false;
  }
};

// Replacement for the now deprecated (C++17) std::iterator
template<typename T>
struct iterator_types {
  typedef T                               value_type;
  typedef ptrdiff_t                       difference_type;
  typedef T*                              pointer;
  typedef T&                              reference;
  typedef std::random_access_iterator_tag iterator_category;
};

// Base class for the iterators
template<class Derived, typename IDX, unsigned BITS, typename W, unsigned UB>
class common {
public:
  std::ostream& print(std::ostream& os) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return os << '<' << (void*)self.m_ptr << '+' << self.m_offset << ',' << self.bits() << '>';
  }

protected:
  static constexpr unsigned Wbits = bitsof<W>::val;
  // UB is the number of bits actually used in a word.
  static_assert(UB <= Wbits,
                "Number of used bits must be less than number of bits in a word");
  static_assert(sizeof(IDX) <= sizeof(W),
                "The size of integral type IDX must be less than the word type W");

public:
  typedef typename iterator_types<IDX>::difference_type difference_type;
  static constexpr unsigned used_bits = UB;

  Derived& operator=(const Derived& rhs) {
    Derived& self = *static_cast<Derived*>(this);
    self.m_ptr    = rhs.m_ptr;
    self.m_offset = rhs.m_offset;
    self.m_bits(rhs.bits());
    return self;
  }

  Derived& operator=(std::nullptr_t p) {
    Derived& self = *static_cast<Derived*>(this);
    self.m_ptr    = nullptr;
    self.m_offset = 0;
  }

  IDX operator*() const {
    const Derived& self = *static_cast<const Derived*>(this);
    return gf_sp_helper::template getfetch<IDX, BITS, W, UB>(self.m_ptr, self.bits(), self.m_offset);
  }

#if __cplusplus >= 202002L
  // With C++20, define the spaceship operator and the rest is inferred
  auto operator<=>(const common& rhs) const {
    const Derived& self = *static_cast<const Derived*>(this);
    const Derived& other = static_cast<const Derived&>(rhs);
    const auto ptr_cmp = self.m_ptr <=> other.m_ptr;
    if(ptr_cmp != 0)
      return ptr_cmp;
    return self.m_offset <=> other.m_offset;
  }
  bool operator==(const common& rhs) const {
    return (*this <=> rhs) == 0;
  }
  bool operator==(std::nullptr_t p) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return self.m_ptr == nullptr && self.m_offset == 0;
  }
#else
  // Previous version of C++ require explicit definition of every operators
  bool operator==(const Derived& rhs) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return self.m_ptr == rhs.m_ptr && self.m_offset == rhs.m_offset;
  }
  bool operator!=(const Derived& rhs) const {
    return !(*this == rhs);
  }

  bool operator==(std::nullptr_t p) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return self.m_ptr == nullptr && self.m_offset == 0;
  }
  bool operator!=(std::nullptr_t p) const {
    return !(*this == nullptr);
  }

  bool operator<(const Derived& rhs) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return self.m_ptr < rhs.m_ptr || (self.m_ptr == rhs.m_ptr && self.m_offset < rhs.m_offset);
  }
  bool operator>(const Derived& rhs) const {
    const Derived& self = *static_cast<const Derived*>(this);
    return self.m_ptr > rhs.m_ptr || (self.m_ptr == rhs.m_ptr && self.m_offset > rhs.m_offset);
  }
  bool operator>=(const Derived& rhs) const {
    return !(*this < rhs);
  }
  bool operator<=(const Derived& rhs) const {
    return !(*this > rhs);
  }
#endif

  Derived& operator++() {
    Derived& self = *static_cast<Derived*>(this);
    self.m_offset += self.bits();
    if(self.m_offset >= UB) {
      ++self.m_ptr;
      self.m_offset -= UB;
    }
    return self;
  }
  Derived operator++(int) {
    Derived res(*static_cast<Derived*>(this));
    ++*this;
    return res;
  }

  Derived& operator--() {
    Derived& self = *static_cast<Derived*>(this);
    if(self.bits() > self.m_offset) {
      --self.m_ptr;
      self.m_offset += UB;
    }
    self.m_offset -= self.bits();
    return self;
  }
  Derived operator--(int) {
    Derived res(*static_cast<Derived*>(this));
    --*this;
    return res;
  }

  Derived& operator+=(difference_type n) {
    Derived&     self    = *static_cast<Derived*>(this);
    if(n < 0) {
      self -= -n;
      return self;
    }

    const size_t nbbits  = self.bits() * n;
    self.m_ptr          += nbbits / UB;
    self.m_offset       += nbbits % UB;
    if(self.m_offset >= UB) {
      ++self.m_ptr;
      self.m_offset -= UB;
    }
    return self;
  }

  Derived operator+(difference_type n) const {
    Derived res(*static_cast<const Derived*>(this));
    return res += n;
  }

  Derived& operator-=(difference_type n) {
    Derived&           self     = *static_cast<Derived*>(this);
    if(n < 0) {
      self += -n;
      return self;
    }

    const size_t      nbbits    = self.bits() * n;
    self.m_ptr                 -= nbbits / UB;
    const unsigned ooffset  = nbbits % UB;
    if(ooffset > self.m_offset) {
      --self.m_ptr;
      self.m_offset += UB;
    }
    self.m_offset -= ooffset;
    return self;
  }

  Derived operator-(difference_type n) const {
    Derived res(*static_cast<const Derived*>(this));
    return res -= n;
  }

  template<unsigned BB, typename DD>
  difference_type operator-(const common<DD, IDX, BB, W, UB>& rhs_) const {
    const Derived& self  = *static_cast<const Derived*>(this);
    const DD&      rhs   = *static_cast<const DD*>(&rhs_);
    ptrdiff_t      wdiff = (self.m_ptr - rhs.m_ptr) * UB;
    if(self.m_offset < rhs.m_offset)
      wdiff += (ptrdiff_t)((UB + self.m_offset) - rhs.m_offset) - (ptrdiff_t)UB;
    else
      wdiff += self.m_offset - rhs.m_offset;
    return wdiff / self.bits();
  }

  IDX operator[](const difference_type n) const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return *(self + n);
  }

  // Extra methods which are not part of an iterator interface

  const W* get_ptr() const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return self.m_ptr;
  }
  unsigned get_offset() const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return self.m_offset;
  }
  unsigned get_bits() const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return self.bits();
  }

  // Get some number of bits
  W get_bits(unsigned bits) const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return gs<W, BITS, W, UB>::get(self.m_ptr, bits, self.m_offset);
  }

  W get_bits(unsigned bits, unsigned offset) const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return gs<W, BITS, W, UB>::get(self.m_ptr, bits, offset);
  }

  template<bool TS = false>
  void set_bits(W x, unsigned bits) {
    Derived& self  = *static_cast<Derived*>(this);
    gs<W, BITS, W, UB>::template set<TS>(x, self.m_ptr, bits, self.m_offset);
  }

  // Get, i.e., don't use fetch
  IDX get() const {
    const Derived& self  = *static_cast<const Derived*>(this);
    return gs<W, BITS, W, UB>::get(self.m_ptr, self.bits(), self.m_offset);
  }
};

template<typename W, int I = sizeof(W)>
struct swap_word_mask {
  static constexpr W value = swap_word_mask<W, I / 2>::value << (4 * I) | swap_word_mask<W, I / 2>::value;
};
template<typename W>
struct swap_word_mask<W, 1> {
  static constexpr W value = 0x55;
};

template<typename W>
inline W swap_word(W w) {
  return ((w & swap_word_mask<W>::value) << 1) | ((w & (swap_word_mask<W>::value << 1)) >> 1);
}
template<typename W>
inline bool compare_swap_words(W w1, W w2) {
  w1 = swap_word(w1);
  w2 = swap_word(w2);
  W bmask = w1 ^ w2;
  bmask &= -bmask;
  return (w1 & bmask) == 0;
}

// Precompute (expensive) division by number of bits. The static
// arrays contain X/k (word_idx) and k*(X/k)
// (word_bits) for k in [0, X].
//
// ** This code is kind of sick!

//helper template, just converts its variadic arguments to array initializer list
template<size_t... values> struct size_t_ary {static const size_t value[sizeof...(values)];};
template<size_t... values> const size_t size_t_ary<values...>::value[] = {values...};

template<size_t X, size_t k = X, size_t... values>
struct word_idx : word_idx <X, k-1, X / k, values...> {};
template<size_t X, size_t... values>
struct word_idx<X, 0, values...> : size_t_ary<(size_t)0, values...> {};

template<size_t X, size_t k = X, size_t... values>
struct word_bits : word_bits <X, k-1, k * (X / k), values...> {};
template<size_t X, size_t... values>
struct word_bits<X, 0, values...> : size_t_ary<(size_t)0, values...> {};

template<typename Iterator>
bool lexicographical_compare_n(Iterator first1, const size_t len1,
                               Iterator first2, const size_t len2) {
  static constexpr unsigned UB = Iterator::used_bits;

  const auto bits            = first1.get_bits();
  auto       left            = std::min(len1, len2) * bits;
  const decltype(len1) Widx  = word_idx<UB>::value[bits];
  const decltype(len1) Wbits = word_bits<UB>::value[bits];

  for( ; left > Wbits; left -= Wbits, first1 += Widx, first2 += Widx) {
    auto w1   = first1.get_bits(Wbits);
    auto w2   = first2.get_bits(Wbits);
    if(w1 != w2) return compare_swap_words(w1, w2);
  }
  if(left > 0) {
    auto w1   = first1.get_bits(left);
    auto w2   = first2.get_bits(left);
    if(w1 != w2) return compare_swap_words(w1, w2);
  }

  return len1 < len2;
}

template<typename D, typename I, unsigned B, typename W, unsigned U>
inline bool operator==(std::nullptr_t lfs, const common<D, I, B, W, U>& rhs) {
  return rhs.operator==(nullptr);
}

template<typename D, typename I, unsigned B, typename W, unsigned U>
inline D operator+(typename common<D, I, B, W, U>::difference_type lhs, const common<D, I, B, W, U>& rhs) {
  return rhs + lhs;
}

template<typename D, typename I, unsigned B, typename W, unsigned U>
inline std::ostream& operator<<(std::ostream& os, const common<D, I, B, W, U>& rhs) {
  return rhs.print(os);
}

template<class Derived,
         typename IDX, unsigned BITS, typename W, bool TS = false, unsigned UB = bitsof<W>::val>
class lhs_setter_common {
protected:
  W*       m_ptr;
  unsigned m_offset;

public:
  typedef compact::iterator<IDX, BITS, W, TS, UB> iterator;
  lhs_setter_common(W* p, unsigned o) : m_ptr(p), m_offset(o) { }
  operator IDX() const {
    const Derived& self = *static_cast<const Derived*>(this);
    return gf_sp_helper::template getfetch<IDX, BITS, W, UB>(m_ptr, self.bits(), m_offset);
  }

  iterator operator&() {
    Derived& self = *static_cast<Derived*>(this);
    return iterator(m_ptr, self.bits(), m_offset);
  }
  inline bool cas(const IDX x, const IDX exp) {
    Derived& self = *static_cast<Derived*>(this);
    return gs<IDX, BITS, W, UB>::cas(x, exp, m_ptr, self.bits(), m_offset);
  }

  // Get, i.e., don't use fetch. Less thread safety for cas_vector
  inline IDX get() const {
    const Derived& self = *static_cast<const Derived*>(this);
    return gs<IDX, BITS, W, UB>::get(m_ptr, self.bits(), m_offset);
  }
};

template<typename IDX, unsigned BITS, typename W, bool TS, unsigned UB>
class lhs_setter;

template<typename IDX, typename W, bool TS, unsigned UB>
class lhs_setter<IDX, 0, W, TS, UB>
  : public lhs_setter_common<lhs_setter<IDX, 0, W, TS, UB>, IDX, 0, W, TS, UB>
{
  typedef lhs_setter_common<lhs_setter<IDX, 0, W, TS, UB>, IDX, 0, W, TS, UB> super;
  unsigned m_bits;                // number of bits in an integral type

public:
  lhs_setter(W* p, int b, int o) : super(p, o), m_bits(b) { }
  lhs_setter& operator=(const IDX x) {
    gf_sp_helper::template setpush<IDX, 0, W, UB>(x, super::m_ptr, m_bits, super::m_offset);
    return *this;
  }
  lhs_setter& operator=(const lhs_setter& rhs) {
    this->operator=((IDX)rhs);
    return *this;
  }

  // Use set, i.e., not push
  lhs_setter& set(const IDX x) {
    gs<IDX, 0, W, UB>::template set<false>(x, super::m_ptr, bits(), super::m_offset);
    return *this;
  }
  // Use set, i.e., not push, but at least thread safe version
  lhs_setter& ts_set(const IDX x) {
    gs<IDX, 0, W, UB>::template set<true>(x, super::m_ptr, bits(), super::m_offset);
    return *this;
  }

  unsigned bits() const { return m_bits; }
};

template<typename IDX, unsigned BITS, typename W, bool TS, unsigned UB>
class lhs_setter
  : public lhs_setter_common<lhs_setter<IDX, BITS, W, TS, UB>, IDX, BITS, W, TS, UB>
{
  typedef lhs_setter_common<lhs_setter<IDX, BITS, W, TS, UB>, IDX, BITS, W, TS, UB> super;

public:
  lhs_setter(W* p, int o) : super(p, o) { }
  lhs_setter(W* p, unsigned bits, int o) : super(p, o) { }
  lhs_setter& operator=(const IDX x) {
    gf_sp_helper::template setpush<IDX, BITS, W, UB>(x, super::m_ptr, super::m_offset);
    return *this;
  }
  lhs_setter& operator=(const lhs_setter& rhs) {
    this->operator=((IDX)rhs);
    return *this;
  }

  // Use set, i.e., not push
  lhs_setter& set(const IDX x) {
    gs<IDX, 0, W, UB>::template set<false>(x, super::m_ptr, bits(), super::m_offset);
    return *this;
  }
  // Use set, i.e., not push, but at least thread safe version
  lhs_setter& ts_set(const IDX x) {
    gs<IDX, 0, W, UB>::template set<true>(x, super::m_ptr, bits(), super::m_offset);
    return *this;
  }

  constexpr unsigned bits() const { return BITS; }
};

template<typename I, unsigned BITS, typename W, bool TS, unsigned UB>
void swap(lhs_setter<I, BITS, W, TS, UB> x, lhs_setter<I, BITS, W, TS, UB> y) {
  I t = x;
  x = (I)y;
  y = t;
}

} // namespace iterator_imp

// Specialization with BITS=0 (dynamic/runtime number of bits used)
template<typename IDX, typename W, bool TS, unsigned UB>
class iterator<IDX, 0, W, TS, UB> :
    public iterator_imp::iterator_types<IDX>,
    public iterator_imp::common<iterator<IDX, 0, W, TS, UB>, IDX, 0, W, UB>
{
  W*       m_ptr;
  unsigned m_bits;                // number of bits in an integral type
  unsigned m_offset;

  friend class iterator<IDX, 0, W, !TS, UB>;
  friend class const_iterator<IDX, 0, W, UB>;
  friend class iterator_imp::common<iterator<IDX, 0, W, TS, UB>, IDX, 0, W, UB>;
  friend class iterator_imp::common<const_iterator<IDX, 0, W, UB>, IDX, 0, W, UB>;

  typedef iterator_imp::iterator_types<IDX> super;
public:
  typedef typename super::value_type                  value_type;
  typedef typename super::difference_type             difference_type;
  typedef IDX                                         idx_type;
  typedef W                                           word_type;
  typedef iterator_imp::lhs_setter<IDX, 0, W, TS, UB> lhs_setter_type;

  iterator() = default;
  iterator(W* p, unsigned b, unsigned o)
    : m_ptr(p), m_bits(b), m_offset(o) { }
  template<unsigned BITS, bool TTS>
  iterator(const iterator<IDX, BITS, W, TTS>& rhs)
    : m_ptr(rhs.m_ptr), m_bits(rhs.m_bits), m_offset(rhs.m_offset) { }
  iterator(std::nullptr_t)
    : m_ptr(nullptr), m_bits(0), m_offset(0) { }

  lhs_setter_type operator*() const { return lhs_setter_type(m_ptr, m_bits, m_offset); }
  lhs_setter_type operator[](const difference_type n) const {
    return *(*this + n);
  }

  // CAS val. Does NOT return existing value at pointer. Return true
  // if successful.
  inline bool cas(const IDX x, const IDX exp) {
    return iterator_imp::gs<IDX, 0, W, UB>::cas(x, exp, m_ptr, m_bits, m_offset);
  }

  unsigned bits() const { return m_bits; }
protected:
  void bits(unsigned b) { m_bits = b; }
};

template<typename IDX, typename W, unsigned UB>
class const_iterator<IDX, 0, W, UB> :
  public iterator_imp::iterator_types<const IDX>,
  public iterator_imp::common<const_iterator<IDX, 0, W, UB>, IDX, 0, W, UB>
{
  const W* m_ptr;
  unsigned m_bits;                // number of bits in an integral type
  unsigned m_offset;

  friend class iterator<IDX, 0, W>;
  friend class iterator_imp::common<iterator<IDX, 0, W, true, UB>, IDX, 0, W, UB>;
  friend class iterator_imp::common<iterator<IDX, 0, W, false, UB>, IDX, 0, W, UB>;
  friend class iterator_imp::common<const_iterator<IDX, 0, W, UB>, IDX, 0, W, UB>;

  typedef iterator_imp::iterator_types<IDX> super;
public:
  typedef typename super::value_type      value_type;
  typedef typename super::difference_type difference_type;
  typedef IDX idx_type;
  typedef W   word_type;


  const_iterator() = default;
  const_iterator(const W* p, unsigned b, unsigned o)
    : m_ptr(p), m_bits(b), m_offset(o) { }
  const_iterator(const const_iterator& rhs)
    : m_ptr(rhs.m_ptr), m_bits(rhs.m_bits), m_offset(rhs.m_offset) { }
  template<unsigned BITS, bool TS>
  const_iterator(const iterator<IDX, BITS, W, TS>& rhs)
    : m_ptr(rhs.m_ptr), m_bits(rhs.m_bits), m_offset(rhs.m_offset) { }
  const_iterator(std::nullptr_t)
    : m_ptr(nullptr), m_bits(0), m_offset(0) { }

  unsigned bits() const { return m_bits; }
  void bits(unsigned b) { m_bits = b; }
};


// No specialization. Static number of bits used.
template<typename IDX, unsigned BITS, typename W, bool TS, unsigned UB>
class iterator :
    public iterator_imp::iterator_types<IDX>,
    public iterator_imp::common<iterator<IDX, BITS, W, TS, UB>, IDX, BITS, W, UB>
{
  W*       m_ptr;
  unsigned m_offset;

  friend class iterator<IDX, BITS, W, !TS, UB>;
  friend class const_iterator<IDX, BITS, W, UB>;
  friend class iterator_imp::common<iterator<IDX, BITS, W, TS, UB>, IDX, BITS, W, UB>;
  friend class iterator_imp::common<const_iterator<IDX, BITS, W, UB>, IDX, BITS, W, UB>;

  typedef iterator_imp::iterator_types<IDX> super;
public:
  typedef typename super::value_type                  value_type;
  typedef typename super::difference_type             difference_type;
  typedef IDX                                         idx_type;
  typedef W                                           word_type;
  typedef iterator_imp::lhs_setter<IDX, BITS, W, TS, UB> lhs_setter_type;

  iterator() = default;
  iterator(W* p, unsigned o)
    : m_ptr(p), m_offset(o) { }
  iterator(W* p, unsigned b, unsigned o)
    : m_ptr(p), m_offset(o) { } // XXX Should we assert that BITS == b?
  template<bool TTS>
  iterator(const iterator<IDX, BITS, W, TTS>& rhs)
    : m_ptr(rhs.m_ptr), m_offset(rhs.m_offset) { }
  iterator(std::nullptr_t)
    : m_ptr(nullptr), m_offset(0) { }

  lhs_setter_type operator*() { return lhs_setter_type(m_ptr, m_offset); }
  lhs_setter_type operator[](const difference_type n) const {
    return *(*this + n);
  }

  // CAS val. Does NOT return existing value at pointer. Return true
  // if successful.
  inline bool cas(const IDX x, const IDX exp) {
    return iterator_imp::gs<IDX, BITS, W, UB>::cas(x, exp, m_ptr, m_offset);
  }

  constexpr unsigned bits() const { return BITS; }
protected:
  void bits(unsigned b) { } // NOOP
};

template<typename IDX, unsigned BITS, typename W, unsigned UB>
class const_iterator :
  public iterator_imp::iterator_types<const IDX>,
  public iterator_imp::common<const_iterator<IDX, BITS, W, UB>, IDX, BITS, W, UB>
{
  const W* m_ptr;
  unsigned m_offset;

  friend class iterator<IDX, BITS, W>;
  friend class iterator_imp::common<iterator<IDX, BITS, W, true, UB>, IDX, BITS, W, UB>;
  friend class iterator_imp::common<iterator<IDX, BITS, W, false, UB>, IDX, BITS, W, UB>;
  friend class iterator_imp::common<const_iterator<IDX, BITS, W, UB>, IDX, BITS, W, UB>;

  typedef iterator_imp::iterator_types<IDX> super;
public:
  typedef typename super::value_type      value_type;
  typedef typename super::difference_type difference_type;
  typedef IDX idx_type;
  typedef W   word_type;


  const_iterator() = default;
  const_iterator(const W* p, unsigned o)
    : m_ptr(p), m_offset(o) { }
  const_iterator(const W* p, unsigned b, unsigned o)
    : m_ptr(p), m_offset(o) { }
  const_iterator(const const_iterator& rhs)
    : m_ptr(rhs.m_ptr), m_offset(rhs.m_offset) { }
  template<bool TS>
  const_iterator(const iterator<IDX, BITS, W, TS>& rhs)
    : m_ptr(rhs.m_ptr), m_offset(rhs.m_offset) { }
  const_iterator(std::nullptr_t)
    : m_ptr(nullptr), m_offset(0) { }

  constexpr unsigned bits() const { return BITS; }
protected:
  void bits(unsigned b) { } // NOOP
};

template<typename I, unsigned BITS, typename W, bool TS, unsigned UB>
struct const_iterator_traits<iterator<I, BITS, W, TS, UB>> {
  typedef const_iterator<I, BITS, W, UB> type;
};
template<typename I, unsigned BITS, typename W, unsigned UB>
struct const_iterator_traits<const_iterator<I, BITS, W, UB>> {
  typedef const_iterator<I, BITS, W, UB> type;
};

template<typename I, unsigned BITS, typename W, bool TS, unsigned UB>
struct parallel_iterator_traits<iterator<I, BITS, W, TS, UB>> {
  typedef iterator<I, BITS, W, true, UB> type;

  // Does a cas on iterator x. Weak though: val is NOT updated to the
  // current value.
  static inline bool cas(type& x, I& expected, const I& val) {
    return x.cas(val, expected);
  }
};

template<typename I, unsigned BITS, typename W, unsigned UB>
struct parallel_iterator_traits<const_iterator<I, BITS, W, UB>> {
  typedef const_iterator<I, BITS, W> type;
};

template<typename I, unsigned BITS, typename W, bool TS, unsigned UB>
struct prefetch_iterator_traits<iterator<I, BITS, W, TS, UB> > {
  template<int level = 0>
  static void read(const iterator<I, BITS, W, TS, UB>& p) { prefetch_iterator_traits<W*>::template read<level>(p.get_ptr()); }
  template<int level = 0>
  static void write(const iterator<I, BITS, W, TS, UB>& p) { prefetch_iterator_traits<W*>::template write<level>(p.get_ptr()); }

};

template<typename I, unsigned BITS, typename W, unsigned UB>
struct prefetch_iterator_traits<const_iterator<I, BITS, W, UB> > {
  template<int level = 0>
  static void read(const const_iterator<I, BITS, W, UB>& p) { prefetch_iterator_traits<const W*>::template read<level>(p.get_ptr()); }
  template<int level = 0>
  static void write(const const_iterator<I, BITS, W, UB>& p) { prefetch_iterator_traits<const W*>::template write<level>(p.get_ptr()); }

};

} // namespace compact

#endif /* __COMPACT_ITERATOR_H__ */
