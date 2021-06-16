#ifndef __SSE_UTIL_H__
#define __SSE_UTIL_H__
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <cstddef>
#include <cassert>
#include <utility>
#include <new>
#include "hedley.h"

namespace sse {

#ifndef unlikely
#  if __GNUC__  || __clang__ || defined(BUILTIN_EXPECT_AVAILABLE)
#    define unlikely(x) HEDLEY_UNLIKELY(x)
#  else
#    define unlikely(x) (x)
#  endif
#endif

#ifndef likely
#define likely(x) HEDLEY_LIKELY(x)
#endif

// From http://stackoverflow.com/questions/12942548/making-stdvector-allocate-aligned-memory
// Accessed 11/7/16
enum class Alignment : size_t
{
    Normal = sizeof(void*),
    SSE    = 16,
    AVX    = 32,
    KB     = 64,
    KL     = 64,
    AVX512 = 64
};


#ifndef USE_ALIGNED_ALLOC
#  if (__cplusplus >= 201703L && defined(_GLIBCXX_HAVE_ALIGNED_ALLOC))
#    define USE_ALIGNED_ALLOC 1
#  else
#    define USE_ALIGNED_ALLOC 0
#  endif
#endif

namespace detail {
    static inline void* allocate_aligned_memory(const size_t align, size_t size) {
        assert(align >= sizeof(void*));
        assert((align & (align - 1)) == 0); // Assert is power of two

        void *ret;
        return posix_memalign(&ret, align, size) ? nullptr: ret;
    }
}


template <typename T, Alignment Align = Alignment::AVX>
class AlignedAllocator;


template <Alignment Align>
class AlignedAllocator<void, Align>
{
public:
    using pointer = void *;
    using const_pointer = const void *;
    using value_type = void;

    template <class U> struct rebind { using other = AlignedAllocator<U, Align>; };
};


template <typename T, Alignment Align>
class AlignedAllocator
{
public:
    typedef T         value_type;
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    typedef std::true_type propagate_on_container_move_assignment;

    template <class U>
    struct rebind { typedef AlignedAllocator<U, Align> other; };

public:
    AlignedAllocator() noexcept {}
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    static constexpr size_type max_size() {return (size_type(~0) - size_type(Align)) / sizeof(T);}

    pointer address(reference x) const noexcept {
        return std::addressof(x);
    }
    const_pointer address(const_reference x) const noexcept {
        return std::addressof(x);
    }

    pointer allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0)
    {
        pointer ret(reinterpret_cast<pointer>(detail::allocate_aligned_memory(static_cast<size_type>(Align) , n * sizeof(T))));
        if(unlikely(ret == nullptr)) throw std::bad_alloc();
        return ret;
    }

    void deallocate(pointer p, size_type) noexcept {std::free(p);}

    template <class U, class ...Args>
    void construct(U* p, Args&&... args) {
        ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...);
    }

    void destroy(pointer p) { p->~T(); }
};


template <typename T, Alignment Align>
class AlignedAllocator<const T, Align>
{
public:
    typedef T         value_type;
    typedef const T*  pointer;
    typedef const T*  const_pointer;
    typedef const T&  reference;
    typedef const T&  const_reference;
    typedef size_t    size_type;
    typedef std::ptrdiff_t difference_type;

    typedef std::true_type propagate_on_container_move_assignment;

    template <class U>
    struct rebind { typedef AlignedAllocator<U, Align> other; };

public:
    AlignedAllocator() noexcept
    {}

    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept
    {}

    size_type
    max_size() const noexcept
    { return (size_type(~0) - size_type(Align)) / sizeof(T); }

    const_pointer
    address(const_reference x) const noexcept
    { return std::addressof(x); }

    pointer
    allocate(size_type n, typename AlignedAllocator<void, Align>::const_pointer = 0)
    {
        pointer ret(reinterpret_cast<pointer>(detail::allocate_aligned_memory(static_cast<size_type>(Align) , n * sizeof(T))));
        if(unlikely(!ret)) throw std::bad_alloc();
        return ret;
    }

    void
    deallocate(pointer p, size_type) noexcept
    { std::free(p); }

    template <class U, class ...Args>
    void
    construct(U* p, Args&&... args)
    { ::new(reinterpret_cast<void*>(p)) U(std::forward<Args>(args)...); }

    void
    destroy(pointer p) { p->~T(); }
};

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator== (const AlignedAllocator<T,TAlign>&, const AlignedAllocator<U, UAlign>&) noexcept
    { return TAlign == UAlign; }

template <typename T, Alignment TAlign, typename U, Alignment UAlign>
inline bool operator!= (const AlignedAllocator<T,TAlign>&, const AlignedAllocator<U, UAlign>&) noexcept
    { return TAlign != UAlign; }

} // namespace sse

#endif
