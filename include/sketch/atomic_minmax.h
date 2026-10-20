#ifndef SKETCH_ATOMIC_MINMAX_H__
#define SKETCH_ATOMIC_MINMAX_H__
// Lock-free register updates for concurrent sketch adds: *p = min(*p, v) and
// *p = max(*p, v) for unsigned 8- and 64-bit registers. Most adds change
// nothing, so a plain load is checked first (no store, the cache line stays
// shared); an update then uses the hardware's atomic min/max where it has one
// (AArch64 LSE STUMIN/STUMAX, also used by clang's __atomic_fetch_min/max), and
// a compare-and-swap loop otherwise (x86, pre-LSE ARM). Relaxed ordering:
// registers are only read once the adding threads are done.
#include <cstdint>
#include <type_traits>

namespace sketch { inline namespace atomic_minmax {

namespace detail {
template<typename T, bool MIN>
inline void cas_update(T *p, T v) noexcept {
    T cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    while(MIN ? v < cur: v > cur)
        if(__atomic_compare_exchange_n(p, &cur, v, /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
}
} // namespace detail

#if defined(__aarch64__) && defined(__ARM_FEATURE_ATOMICS)
#  define SKETCH_ATOMIC_MINMAX_NATIVE 1
#elif defined(__has_builtin)
#  if __has_builtin(__atomic_fetch_min) && __has_builtin(__atomic_fetch_max)
#    define SKETCH_ATOMIC_MINMAX_BUILTIN 1
#  endif
#endif

// Any unsigned register width; u8 and u64 (HyperLogLog, b-bit MinHash
// registers) take the native instructions below where available.
template<typename T>
inline void atomic_store_min(T *p, T v) noexcept {
    static_assert(std::is_unsigned<T>::value, "unsigned registers");
    if(__atomic_load_n(p, __ATOMIC_RELAXED) <= v) return;
#if defined(SKETCH_ATOMIC_MINMAX_BUILTIN)
    __atomic_fetch_min(p, v, __ATOMIC_RELAXED);
#else
    detail::cas_update<T, true>(p, v);
#endif
}
template<typename T>
inline void atomic_store_max(T *p, T v) noexcept {
    static_assert(std::is_unsigned<T>::value, "unsigned registers");
    if(__atomic_load_n(p, __ATOMIC_RELAXED) >= v) return;
#if defined(SKETCH_ATOMIC_MINMAX_BUILTIN)
    __atomic_fetch_max(p, v, __ATOMIC_RELAXED);
#else
    detail::cas_update<T, false>(p, v);
#endif
}

#if defined(SKETCH_ATOMIC_MINMAX_NATIVE)
inline void atomic_store_min(uint64_t *p, uint64_t v) noexcept {
    if(__atomic_load_n(p, __ATOMIC_RELAXED) > v) __asm__ __volatile__("stumin %x1, %0": "+Q"(*p): "r"(v));
}
inline void atomic_store_max(uint64_t *p, uint64_t v) noexcept {
    if(__atomic_load_n(p, __ATOMIC_RELAXED) < v) __asm__ __volatile__("stumax %x1, %0": "+Q"(*p): "r"(v));
}
inline void atomic_store_min(uint8_t *p, uint8_t v) noexcept {
    if(__atomic_load_n(p, __ATOMIC_RELAXED) > v) __asm__ __volatile__("stuminb %w1, %0": "+Q"(*p): "r"(uint32_t(v)));
}
inline void atomic_store_max(uint8_t *p, uint8_t v) noexcept {
    if(__atomic_load_n(p, __ATOMIC_RELAXED) < v) __asm__ __volatile__("stumaxb %w1, %0": "+Q"(*p): "r"(uint32_t(v)));
}
#endif

}} // namespace sketch::atomic_minmax

#endif /* SKETCH_ATOMIC_MINMAX_H__ */
