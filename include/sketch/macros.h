#pragma once


// Versioning
#define sk__str__(x) #x
#define sk__xstr__(x) sk__str__(x)
#define SKETCH_SHIFT 16
#define SKETCH_MAJOR 0
#define SKETCH_MINOR 7
#define SKETCH_VERSION_INTEGER (SKETCH_MAJOR << SKETCH_SHIFT) | SKETCH_MINOR
#define SKETCH_VERSION SKETCH_MAJOR.SKETCH_MINOR
#define SKETCH_VERSION_STR sk__xstr__(SKETCH_VERSION)


// INLINE
#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif


// OpenMP

#ifdef _OPENMP
#define OMP_PRAGMA(...) _Pragma(__VA_ARGS__)
#define OMP_ONLY(...) __VA_ARGS__
#else
#define OMP_PRAGMA(...)
#define OMP_ONLY(...)
#endif


#if __CUDACC__ || __GNUC__ || __clang__
#  define SK_RESTRICT __restrict__
#elif _MSC_VER
#  define SK_RESTRICT __restrict
#else
#  define SK_RESTRICT
#endif

#ifdef INCLUDE_CLHASH_H_
#  define ENABLE_CLHASH 1
#elif ENABLE_CLHASH
#  include "clhash.h"
#endif

#if !NDEBUG
#  define DBG_ONLY(...) __VA_ARGS__
#else
#  define DBG_ONLY(...)
#endif

#if VERBOSE_AF
#  define VERBOSE_ONLY(...) __VA_ARGS__
#else
#  define VERBOSE_ONLY(...)
#endif

#ifndef FOREVER
#  define FOREVER for(;;)
#endif

#if __has_cpp_attribute(no_unique_address)
#  define SK_NO_ADDRESS [[no_unique_address]]
#else
#  define SK_NO_ADDRESS
#endif

#ifndef CONST_IF
#  if defined(__cpp_if_constexpr) && __cplusplus >= __cpp_if_constexpr
#    define CONST_IF(...) if constexpr(__VA_ARGS__)
#  else
#    define CONST_IF(...) if(__VA_ARGS__)
#  endif
#endif
