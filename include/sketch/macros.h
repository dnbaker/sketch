#pragma once
#ifndef SKETCH_MACROS_H__
#define SKETCH_MACROS_H__
#include "hedley.h"


// INLINE
#ifndef INLINE
#  if __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif

// unlikely/likely
#ifndef unlikely
#   define unlikely(x) HEDLEY_UNLIKELY((x))
#endif

#ifndef likely
#  if defined(__GNUC__) || defined(__INTEL_COMPILER)
#    define likely(x) HEDLEY_LIKELY(!!(x))
#  else
#    define likely(x) (x)
#  endif
#endif


// OpenMP

#ifdef _OPENMP
#  ifndef OMP_PRAGMA
#    define OMP_PRAGMA(x) _Pragma(x)
#  endif
#  ifndef OMP_ONLY
#     define OMP_ONLY(...) __VA_ARGS__
#  endif
#  ifndef OMP_PFOR
#    define OMP_PFOR OMP_PRAGMA("omp parallel for")
#  endif
#  ifndef OMP_PFOR_DYN
#    define OMP_PFOR_DYN OMP_PRAGMA("omp parallel for schedule(dynamic)")
#  endif
#  ifndef OMP_ELSE
#    define OMP_ELSE(x, y) x
#  endif
#  ifndef OMP_ATOMIC
#    define OMP_ATOMIC OMP_PRAGMA("omp atomic")
#  endif
#  ifndef OMP_CRITICAL
#    define OMP_CRITICAL OMP_PRAGMA("omp critical")
#  endif
#  ifndef OMP_SECTIONS
#    define OMP_SECTIONS OMP_PRAGMA("omp sections")
#  endif
#  ifndef OMP_SECTION
#    define OMP_SECTION OMP_PRAGMA("omp section")
#  endif
#  ifndef OMP_BARRIER
#    define OMP_BARRIER OMP_PRAGMA("omp barrier")
#  endif
#  ifndef OMP_SET_NT
#    define OMP_SET_NT(x) omp_set_num_threads(x)
#  endif
#else
#  ifndef OMP_PRAGMA
#    define OMP_PRAGMA(x)
#  endif
#  ifndef OMP_ONLY
#    define OMP_ONLY(...)
#  endif
#  ifndef OMP_ELSE
#    define OMP_ELSE(x, y) y
#  endif
#  ifndef OMP_PFOR
#    define OMP_PFOR
#  endif
#  ifndef OMP_PFOR_DYN
#    define OMP_PFOR_DYN
#  endif
#  ifndef OMP_ATOMIC
#    define OMP_ATOMIC
#  endif
#  ifndef OMP_CRITICAL
#    define OMP_CRITICAL
#  endif
#  ifndef OMP_SECTIONS
#    define OMP_SECTIONS
#  endif
#  ifndef OMP_SECTION
#    define OMP_SECTION
#  endif
#  ifndef OMP_BARRIER
#    define OMP_BARRIER
#  endif
#  ifndef OMP_SET_NT
#    define OMP_SET_NT(x)
#  endif
#endif


#ifndef SK_RESTRICT
#  if __CUDACC__ || __GNUC__ || __clang__
#    define SK_RESTRICT __restrict__
#  elif _MSC_VER
#    define SK_RESTRICT __restrict
#  else
#    define SK_RESTRICT
#  endif
#endif

#ifdef __CUDA_ARCH__
#  define CUDA_ARCH_ONLY(...) __VA_ARGS__
#  define HOST_ONLY(...)
#else
#  define CUDA_ARCH_ONLY(...)
#  define HOST_ONLY(...) __VA_ARGS__
#endif

#ifdef __CUDACC__
#  define CUDA_PRAGMA(x) _Pragma(x)
#  define CUDA_ONLY(...) __VA_ARGS__
#else
#  define CUDA_PRAGMA(x)
#  define CUDA_ONLY(...)
#endif

#define CPP_PASTE(...) sk__xstr__(__VA_ARGS__)
#define CPP_PASTE_UNROLL(...) sk__xstr__("unroll" __VA_ARGS__)


#ifndef THREADSAFE_ELSE
#  ifndef NOT_THREADSAFE
#    define THREADSAFE_ELSE(x, y) x
#    define THREADSAFE_ONLY(...) __VA_ARGS__
#  else
#    define THREADSAFE_ELSE(x, y) y
#    define THREADSAFE_ONLY(...)
#  endif
#endif


#if !NDEBUG
#  define DBG_ONLY(...) __VA_ARGS__
#  define DBG_ELSE(x, y) x
#else
#  define DBG_ONLY(...)
#  define DBG_ELSE(x, y) y
#endif

#if VERBOSE_AF
#  define VERBOSE_ONLY(...) __VA_ARGS__
#else
#  define VERBOSE_ONLY(...)
#endif

#ifndef FOREVER
#  define FOREVER for(;;)
#endif

#ifndef SK_UNROLL
#  define SK_UNROLL _Pragma("message \"The macro, it does nothing\"")
   // Don't use SK_UNROLL, it only tells you if these below macros are defined.
#  if defined(__GNUC__) && !defined(__clang__)
#    define SK_UNROLL_4  _Pragma("GCC unroll 4")
#    define SK_UNROLL_8  _Pragma("GCC unroll 8")
#    define SK_UNROLL_16 _Pragma("GCC unroll 16")
#    define SK_UNROLL_32 _Pragma("GCC unroll 32")
#    define SK_UNROLL_64 _Pragma("GCC unroll 64")
#  elif defined(__CUDACC__) || defined(__clang__)
#    define SK_UNROLL_4  _Pragma("unroll 4")
#    define SK_UNROLL_8  _Pragma("unroll 8")
#    define SK_UNROLL_16 _Pragma("unroll 16")
#    define SK_UNROLL_32 _Pragma("unroll 32")
#    define SK_UNROLL_64 _Pragma("unroll 64")
#  else
#    define SK_UNROLL_4
#    define SK_UNROLL_8
#    define SK_UNROLL_16
#    define SK_UNROLL_32
#    define SK_UNROLL_64
#  endif
#endif

#if defined(__has_cpp_attribute) && __cplusplus >= __has_cpp_attribute(no_unique_address)
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

#ifndef BLAZE_CHECK_DEBUG
#  ifndef NDEBUG
#    define BLAZE_CHECK_DEBUG
#  else
#    define BLAZE_CHECK_DEBUG , ::blaze::unchecked
#  endif
#endif

#endif /* SKETCH_MACROS_H__ */
