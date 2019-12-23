#pragma once
#ifndef SKETCH_MACROS_H__
#define SKETCH_MACROS_H__



// INLINE
#ifndef INLINE
#  ifdef __CUDACC__
#    define INLINE __forceinline__ // inline
#  elif __GNUC__ || __clang__
#    define INLINE __attribute__((always_inline)) inline
#  else
#    define INLINE inline
#  endif
#endif


// OpenMP

#ifdef _OPENMP
#  ifndef OMP_PRAGMA
#    define OMP_PRAGMA(...) _Pragma(__VA_ARGS__)
#  endif
#  ifndef OMP_ONLY
#     define OMP_ONLY(...) __VA_ARGS__
#  endif
#else
#  ifndef OMP_PRAGMA
#    define OMP_PRAGMA(...)
#  endif
#  ifndef OMP_ONLY
#    define OMP_ONLY(...)
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

#define sk__str__(x) #x
#define sk__xstr__(x) sk__str__(x)
#define CPP_PASTE(...) sk__xstr__(__VA_ARGS__)
#define CPP_PASTE_UNROLL(...) sk__xstr__("unroll" __VA_ARGS__)


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

#ifndef SK_UNROLL
#  define SK_UNROLL _Pragma("message \"The macro, it does nothing\"")
   // Don't use SK_UNROLL, it only tells you if these below macros are defined.
#  if defined(__CUDACC__)
#    define SK_UNROLL_4  _Pragma("unroll 4")
#    define SK_UNROLL_8  _Pragma("unroll 8")
#    define SK_UNROLL_16 _Pragma("unroll 16")
#    define SK_UNROLL_32 _Pragma("unroll 32")
#    define SK_UNROLL_64 _Pragma("unroll 64")
#  elif defined(__GNUC__)
#    define SK_UNROLL_4  _Pragma("GCC unroll 4")
#    define SK_UNROLL_8  _Pragma("GCC unroll 8")
#    define SK_UNROLL_16 _Pragma("GCC unroll 16")
#    define SK_UNROLL_32 _Pragma("GCC unroll 32")
#    define SK_UNROLL_64 _Pragma("GCC unroll 64")
#  else
#    define SK_UNROLL_4
#    define SK_UNROLL_8
#    define SK_UNROLL_16
#    define SK_UNROLL_32
#    define SK_UNROLL_64
#  endif
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

#endif /* SKETCH_MACROS_H__ */
