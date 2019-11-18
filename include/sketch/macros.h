#pragma once
#ifndef SKETCH_MACROS_H__
#define SKETCH_MACROS_H__



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

#define CPP_PASTE(...) sk__xstr__(__VA_ARGS__)
#define CPP_PASTE_UNROLL(...) sk__xstr__("unroll" __VA_ARGS__)

#ifdef __CUDA_ARCH__
#  define SK_UNROLL(...) _Pragma(CPP_PASTE(unroll __VA_ARGS__))
#elif defined(__clang__)
#  define SK_UNROLL(...) _Pragma(CPP_PASTE(unroll __VA_ARGS__))
#elif defined(__GNUC__) && __GNUC__ > 8
#  define SK_UNROLL(...) _Pragma(CPP_PASTE(GCC unroll __VA_ARGS__))
#else
#  define SK_UNROLL(...)
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

#endif /* SKETCH_MACROS_H__ */
