#ifndef _LOG_UTIL_H__
#define _LOG_UTIL_H__
#ifndef __STDC_FORMAT_MACROS
#  define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>

#define _FUNCTION_MACRO_ __PRETTY_FUNCTION__
#define LOG_INFO(...) log_info(__func__, ##__VA_ARGS__);
#define LOG_WARNING(...) log_warning(_FUNCTION_MACRO_, ##__VA_ARGS__);
#define LOG_EXIT(...) log_error(_FUNCTION_MACRO_, __LINE__, ##__VA_ARGS__);
#if !NDEBUG
#    define LOG_DEBUG(...) log_debug(_FUNCTION_MACRO_, __LINE__, ##__VA_ARGS__);
#    define LOG_ASSERT(condition) log_assert(_FUNCTION_MACRO_, __LINE__, ((uint64_t)(condition)), (#condition))
#else
#    define LOG_DEBUG(...)
#    define LOG_ASSERT(...)
#endif

static inline void log_debug(const char *func, int line, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[D:%s:%d] ", func, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static inline void log_warning(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[W:%s] ", func);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static inline void log_info(const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[%s] ", func);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

static inline void log_error(const char *func, int line, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[E:%s:%d] ", func, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

static inline void log_assert(const char *func, int line, int assertion, const char *assert_str) {
    if(!assertion) {
        fprintf(stderr, "[E:%s:%d] Assertion '%s' failed.",
                func, line, assert_str);
        exit(EXIT_FAILURE);
    }
}

#endif /* #ifndef _LOG_UTIL_H__ */
