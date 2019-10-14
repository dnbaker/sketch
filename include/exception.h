#pragma once
#ifndef SKETCHCEPTION_H__
#define SKETCHCEPTION_H__
#include <stdexcept>
#if ZWRAP_USE_ZSTD
#  include "zstd_zlibwrapper.h"
#else
#  include <zlib.h>
#endif

namespace sketch {

inline namespace exception {

class NotImplementedError: public std::runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): std::runtime_error(std::forward<Args>(args)...) {}

    NotImplementedError(): std::runtime_error("NotImplemented.") {}
};

class UnsatisfiedPreconditionError: public std::runtime_error {
public:
    UnsatisfiedPreconditionError(std::string msg): std::runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPreconditionError(): std::runtime_error("Unsatisfied precondition.") {}
};


static int precondition_require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw UnsatisfiedPreconditionError(s + " Error code: " + std::to_string(ec));
        else throw UnsatisfiedPreconditionError(s);
    }
    return ec;
}

class UnsatisfiedPostconditionError: public std::runtime_error {
public:
    UnsatisfiedPostconditionError(std::string msg): std::runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPostconditionError(): std::runtime_error("Unsatisfied precondition.") {}
};

static int postcondition_require(bool condition, std::string s, int ec=0) {
    if(!condition) {
        if(ec) throw UnsatisfiedPostconditionError(s + " Error code: " + std::to_string(ec));
        else throw UnsatisfiedPostconditionError(s);
    }
    return ec;
}

#define PREC_REQ_EC(condition, s, ec) \
    ::sketch::exception::precondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#define PREC_REQ(condition, s) PREC_REQ_EC(condition, s, 0)
#define POST_REQ_EC(condition, s, ec) \
    ::sketch::exception::postcondition_require(condition, std::string(s) + '[' + __FILE__ + '|' + __PRETTY_FUNCTION__ + "|#L" + std::to_string(__LINE__) + "] Failing condition: \"" + #condition + '"', ec)
#define POST_REQ(condition, s) POST_REQ_EC(condition, s, 0)

class ZlibError: public std::runtime_error {
    static const char *es(int c) {
        static constexpr const char * const z_errmsg[10] = {
            (z_const char *)"need dictionary",     /* Z_NEED_DICT       2  */
            (z_const char *)"stream end",          /* Z_STREAM_END      1  */
            (z_const char *)"",                    /* Z_OK              0  */
            (z_const char *)"file error",          /* Z_ERRNO         (-1) */
            (z_const char *)"stream error",        /* Z_STREAM_ERROR  (-2) */
            (z_const char *)"data error",          /* Z_DATA_ERROR    (-3) */
            (z_const char *)"insufficient memory", /* Z_MEM_ERROR     (-4) */
            (z_const char *)"buffer error",        /* Z_BUF_ERROR     (-5) */
            (z_const char *)"incompatible version",/* Z_VERSION_ERROR (-6) */
            (z_const char *)""
        };
        return z_errmsg[Z_NEED_DICT - c];
    }
public:
    ZlibError(int ze, std::string s): std::runtime_error(std::string("zlibError [") + es(ze) + "]" + s) {}
    ZlibError(std::string s): ZlibError(Z_ERRNO, s) {}
};

#ifdef __CUDACC__
struct CudaError: public std::runtime_error {
public:
    CudaError(cudaError_t ce, std::string s): std::runtime_error(std::string("cudaError_t [") + cudaGetErrorString(ce) + "]" + s) {}
};
#endif // __CUDACC__
} // exception

} // sketch

#endif
