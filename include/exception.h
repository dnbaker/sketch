#pragma once
#ifndef SKETCHCEPTION_H__
#define SKETCHCEPTION_H__
#include <stdexcept>

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

class UnsatisfiedPostconditionError: public std::runtime_error {
public:
    UnsatisfiedPostconditionError(std::string msg): std::runtime_error(std::string("Unsatisfied precondition: ") + msg) {}

    UnsatisfiedPostconditionError(): std::runtime_error("Unsatisfied precondition.") {}
};

struct ZlibError: public std::runtime_error {
public:
    ZlibError(int ze, std::string s): std::runtime_error(std::string("zlibError [") + zError(ze) + "]" + s) {}
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
