#pragma once
#ifndef SKETCHCEPTION_H__
#define SKETCHCEPTION_H__
#include <stdexcept>

namespace sketch {

namespace exception {

class NotImplementedError: public std::runtime_error {
public:
    template<typename... Args>
    NotImplementedError(Args &&...args): std::runtime_error(std::forward<Args>(args)...) {}

    NotImplementedError(): std::runtime_error("NotImplemented.") {}
};

#ifdef __CUDACC__
struct CudaError: public std::runtime_error {
public:
    CudaError(cudaError_t ce, std::string s): std::runtime_error(std::string("cudaError_t [") + cudaGetErrorString(ce) + "]" + s) {}
};
#endif // __CUDACC__
}
using namespace exception;


} // sketch

#endif
