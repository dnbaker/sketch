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

}
using namespace exception;

}

#endif
