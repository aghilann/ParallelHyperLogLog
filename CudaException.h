// CudaException.h
#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#include <exception>
#include <string>

class CudaException : public std::exception {
private:
    std::string message;
public:
    CudaException(const std::string& msg) : message(msg) {}

    virtual const char* what() const throw() {
        return message.c_str();
    }
};

#endif // CUDA_EXCEPTION_H
