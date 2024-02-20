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