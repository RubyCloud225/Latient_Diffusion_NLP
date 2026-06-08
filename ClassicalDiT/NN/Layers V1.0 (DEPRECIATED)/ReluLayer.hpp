#ifndef RELU_HPP
#define RELU_HPP
#include <vector>

class ReLu {
    public:
    // constructor
    ReLu();
    // method to compute the ReLU function
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) const;
    // method for backward propagation
    std::vector<std::vector<std::vector<double>>> backward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<double>& upstreamGradient);

    private:
    // private method to compute the derivative of the ReLU function
    double relu(double x);
    double reluGradient(double x);
};

#endif
