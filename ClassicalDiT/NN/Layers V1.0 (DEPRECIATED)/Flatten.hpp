#ifndef FLATTEN_HPP
#define FLATTEN_HPP
#include <vector>

class Flatten {
public:
    // Forward pass: Convert 3D input to 1D output
    
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input) const;

    // Backward pass: Convert 1D gradient back to 3D gradient
    std::vector<std::vector<std::vector<double>>> backward(std::vector<double>& gradient, 
                                                            std::vector<std::vector<std::vector<double>>>& inputShape);
};

#endif // FLATTEN_HPP