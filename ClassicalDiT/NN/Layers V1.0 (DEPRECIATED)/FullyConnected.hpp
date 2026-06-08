#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP
#include <vector>

class FullyConnected {
    public:
        std::vector<std::vector<std::vector<double>>> Inputweights(const std::vector<std::vector<std::vector<double>>>& input) const;
        // Activation
        std::vector<std::vector<std::vector<double>>> Activation(const std::vector<std::vector<std::vector<double>>>& input, std::vector<std::vector<std::vector<double>>> weights) const;

};

#endif // FULLYCONNECTED_HPP
