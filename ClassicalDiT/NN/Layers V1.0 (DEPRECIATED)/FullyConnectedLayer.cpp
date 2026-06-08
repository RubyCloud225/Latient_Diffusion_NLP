#include "FullyConnected.hpp"
#include <vector>
#include <algorithm> // for std::max
#include <iostream>
#include <ostream>

std::vector<std::vector<std::vector<double>>> FullyConnected::Inputweights(const std::vector<std::vector<std::vector<double>>>& input) const {
    std::vector<std::vector<std::vector<double>>> weights(input.size(), std::vector<std::vector<double>>(input[0].size(), std::vector<double>(input[0][0].size())));
    
    // Initialize weights with random values or zeros
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                weights[i][j][k] = 0.0; // or use a random initialization
            }
        }
    }
    
    // Debug print weights dimensions
    std::cout << "FullyConnected::Inputweights weights dimensions: " << weights.size() << " x " << weights[0].size() << " x " << weights[0][0].size() << std::endl;
    
    return weights;
}

std::vector<std::vector<std::vector<double>>> FullyConnected::Activation(const std::vector<std::vector<std::vector<double>>>& input, std::vector<std::vector<std::vector<double>>> weights) const {
    std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size())));
    
    // Debug print input and weights dimensions
    std::cout << "FullyConnected::Activation input dimensions: " << input.size() << " x " << input[0].size() << " x " << input[0][0].size() << std::endl;
    std::cout << "FullyConnected::Activation weights dimensions: " << weights.size() << " x " << weights[0].size() << " x " << weights[0][0].size() << std::endl;
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            for (size_t k = 0; k < weights[0][0].size(); ++k) {
                // Example activation function: ReLU
                output[i][j][k] = std::max(0.0, input[i][j][k]); // Replace with your activation function
            }
        }
    }
    return output;
}

// backward pass
std::vector<std::vector<std::vector<double>>> Backward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<std::vector<double>>>& grad_output) {
    std::vector<std::vector<std::vector<double>>> grad_input(input.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size())));
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            for (size_t k = 0; k < weights[0][0].size(); ++k) {
                double sum = 0.0;
                for (size_t l = 0; l < input[i].size(); ++l) {
                    sum += grad_output[i][j][k] * weights[l][j][k];
                }
                grad_input[i][j][k] = sum; // Add bias gradient if needed
            }
        }
    }
    
    return grad_input;
}
