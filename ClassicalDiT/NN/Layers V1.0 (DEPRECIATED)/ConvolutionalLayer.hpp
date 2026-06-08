#ifndef CONVOLUTIONALLAYER_HPP
#define CONVOLUTIONALLAYER_HPP

#include <vector>
#include <iostream>
#include <random>
#include <cmath>

class ConvolutionalLayer {
public:
    ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride);
    std::vector<std::vector<std::vector<double>>> forwardPass(const std::vector<std::vector<std::vector<double>>>& input) const;
    // Add methods for backward pass, weight updates, etc. as needed

private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    std::vector<std::vector<std::vector<double>>> kernels_; // 3D vector for kernels
    std::vector<double> biases_; // Biases for each output channel

    void initializeWeights();
};

#endif // CONVOLUTIONALLAYER_HPP