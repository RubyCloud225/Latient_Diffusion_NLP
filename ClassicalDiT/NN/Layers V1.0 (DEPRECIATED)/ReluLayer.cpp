#include "ReluLayer.hpp"

ReLu::ReLu() {}

// Method to compute the output of the ReLU layer
std::vector<std::vector<std::vector<double>>>ReLu::forward(const std::vector<std::vector<std::vector<double>>>& input) const {
    std::vector<std::vector<std::vector<double>>> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i].resize(input[i].size());
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j].resize(input[i][j].size());
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                output[i][j][k] = std::max(0.0, input[i][j][k]);
            }
        }
    }
    return output;
}

// Helper function to compute the ReLu activation
double ReLu::relu(double x) {
    return std::max(0.0, x);
}

//Method for backward Propagation
std::vector<std::vector<std::vector<double>>> ReLu::backward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<double>& upstreamGradient) {
    std::vector<std::vector<std::vector<double>>> downstreamGradient(input.size(), std::vector<std::vector<double>>(input[0].size(), std::vector<double>(input[0][0].size())));
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                downstreamGradient[i][j][k] = reluGradient(input[i][j][k]) * upstreamGradient[i];
            }
        }
    }
    return downstreamGradient;
}

double ReLu::reluGradient(double x) {
    return x > 0 ? 1 : 0.0;
}

