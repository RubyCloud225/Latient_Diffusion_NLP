#include "Flatten.hpp"

// forward pass
std::vector<std::vector<std::vector<double>>> Flatten::forward(const std::vector<std::vector<std::vector<double>>>& input) const {

    std::vector<double> output;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            for (const auto& value : row) {
                output.push_back(value);
            }
        }
    }
    return { { output } };
}

std::vector<std::vector<std::vector<double>>> Flatten::backward(std::vector<double>& gradient, std::vector<std::vector<std::vector<double>>>& inputShape) {
    std::vector<std::vector<std::vector<double>>> reshapedGradient(inputShape.size(), std::vector<std::vector<double>>(inputShape[0].size(), std::vector<double>(inputShape[0][0].size())));
    size_t index = 0;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        for (size_t j = 0; j < inputShape[i].size(); ++j) {
            for (size_t k = 0; k < inputShape[i][j].size(); ++k) {
                if (index < gradient.size()) {
                    reshapedGradient[i][j][k] = gradient[index++];
                } else {
                    reshapedGradient[i][j][k] = 0; // Fill with zeros if gradient is shorter than expected
                }
            }
        }
    }
    return reshapedGradient;
}