#include "PoolingLayer.hpp"

PoolingLayer::PoolingLayer(int inputHeight, int inputWidth, int poolSize, int stride, int padding)
: inputHeight(inputHeight), inputWidth(inputWidth), poolSize(poolSize), stride(stride), padding(padding) {}

std::vector<std::vector<std::vector<double>>> PoolingLayer::forward(const std::vector<std::vector<std::vector<double>>>& input) const {
    //calculate the output dimensions
    int paddingHeight = inputHeight + 2 * padding;
    int paddingWidth = inputWidth + 2 * padding;

    // Initialize padded input
    std::vector<std::vector<std::vector<double>>> paddedInput(paddingHeight, std::vector<std::vector<double>>(paddingWidth, std::vector<double>(input[0][0].size(), 0)));

    //Fill the padded input with the input values
    for (int i = 0; i < inputHeight; i++) {
        for (int j = 0; j < inputWidth; j++) {
            for (int k = 0; k < input[i][j].size(); k++) {
                paddedInput[i + padding][j + padding].push_back(input[i][j][k]);
            }
        }
    }
    // Calculate the output dimensions
    int outputHeight = (paddingHeight - poolSize) / stride + 1;
    int outputWidth = (paddingWidth - poolSize) / stride + 1;
    // Initialize output
    std::vector<std::vector<std::vector<double>>> output(outputHeight, std::vector<std::vector<double>>(outputWidth, std::vector<double>(input[0][0].size(), 0)));
    // Perform pooling
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            std::vector<double> maxVals = paddedInput[i * stride][j * stride];
            for (int m = 0; m < poolSize; ++m) {
                for (int n = 0; n < poolSize; ++n) {
                    int row = i * stride + m;
                    int col = j * stride + n;
                    if (row < paddingHeight && col < paddingWidth) {
                        for (size_t k = 0; k < maxVals.size(); ++k) {
                            maxVals[k] = std::max(maxVals[k], paddedInput[row][col][k]);
                        }
                    }
                }
            }
            output[i][j] = maxVals;
        }
    }
    return output;
}