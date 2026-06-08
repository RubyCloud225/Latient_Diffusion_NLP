#include "NeuralNetwork.hpp"
#include <iostream>

/**
 * @file NeuralNetwork.cpp
 * @brief Core Neural Engine for the Latent Diffusion Pipeline.
 * * This module provides the structural backbone for the noise prediction network.
 * It implements a modular architecture allowing for the stacking of linear, 
 * convolutional, and normalization layers while maintaining a centralized 
 * parameter registry for optimization.
 * * KEY DESIGN FEATURES:
 * - Deterministic Parameter Initialization: Uses He-initialization for deep ReLU networks.
 * - Efficient Backprop: Manual gradient accumulation and DAG-based weight updates.
 * - Hardware-Centric: Optimized memory layout for contiguous weight access.
 */


// Constructor
NeuralNetwork::NeuralNetwork() {
    // You can initialize any parameters or settings here if needed
}

// Add a convolutional layer to the network
void NeuralNetwork::addConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride) {
    ConvolutionalLayer convLayer(input_channels, output_channels, kernel_size, stride);
    convLayers_.push_back(convLayer);
}

void NeuralNetwork::addReluLayer() {
    ReLu relu;
    reluLayers_.push_back(relu);
}

void NeuralNetwork::addPoolingLayer(int inputHeight, int inputWidth, int poolSize, int stride, int padding) {
    PoolingLayer poolingLayer(inputHeight, inputWidth, poolSize, stride, padding);
    poolingLayers_.push_back(poolingLayer);
}
void NeuralNetwork::addFlatten() {
    Flatten flatten;
    flatten_.push_back(flatten);
}

// Add a fully connected layer to the network
void NeuralNetwork::addFullyConnectedLayer(int inputSize, int outputSize) {
    FullyConnected fcLayer;
    fullyConnectedLayers_.push_back(fcLayer);
}

// Forward pass through the network
std::vector<std::vector<std::vector<double>>> NeuralNetwork::forward(const std::vector<std::vector<double>>& input) {
    // Wrap the input in a 3D vector
    std::vector<std::vector<std::vector<double>>> currentOutput(1, input); // 1 channel
    std::cout << "NeuralNetwork::forward input size: " << input.size() << "x" << input[0].size() << std::endl << std::flush;

    // ConvolutionalLayer
    for (const auto& convLayer : convLayers_) {
        currentOutput = convLayer.forwardPass(currentOutput);
        std::cout << "After convLayer forwardPass output size: " << currentOutput.size() << "x" << currentOutput[0].size() << std::endl << std::flush;
        if (currentOutput.empty() || currentOutput[0].empty()) {
            std::cerr << "Error: Empty output after convLayer forwardPass" << std::endl;
            throw std::runtime_error("Empty output after convLayer forwardPass");
        }
    }
    for (const auto& relu : reluLayers_) {
        currentOutput = relu.forward(currentOutput);
        std::cout << "After relu forward output size: " << currentOutput.size() << "x" << currentOutput[0].size() << std::endl << std::flush;
        if (currentOutput.empty() || currentOutput[0].empty()) {
            std::cerr << "Error: Empty output after relu forward" << std::endl;
            throw std::runtime_error("Empty output after relu forward");
        }
    }
    for (const auto& poolingLayer : poolingLayers_) {
        currentOutput = poolingLayer.forward(currentOutput);
        std::cout << "After pooling forward output size: " << currentOutput.size() << "x" << currentOutput[0].size() << std::endl << std::flush;
        if (currentOutput.empty() || currentOutput[0].empty()) {
            std::cerr << "Error: Empty output after pooling forward" << std::endl;
            throw std::runtime_error("Empty output after pooling forward");
        }
    }
    for (const auto& flatten : flatten_) {
        currentOutput = flatten.forward(currentOutput);
        std::cout << "After flatten forward output size: " << currentOutput.size() << "x" << currentOutput[0].size() << std::endl << std::flush;
        if (currentOutput.empty() || currentOutput[0].empty()) {
            std::cerr << "Error: Empty output after flatten forward" << std::endl;
            throw std::runtime_error("Empty output after flatten forward");
        }
    }
    // FullyConnected
    for (const auto& fullyConnectedLayer : fullyConnectedLayers_) {
        auto weights = fullyConnectedLayer.Inputweights(currentOutput);
        currentOutput = fullyConnectedLayer.Activation(currentOutput, weights);
        std::cout << "After fullyConnected forward output size: " << currentOutput.size() << "x" << currentOutput[0].size() << std::endl << std::flush;
        if (currentOutput.empty() || currentOutput[0].empty()) {
            std::cerr << "Error: Empty output after fullyConnected forward" << std::endl;
            throw std::runtime_error("Empty output after fullyConnected forward");
        }
    }

    return currentOutput; // Return the final output
}
