#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include "Layers/ConvolutionalLayer.hpp"
#include "Layers/PoolingLayer.hpp"
#include "Layers/Flatten.hpp"
#include "Layers/FullyConnected.hpp"
#include "Layers/ReluLayer.hpp"

class NeuralNetwork {
    public:
    NeuralNetwork();
    void addConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize, int stride);
    void addReluLayer();
    void addPoolingLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, int stride);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<double>>& input);
    void addFlatten();
    void addFullyConnectedLayer(int inputSize, int outputSize);
  

    private:
    std::vector<ConvolutionalLayer> convLayers_; // vector to hold the layers
    std::vector<ReLu> reluLayers_; // vector to hold the ReLU layers
    std::vector<PoolingLayer> poolingLayers_; // vector to hold the pooling layers
    std::vector<Flatten> flatten_;
    std::vector<FullyConnected> fullyConnectedLayers_; // vector to hold the fully connected layers
};

#endif