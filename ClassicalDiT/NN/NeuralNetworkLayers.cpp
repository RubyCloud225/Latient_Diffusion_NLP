#include "NeuralNetworkLayers.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <functional>
#include <cassert>



ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride) :
    input_channels_(input_channels),
    output_channels_(output_channels),
    kernel_size_(kernel_size),
    stride_(stride) {
        initialiseWeights();
    }

void ConvolutionalLayer::initialiseWeights() {
    std::mt19937 rng(static_cast<unsigned>(output_channels_ * kernel_size_));
    double std_dev = std::sqrt(2.0 / (input_channels_ * kernel_size_ * kernel_size_));
    std::normal_distribution<double> dist(0.0, std_dev);

    kernels_.assign(output_channels_,
        std::vector<std::vector<double>>(input_channels_,
            std::vector<double>(kernel_size_ * kernel_size_)));

    for (int oc = 0; oc < output_channels_; ++oc)
        for (int ic = 0; ic < input_channels_; ++ic)
            for (int k = 0; k < kernel_size_ * kernel_size_; ++k)
                kernels_[oc][ic][k] = dist(rng);

    biases_.assign(output_channels_, 0.0);
}

std::vector<std::vector<std::vector<double>>> ConvolutionalLayer::forwardPass(
    const std::vector<std::vector<std::vector<double>>>& input) const {
        int in_C = static_cast<int>(input.size());
        int iH = static_cast<int>(input[0].size());
        int iW = static_cast<int>(input[0][0].size());
        int oH = (iH - kernel_size_) / stride_ + 1;
        int oW = (iW - kernel_size_) / stride_ + 1;

        if (oH <= 0 || oW <= 0) throw std::invalid_argument(
            "ConvolutionalLayer::forwardPass: kernel larger than input. "
            "iH=" + std::to_string(iH) + " iW=" + std::to_string(iW) + " k=" + std::to_string(kernel_size_)
        );

        std::vector<std::vector<std::vector<double>>> output(
            output_channels_,
            std::vector<std::vector<double>>(oH, std::vector<double>(oW, 0.0))
        );

        for (int oc = 0; oc < output_channels_; ++oc)
        for (int i = 0; i < oH; ++i)
        for (int j = 0; j < oW; ++j) {
            double sum = biases_[oc];
            for (int ic = 0; ic < in_C; ++ic)
            for (int ki = 0; ki < kernel_size_; ++ki)
            for (int kj = 0; kj < kernel_size_; ++kj) {
                int ir = i * stride_ + ki;
                int ic2 = j * stride_ + kj;
                sum += input[ic][ir][ic2] * kernels_[oc][ic][ki * kernel_size_ + kj];
            }
            output[oc][i][j] = sum;
        }
        return output;
}

int ConvolutionalLayer::output_channels() const { return output_channels_; }


PoolingLayer::PoolingLayer(int poolSize, int stride, int padding) : poolSize_(poolSize), stride_(stride), padding_(padding) {}
std::vector<std::vector<std::vector<double>>> PoolingLayer::forward(
        const std::vector<std::vector<std::vector<double>>>& input) const 
    {
        int C = static_cast<int>(input.size());
        int iH = static_cast<int>(input[0].size());
        int iW = static_cast<int>(input[0][0].size());
        int pH = iH + 2 * padding_;
        int pW = iW + 2 * padding_;

        // Allocate padded input [C][pH][pW], zero-initialised.
        // Assign interior; border cells remain 0
        std::vector<std::vector<std::vector<double>>> padded(C, std::vector<std::vector<double>>(
            pH, std::vector<double>(pW, 0.0)
        ));

        for (int c = 0; c < C; ++c)
        for (int h = 0; h < iH; ++h)
        for (int w = 0; w < iW; ++w)
            padded[c][h + padding_][w + padding_] = input[c][h][w];
        
        int oH = (pH - poolSize_) / stride_ + 1;
        int oW = (pW - poolSize_) / stride_ + 1;

        std::vector<std::vector<std::vector<double>>>output(C, std::vector<std::vector<double>>(oH, std::vector<double>(oW, 0.0)));

        for (int c = 0; c < C;  ++c)
        for (int i = 0; i < oH; ++i)
        for (int j = 0; j < oW; ++j) {
            double mx = -std::numeric_limits<double>::infinity();
            for (int m = 0; m < poolSize_; ++m)
            for (int n = 0; n < poolSize_; ++n) {
                int r = i * stride_ + m;
                int s = j * stride_ + n;
                if (r < pH && s < pW)
                    mx = std::max(mx, padded[c][r][s]);
            }
            output[c][i][j] = mx;
        }
        return output;
    }

std::vector<std::vector<std::vector<double>>> ReLu::forward(
        const std::vector<std::vector<std::vector<double>>>& input) const {
            auto output = input;
            for (auto& ch : output)
                for (auto& row : ch)
                    for (auto& v : row)
                        if (v < 0.0) v = 0.0;
            return output;
    }

std::vector<std::vector<std::vector<double>>> ReLu::backward(
        const std::vector<std::vector<std::vector<double>>>& input,
        const std::vector<std::vector<std::vector<double>>>& upstream_grad
    ) const {
        auto grad = upstream_grad;
        for (size_t c = 0; c < input.size(); ++c)
            for (size_t h = 0; h < input[c].size(); ++h)
            for (size_t w = 0; w < input[c][h].size(); ++w)
                if (input[c][h][w] <= 0.0) grad[c][h][w] = 0.0;
        return grad;
    }

std::vector<std::vector<std::vector<double>>> Flatten::forward(
        const std::vector<std::vector<std::vector<double>>>& input
    ) const {
        std::vector<double> flat;
        flat.reserve(input.size() * input[0].size() * input[0][0].size());
        for (const auto& ch  : input)
        for (const auto& row : ch)
        for (const double v  : row)
            flat.push_back(v);
        return { { flat } };  // [1][1][C*H*W]
    }

std::vector<std::vector<std::vector<double>>> Flatten::backward(
    const std::vector<double>& gradient,
    const std::vector<std::vector<std::vector<double>>>& inputShape) const {
            std::vector<std::vector<std::vector<double>>> out(
            inputShape.size(), std::vector<std::vector<double>>(
                inputShape[0].size(),
                std::vector<double>(inputShape[0][0].size(), 0.0)));
        size_t idx = 0;
        for (size_t c = 0; c < inputShape.size();    ++c)
        for (size_t h = 0; h < inputShape[c].size(); ++h)
        for (size_t w = 0; w < inputShape[c][h].size(); ++w) {
            out[c][h][w] = (idx < gradient.size()) ? gradient[idx++] : 0.0;
        }
        return out;
    }

FullyConnected::FullyConnected(int in_dim, int out_dim, uint32_t seed)
        : in_dim_(in_dim), out_dim_(out_dim),
          weight_(out_dim * in_dim, 0.0), bias_(out_dim, 0.0)
    {
        std::mt19937 rng(seed);
        double std_dev = std::sqrt(2.0 / in_dim);
        std::normal_distribution<double> dist(0.0, std_dev);
        for (auto& w : weight_) w = dist(rng);
    }

std::vector<double> FullyConnected::forward(const std::vector<double>& x,
                                bool apply_relu) const
    {
        assert(static_cast<int>(x.size()) == in_dim_);
        std::vector<double> out(out_dim_, 0.0);
        for (int o = 0; o < out_dim_; ++o) {
            double sum = bias_[o];
            for (int i = 0; i < in_dim_; ++i)
                sum += weight_[o * in_dim_ + i] * x[i];
            out[o] = apply_relu ? std::max(0.0, sum) : sum;
        }
        return out;
    }
 
int FullyConnected::in_dim()  const { return in_dim_; }
int FullyConnected::out_dim() const { return out_dim_; }

std::vector<double>& FullyConnected::weights() { return weight_; }
std::vector<double>& FullyConnected::biases()  { return bias_; }

void NeuralNetwork::addConvolutionalLayer(int input_channels, int output_channels,
                               int kernel_size, int stride = 1)
    {
        conv_store_.emplace_back(input_channels, output_channels,
                                  kernel_size, stride);
        stages_.push_back({StageType::Conv,
                           static_cast<int>(conv_store_.size()) - 1});
    }

void NeuralNetwork::addReluLayer() {
        relu_store_.emplace_back();
        stages_.push_back({StageType::Relu,
                           static_cast<int>(relu_store_.size()) - 1});
    }

void NeuralNetwork::addPoolingLayer(int poolSize, int stride, int padding = 0) {
        pool_store_.emplace_back(poolSize, stride, padding);
        stages_.push_back({StageType::Pool,
                           static_cast<int>(pool_store_.size()) - 1});
    }

void NeuralNetwork::addFlatten() {
        flatten_store_.emplace_back();
        stages_.push_back({StageType::Flat,
                           static_cast<int>(flatten_store_.size()) - 1});
        flattened_ = true;
    }

void NeuralNetwork::addFullyConnectedLayer(int inputSize, int outputSize,
                                bool apply_relu = false) {
        fc_store_.emplace_back(inputSize, outputSize);
        fc_relu_.push_back(apply_relu);
        stages_.push_back({StageType::FC,
                           static_cast<int>(fc_store_.size()) - 1});
    }
    
std::vector<std::vector<std::vector<double>>> NeuralNetwork::forward(
        const std::vector<std::vector<double>>& input)
    {
        // Wrap 2D input as single-channel [1][H][W]
        std::vector<std::vector<std::vector<double>>> x(1, input);
 
        for (const auto& stage : stages_) {
            switch (stage.type) {
            case StageType::Conv:
                x = conv_store_[stage.idx].forwardPass(x);
                break;
            case StageType::Relu:
                x = relu_store_[stage.idx].forward(x);
                break;
            case StageType::Pool:
                x = pool_store_[stage.idx].forward(x);
                break;
            case StageType::Flat:
                x = flatten_store_[stage.idx].forward(x);
                break;
            case StageType::FC: {
                // Unwrap [1][1][N] -> flat vector, run FC, re-wrap to [1][1][M]
                if (x.empty() || x[0].empty() || x[0][0].empty())
                    throw std::runtime_error("NeuralNetwork: empty tensor before FC layer");
                const std::vector<double>& flat_in = x[0][0];
                bool relu = fc_relu_[stage.idx];
                std::vector<double> flat_out = fc_store_[stage.idx].forward(flat_in, relu);
                x = { { flat_out } };
                break;
            }
            }
 
            if (x.empty() || x[0].empty())
                throw std::runtime_error("NeuralNetwork: layer produced empty output");
        }
        return x;
    }