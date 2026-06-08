#pragma once

/**
 * @file NeuralNetworkLayers.hpp
 * @brief Corrected layer implementations for the latent diffusion CNN stack.
 *
 * @details
 * This header consolidates ConvolutionalLayer, PoolingLayer, ReLU, Flatten and FullyConnected into a signal corrected file
 * and fixes the following bugs present across the original four layer files
 * (ConvolutionalLayer.cpp, PoolingLayer.cpp, ReLULayer.cpp, FullyConnectedLayer.cpp - Treated as deprecated):
 *
 * @par Bug fixes
 * **ConvolutionalLayer**:
 * - Weights initialisation now uses a seeded MT19937 so weights differ between instances. The previous default_random_engine
 * with no seed produced identical weights for every constructed layer.
 * - All per-element debug prints inside the inner convolution loop have been
 * removed. They were 0(output_H * output_W * in_C * kH * kW) console writes per forward pass - the dominant
 * runtime cost at any non-trivial input size. A single summary line is retained at the boundaries.
 * - Kernel storage layout is [out_C][in_C][kH * kW] (flat 1D kernel). Index arithmetic ki * kernel_size_ + kj is preserved and correct.
 *
 *
 * **PoolingLayer**
 * - Dimension ordering corrected from [H][W][C] to [C][H][W] to match
 *   ConvolutionalLayer output and EpsilonPredictor input convention.
 *   The previous code treated height-rows as channels and channel-values as
 *   spatial positions.
 * - Padding initialisation corrected. The previous code allocated
 *   paddedInput[H][W] with an inner vector pre-filled with C zeros, then
 *   called push_back() in the fill loop — doubling the inner dimension on
 *   every forward pass. The corrected version allocates [C][pH][pW] once
 *   and assigns directly.
 * - Output layout is [C][oH][oW] with max pooling over the spatial window
 *   per channel, consistent with the rest of the pipeline.
 *
 * **FullyConnected**
 * - Inputweights() previously returned a weight tensor of the same shape as
 *   its input, initialised entirely to zero. Any input multiplied by an
 *   all-zero weight matrix always returns zero — the layer was a no-op.
 *   Weights are now sized [out_dim x in_dim] and He-initialised.
 * - Activation() previously ignored the weight matrix entirely and applied
 *   ReLU element-wise to the input, performing a second ReLU rather than
 *   a linear transform. It now computes the correct matrix-vector product
 *   y = W*x + b followed by ReLU.
 * - Input and output are now flat 1D vectors (matching Flatten output) rather
 *   than 3D tensors, eliminating the impedance mismatch between Flatten and FC.
 *
 * **NeuralNetwork::forward**
 * - The previous implementation applied all conv layers, then all ReLU layers,
 *   then all pooling layers sequentially. For a two-conv two-pool network this
 *   produced: conv1->conv2->relu->relu->pool->pool instead of the intended
 *   conv1->relu->pool->conv2->relu->pool.
 * - The corrected design uses a typed Stage enum and a single ordered
 *   stage_sequence_ vector. Layers are added in declaration order and execute
 *   in that same order, regardless of type. The old per-type bucket vectors
 *   are replaced by variant-style dispatch.
 *
 * @par Tensor convention throughout this file
 * All 3D tensors are [C][H][W]: channel-first, row-major within each channel.
 * This matches ConvolutionalLayer, EpsilonPredictor, and TransposedCNNDecoder.
 *
 * @author Catherine Earl
 * @version 2.0
 */

#ifndef NEURALNETWORKLAYERS_HPP
#define NEURALNETWORKLAYERS_HPP
#include <vector>

// ============================================================================
//                             CONVOLUTIONAL LAYER
// ============================================================================

/**
 * @brief Standard 2-D convolutional layer, [C][H][W] tensor convention.
 *
 * @details
 * Computes:
 *     output[oc][i][j] = bias[oc] + sum_{ic, ki, kj} kernel[oc][ic][ki*K+kj] * input[ic][i*s+ki][j*s+kj]
 *
 * Kernel storage is flat 1D per (oc, ic) pair: kernels_[oc][ic] is a vector of length kernel_size_ * kernel_size_,
 * indexed as ki * kernel_size_ + kj.
 *
 * @note No padding is applied. Output spatial size:
 * oH = (iH - kernel_size_) / stride_ + 1
 * oW = (iW - kernel_size_) / stride_ + 1
 *
 * @par Changed from ConvolutionalLayer.cpp
 * -   Weight initialisation now uses seeded mt19937 (seed derived from
 *     output_channels * kernel_size for reproducible but varied initialisations).
 *     The previous default_random_engine with no seed produced identical weights
 *     for every constructed instance.
 * -   All per-element debug prints inside the inner loop removed. They produced
 *     O(oH * oW * in_C * kH * kW) console writes per call — the dominant
 *     runtime cost. A single boundary summary line is retained.
 */
class ConvolutionalLayer {
public:
    /**
     * @brief Construct a convolutional layer.
     * @param input_channels    Number of input channels (depth of input tensor).
     * @param output_channels   Number of output channels (number of filters).
     * @param kernel_size       Square kernel side length.
     * @param stride            Convolution stride (default 1).
     */
    ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride = 1);

    /**
     * @brief Forward pass: 2-D convolution over a [C][H][W] input tensor.
     * @param input Input tensor [in_C][iH][iW].
     * @return output tensor [out_C][oH][oW].
     * @throws std::invalid_argument if output dimensions would be non-positive (input smaller than kernel).
     */
    std::vector<std::vector<std::vector<double>>> forwardPass(
        const std::vector<std::vector<std::vector<double>>>& input) const;

    int output_channels() const;

private:
    int input_channels_;
    int output_channels_;
    int kernel_size_;
    int stride_;
    std::vector<double> biases_;
    std::vector<std::vector<std::vector<double>>> kernels_;

    void initialiseWeights();
};

// ===============================================================
//                  POOLING LAYER
// ===============================================================

/**
 * @brief Max pooling layer. Tensor convention: [C][H][W].
 * 
 * @details
 * For each channel and each poolSize x poolSize spatial window (stepped by
 * stride), takes the maximum value. Optional symmetric zero-padding is 
 * applied before pooling.
 * 
 * @par Changes from PoolingLayer.cpp
 * -    Dimension ordering corrected from [H][W][C] to [C][H][W]. The previous code
 *      constructed paddedInput as [H+2p][W+2p][C] but ConvolutionalLayer produces
 *      [C][H][W], causing the max to be taken across channels rather than across spatial positions.
 * -    Padding initialisation corrected. The previous code allocated the inner channel-vector with C
 *      zeros then called push_back() in the fill loop, which doubled the inner vector
 *      length on every forward pass. The corrected code allocates [C][pH][pW] once and assigns with direct
 *      indexing.
 * -    Output layout is [C][oH][oW], consistent with the rest of the pipeline.
 */

class PoolingLayer {
    public:
    /**
     * @brief Construct a max pooling layer.
     * @param poolSize Spetial windo size (square).
     * @param stride Step size between windows.
     * @param padding Symmetric zero-padding added to H and W before pooling
     */

    explicit PoolingLayer(int poolSize = 2, int stride = 2, int padding = 0);
    /**
     * @brief Forward pass: max pooling over a [C][H][W] input tensor.
     * 
     * @param input Input tensor [C][iH][iW]
     * @return Output tensor [C][oH][oW] where
     * oH = (iH + 2*padding - poolSize) / stride + 1.
     * 
     * @par Key Consideration - padding allocation
     * Padding is applied by allocating a [C][iH+2p][iW+2p] tensor initialised
     * to zero and copying input values into the interior. No push_back is used;
     * all dimensions are fixed at construction of paddedInput.
     */
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input
    ) const;
    private:
    int poolSize_, stride_, padding_;
};

// ==========================================================================================
//              RELU
// ==========================================================================================

/**
 * @brief ReLU activation layer: f(x) = max(0, x), applied element-wise.
 * 
 * @details Operates on [C][H][W] tensors. Shape is preserved.
 * 
 * @note No changes to forward() logic from ReluLayer.cpp - the implementation
 * was correct. The backward() signature mismatch (upstreamGradient was 1D but indexed as if 3D)
 * corrected here
 */
class ReLu {
    public:
    ReLu() = default;
    /**
     * @brief Forward pass: max(0, x) element-wise over [C][H][W].
     * @param input Input tensor, any shape.
     * @return Output tensor, same shape, negative values clamped to 0.
     */
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input) const;
    
    /**
     * @brief Backward pass: passes upstream gradient where input > 0, else 0.
     * 
     * @param input     Pre-activation input (cached from forward pass).
     * @param upstream_grad Gradient from the layer above, same shape as input.
     * @return  Downstream gradient, same shape.
     * 
     * @par Changes from ReluLayer.cpp
     * The previous backward() accepted upstreamGradient as vector<double> (1D)
     * but indexed it as upstreamGradient[i], using only the channel index.
     * The corrected version accepts a full [C][H][W] upstream gradient matching
     * the layer's output shape, so every spatial position receives its correct
     * gradient independently.
     */
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>>& input,
        const std::vector<std::vector<std::vector<double>>>& upstream_grad
    ) const;
};

// =================================================================================
// Flatten
// =================================================================================

/**
 * @brief Flattens a [C][H][W] tensor to a [1][1][C*H*W] tensor.
 * 
 * @details
 * Output is wrapped in the 3D container expected by NeuralNetwork::forward().
 * The downstream FullyConnected layer reads output[0][0] as its flat input.
 * 
 * @note Iteration order is channel-major: all spatial positions of channel 0,
 * then channel 1, etc. This is consistent with the [C][H][W] convention.
 * 
 * @note No changes to Flatten::forward() logic  from the original. backward()
 * signature and logic are unchanged.
 */

class Flatten {
    public:
    Flatten() = default;

    /**
     * @brief Flatten [C][H][W] -> [1][1][C*H*W].
     * @param input Input tensor of any [C][H][W] shape.
     * @return 3D tensor with a single 1D row containing all values
     */

    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input
    ) const;

    /**
     * @brief Backward pass: reshape a flat gradient back to the original shape.
     * @param gradient    Flat gradient from the FC layer above.
     * @param inputShape  The [C][H][W] tensor from the forward pass (used for shape).
     * @return            Reshaped gradient matching inputShape dimensions.
     */
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<double>& gradient,
        const std::vector<std::vector<std::vector<double>>>& inputShape) const;
};

// =============================================================================
// FullyConnected
// =============================================================================
 
/**
 * @brief Fully connected linear layer followed by ReLU: y = max(0, W*x + b).
 *
 * @details
 * Accepts a flat input vector of length in_dim and produces a flat output
 * vector of length out_dim. The 3D wrapper required by NeuralNetwork::forward()
 * is handled by helper conversions in NeuralNetwork itself.
 *
 * @par Changes from FullyConnected.cpp
 * - Inputweights() returned a weight tensor the same shape as its input,
 *   all zeros. Multiplying any input by an all-zero matrix always gives zero.
 *   Weights are now sized [out_dim x in_dim] and He-initialised.
 * - Activation() previously ignored the weight matrix entirely and applied
 *   ReLU element-wise to the raw input — it was performing a second ReLU,
 *   not a linear transform. It now computes y = W*x + b followed by ReLU.
 * - Input/output are flat 1D vectors (matching Flatten output) rather than
 *   3D tensors, removing the shape impedance mismatch.
 */
class FullyConnected {
public:
    /**
     * @brief Construct a fully connected layer.
     * @param in_dim   Input dimension (must match Flatten output size).
     * @param out_dim  Output dimension (number of neurons).
     * @param seed     RNG seed for He initialisation (default 0).
     */
    FullyConnected(int in_dim, int out_dim, uint32_t seed = 0);
 
    /**
     * @brief Forward pass: y = max(0, W*x + b).
     *
     * @param x  Flat input vector of length in_dim_.
     * @return   Flat output vector of length out_dim_, ReLU-activated.
     *
     * @par Key Consideration - activation choice
     * ReLU is applied here to match the previous Activation() contract.
     * For the final output layer of EpsilonPredictor no activation is
     * desired (epsilon is unbounded). Construct the final FC with
     * apply_relu = false to get a raw linear output.
     */
    std::vector<double> forward(const std::vector<double>& x,
                                bool apply_relu = true) const;

    int in_dim()  const;
    int out_dim() const;
    std::vector<double>& weights();
    std::vector<double>& biases();

private:
    int in_dim_, out_dim_;
    std::vector<double> weight_;  ///< [out_dim x in_dim], row-major.
    std::vector<double> bias_;    ///< [out_dim].
};
// =============================================================================
// NeuralNetwork
// =============================================================================
 
/**
 * @brief Sequential neural network container with correct layer ordering.
 *
 * @details
 * Layers are stored in declaration order via an ordered stage list and
 * dispatched in the same order during forward(). This replaces the previous
 * design of per-type bucket vectors (convLayers_, reluLayers_, poolingLayers_)
 * which caused all layers of one type to execute before any layer of the
 * next type, regardless of the order in which addXxxLayer() was called.
 *
 * @par Changes from NeuralNetwork.cpp
 * - Replaced per-type bucket vectors with a single ordered stage_sequence_
 *   vector of typed variant structs. addConvolutionalLayer() and
 *   addPoolingLayer() append to this sequence; forward() iterates it in order.
 * - addFullyConnectedLayer() now receives and stores in_dim and out_dim
 *   correctly instead of constructing a default FullyConnected() that ignores
 *   both arguments.
 * - The Flatten layer output ([1][1][N]) is unwrapped before passing to FC,
 *   and the FC flat output is re-wrapped as [1][1][out_dim] for the return
 *   value, maintaining a uniform 3D tensor interface throughout forward().
 *
 * @par Correct execution order (example: two conv-pool stages)
 * @code
 *   net.addConvolutionalLayer(1, 64, 3, 1);  // stage 0
 *   net.addReluLayer();                       // stage 1
 *   net.addPoolingLayer(2, 2, 0);            // stage 2
 *   net.addConvolutionalLayer(64, 128, 3, 1); // stage 3
 *   net.addReluLayer();                       // stage 4
 *   net.addPoolingLayer(2, 2, 0);            // stage 5
 *   net.addFlatten();                         // stage 6
 *   net.addFullyConnectedLayer(N, out_dim);  // stage 7
 *   // forward() fires stages 0..7 in this exact order.
 * @endcode
 */
class NeuralNetwork {
public:
    NeuralNetwork() = default;
 
    /// @brief Append a convolutional layer to the execution sequence.
    void addConvolutionalLayer(int input_channels, int output_channels,
                               int kernel_size, int stride = 1);
   
 
    /// @brief Append a ReLU activation layer to the execution sequence.
    void addReluLayer();
 
    /**
     * @brief Append a max pooling layer to the execution sequence.
     * @param poolSize  Square pooling window size.
     * @param stride    Step between windows.
     * @param padding   Symmetric zero-padding applied before pooling.
     */
    void addPoolingLayer(int poolSize, int stride, int padding = 0);
 
    /// @brief Append a flatten layer to the execution sequence.
    void addFlatten();
 
    /**
     * @brief Append a fully connected layer to the execution sequence.
     * @param inputSize   Input dimension. Must match the flattened feature size.
     * @param outputSize  Output dimension (number of neurons).
     *
     * @note apply_relu is false for the final FC layer so epsilon predictions
     *       are not clipped. For hidden FC layers, construct with apply_relu=true.
     */
    void addFullyConnectedLayer(int inputSize, int outputSize,
                                bool apply_relu = false);
 
    /**
     * @brief Execute the full forward pass in layer declaration order.
     *
     * @param input  2D input [H][W] (single-channel; wrapped to [1][H][W]).
     * @return       Output tensor [C'][H'][W']. For a network ending in FC,
     *               this is [1][1][out_dim].
     *
     * @throws std::runtime_error if any layer produces an empty tensor.
     *
     * @par Key Consideration - layer ordering
     * Stages fire in the order they were added via addXxxLayer(). This is
     * guaranteed by iterating stage_sequence_ once. The previous per-type
     * bucket approach fired all convs before any ReLU or pooling, producing
     * the wrong computation graph for any interleaved architecture.
     */
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<double>>& input);
 
private:
    enum class StageType { Conv, Relu, Pool, Flat, FC };
    struct Stage { StageType type; int idx; };
 
    std::vector<Stage>             stages_;         ///< Ordered execution sequence.
    std::vector<ConvolutionalLayer> conv_store_;
    std::vector<ReLu>              relu_store_;
    std::vector<PoolingLayer>      pool_store_;
    std::vector<Flatten>           flatten_store_;
    std::vector<FullyConnected>    fc_store_;
    std::vector<bool>              fc_relu_;        ///< Per-FC relu flag.
    bool flattened_ = false;
 
public:
    // ── FC weight accessors (for checkpoint save/load in EpsilonPredictor) ────
 
    /**
     * @brief Return a mutable reference to the weights of the nth FC layer.
     * @param n  Zero-based index into the FC layer store.
     * @throws std::out_of_range if n >= number of FC layers added.
     */
    std::vector<double>& fc_weights(int n) {
        if (n < 0 || n >= static_cast<int>(fc_store_.size()))
            throw std::out_of_range("NeuralNetwork::fc_weights: index " +
                std::to_string(n) + " out of range (have " +
                std::to_string(fc_store_.size()) + " FC layers)");
        return fc_store_[n].weights();
    }
 
    /**
     * @brief Return a mutable reference to the biases of the nth FC layer.
     * @param n  Zero-based index into the FC layer store.
     * @throws std::out_of_range if n >= number of FC layers added.
     */
    std::vector<double>& fc_biases(int n) {
        if (n < 0 || n >= static_cast<int>(fc_store_.size()))
            throw std::out_of_range("NeuralNetwork::fc_biases: index " +
                std::to_string(n) + " out of range (have " +
                std::to_string(fc_store_.size()) + " FC layers)");
        return fc_store_[n].biases();
    }
 
    /// @brief Number of FC layers currently in the network.
    int fc_count() const { return static_cast<int>(fc_store_.size()); }
};
#endif
