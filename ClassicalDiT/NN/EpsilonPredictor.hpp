#pragma once

/**
 * @file EpsilonPredictor.hpp
 * @brief CNN-based Noise Prediction Network (epsilon_theta) for Latent Diffusion.
 *
 * @details
 * Predicts the noise epsilon added to a clean latent x_0 to produce the noisy
 * observation x_t at diffusion timestep t:
 *
 *   x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * epsilon,
 *   epsilon ~ N(0, I)
 *
 * This version delegates all layer execution to NeuralNetwork (from
 * NeuralNetworkLayers.hpp), which provides correct Conv->ReLU->Pool ordering,
 * fixed PoolingLayer [C][H][W] convention, and a working FullyConnected layer.
 *
 * @par Architecture
 * @code
 *   x_t  [latent_dim]  --+
 *   t_emb[embed_dim]   --+--> concat [latent_dim + embed_dim]
 *                              |
 *                    zero-pad --> reshape [grid_H][grid_W]  (2D, single channel)
 *                              |
 *           NeuralNetwork:
 *             Conv(1->64,   k=3, s=1)  +  ReLU          [64][grid_H][grid_W]
 *             Conv(64->128, k=3, s=1)  +  ReLU          [128][grid_H][grid_W]
 *             Conv(128->64, k=3, s=1)  +  ReLU          [64][grid_H][grid_W]
 *             Flatten                                    [1][1][64*grid_H*grid_W]
 *             FC(64*grid_H*grid_W -> latent_dim, relu=false)
 *                              |
 *                    epsilon_pred [latent_dim]  in R (double, NOT int)
 * @endcode
 *
 * Same-padding is achieved by wrapping Conv calls with pad=kernel/2. NeuralNetwork
 * uses ConvolutionalLayer which does not pad internally, so the input grid is
 * sized so that a k=3 kernel with no padding still preserves spatial dimensions:
 * the grid is chosen large enough that (grid_H - 2) still covers all data after
 * three same-dimension conv stages.
 *
 * @par Key Considerations
 * - Returns vector<double> — epsilon is continuous noise, never int.
 * - NeuralNetwork is a persistent member, constructed once. The previous
 *   EpsilonPredictor rebuilt the network on every predictEpsilon() call,
 *   resetting weights each time so nothing was ever learned.
 * - Timestep t is sinusoidally embedded and concatenated to x_t before
 *   the network sees any input. Without this the network cannot distinguish
 *   noise levels at different points in the diffusion schedule.
 * - Layer ordering is correct: NeuralNetwork's Stage dispatch fires each
 *   Conv->ReLU pair in declaration order. The previous NeuralNetwork::forward()
 *   fired all convs then all ReLUs then all pooling regardless of add order.
 * - FullyConnected now performs y = W*x + b. The previous Activation() ignored
 *   the weight matrix entirely and applied a second ReLU to the raw input.
 * - PoolingLayer [C][H][W] convention is now correct. The previous version
 *   treated H-rows as channels, taking max across channels not spatial windows.
 * - The output FC layer is constructed with apply_relu=false. Epsilon is
 *   unbounded in R; clipping to [0, inf) via ReLU would bias predictions
 *   toward zero at every timestep.
 *
 * @par Changes from EpsilonPredictor v2 (self-contained Conv2D/BN2D/FC structs)
 * - Private Conv2D, BN2D, and FC structs replaced by NeuralNetwork from
 *   NeuralNetworkLayers.hpp, unifying the layer implementation across the
 *   entire pipeline.
 * - BN2D removed from the conv stages. NeuralNetwork's ConvolutionalLayer
 *   does not include BatchNorm; regularisation comes from He initialisation
 *   and the diffusion NLL loss signal. BN can be reintroduced by extending
 *   NeuralNetworkLayers.hpp with a BatchNorm stage type.
 * - Checkpoint format updated: FC weights/biases are accessed via
 *   NeuralNetwork's FullyConnected::weights() and FullyConnected::biases()
 *   accessors. Conv kernel/bias access requires extending NeuralNetworkLayers
 *   with public weight accessors if full checkpoint coverage is needed.
 * - grid_H is set to (sqrt(latent_dim + embed_dim) + 2) to absorb the 1-pixel
 *   border lost per conv stage with no padding, so the spatial dimension after
 *   three k=3 unpadded convs still covers the original data.
 *
 * @par References
 * - Ho et al. (2020) "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
 * - He et al. (2015) "Delving Deep into Rectifiers". He initialisation.
 *
 * @author  OmniHenos / MAHQ_PY research programme
 * @version 3.0
 */

#include "NeuralNetworkLayers.hpp"

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cassert>
#include <iostream>


// =============================================================================
// EpsilonPredictor
// =============================================================================

/**
 * @brief CNN epsilon predictor epsilon_theta(x_t, t) backed by NeuralNetwork.
 *
 * @details
 * Wraps a NeuralNetwork instance (from NeuralNetworkLayers.hpp) with:
 *   - Sinusoidal timestep embedding concatenated to x_t before the network.
 *   - A fixed square grid reshape so the 1-D latent+embedding vector maps
 *     cleanly to the [H][W] input expected by NeuralNetwork::forward().
 *   - Persistent weights: the NeuralNetwork member is constructed once and
 *     updated by the training loop; it is not recreated per call.
 *
 * @par Connecting to GaussianDiffusion
 * @code
 *   EpsilonPredictor eps_net(64, 32);
 *   auto model_predict_epsilon = [&eps_net](
 *       const std::vector<double>& x_t, int t,
 *       const std::vector<double>& ) -> std::vector<double>
 *   {
 *       return eps_net.predictEpsilon(x_t, t);
 *   };
 * @endcode
 *
 * @par Connecting to TransposedCNNDecoder after reverse diffusion
 * @code
 *   // After DDPM reverse loop converges x_t -> x_0:
 *   auto z = doubles_to_floats({x_0});
 *   Tensor4D img = decoder.forward(z);
 *   TransposedCNNDecoder::save_ppm(img, "output.ppm");
 * @endcode
 *
 * @par Thread Safety
 * predictEpsilon() is not thread-safe when the NeuralNetwork's BN running
 * statistics are being updated during training. For concurrent inference
 * (read-only), use separate EpsilonPredictor instances per thread.
 *
 * @par Checkpoint Format
 * Only the FC layer weights and biases are serialised via the accessors
 * exposed by FullyConnected. Conv weights are re-initialised on load from
 * the same deterministic seed; extend NeuralNetworkLayers with public kernel
 * accessors to add full conv checkpoint coverage.
 */
class EpsilonPredictor {
public:

    // ── Public configuration ──────────────────────────────────────────────────

    int latent_dim; ///< Dimension of x_t. Must match GaussianDiffusion latent dimension.
    int embed_dim;  ///< Sinusoidal timestep embedding dimension. Must be even.
    int grid_H;     ///< Height of the 2D grid passed to NeuralNetwork::forward().
    int grid_W;     ///< Width of the 2D grid (equals grid_H; square).

    // ── Construction ──────────────────────────────────────────────────────────

    /**
     * @brief Construct EpsilonPredictor and build the NeuralNetwork layer stack.
     *
     * @param latent  Latent vector dimension (default 64). Must match the Clifford
     *                compression output and GaussianDiffusion latent dimension.
     * @param embed   Sinusoidal embedding size for timestep t (default 32).
     *                Must be even. Larger values give finer schedule resolution.
     * @param seed    RNG seed forwarded to NeuralNetwork layer initialisations
     *                (default 42). Same seed produces identical starting weights.
     *
     * @par Grid dimension derivation
     * The concatenated vector (x_t ++ t_emb) has length latent_dim + embed_dim.
     * It is zero-padded and reshaped to a square grid of side:
     *
     *   grid_H = ceil(sqrt(latent_dim + embed_dim)) + 2
     *
     * The +2 absorbs the 1-pixel border trimmed per k=3 conv stage (three stages
     * lose 3*2 = 6 border pixels total; +2 is a conservative lower bound that
     * keeps the spatial dimension positive after all conv stages). With same-padding
     * conv this would not be needed, but ConvolutionalLayer applies no padding.
     *
     * @par Network construction order
     * Stages are appended to NeuralNetwork in this exact sequence:
     *   Conv(1->64, k=3) + ReLU
     *   Conv(64->128, k=3) + ReLU
     *   Conv(128->64, k=3) + ReLU
     *   Flatten
     *   FC(flat_size -> latent_dim, relu=false)
     *
     * The flat_size after three unpadded k=3 convs on a grid_H x grid_H input is:
     *   flat_size = 64 * (grid_H - 6) * (grid_W - 6)
     *
     * @note FC is the only layer with publicly accessible weights for checkpointing.
     *       Conv weights are deterministically re-initialised on load from seed.
     */
    EpsilonPredictor(int latent = 64, int embed = 32, uint32_t seed = 42);

    // ── Timestep embedding ────────────────────────────────────────────────────

    /**
     * @brief Sinusoidal embedding of diffusion timestep t (Ho et al. 2020 §3.3).
     *
     * @details
     *   emb[2i]   = sin(t / 10000^(2i / embed_dim))
     *   emb[2i+1] = cos(t / 10000^(2i / embed_dim))
     *
     * Produces a unique, smoothly varying vector for every integer t in [0, T-1].
     * Parameter-free — no additional weights needed to encode timestep information.
     *
     * @param t  Diffusion timestep in [0, T-1].
     * @return   Embedding vector of length embed_dim, values in [-1, 1].
     *
     * @par Key Consideration - why sinusoidal
     * A learned embedding table would add T*embed_dim parameters (32,000 for
     * T=1000, D=32) and would not generalise beyond timesteps seen during
     * training. Sinusoidal embeddings are parameter-free and interpolate
     * smoothly, matching the smooth progression of the beta schedule.
     *
     * @par Key Consideration - embed_dim must be even
     * The loop writes two elements per iteration (sin, cos). An odd embed_dim
     * leaves the last element at zero. Default embed=32 satisfies this.
     */
    std::vector<double> timestep_embedding(int t) const;

    // ── Forward pass ──────────────────────────────────────────────────────────

    /**
     * @brief Predict noise epsilon_theta(x_t, t) for a single latent sample.
     *
     * @details
     * Pipeline:
     * -# Compute sinusoidal timestep embedding -> t_emb [embed_dim].
     * -# Concatenate x_t ++ t_emb -> flat [latent_dim + embed_dim].
     * -# Zero-pad flat to grid_H * grid_W; reshape to [grid_H][grid_W] (2D).
     * -# Pass to NeuralNetwork::forward(), which executes:
     *      Conv(1->64, k=3) + ReLU
     *      Conv(64->128, k=3) + ReLU
     *      Conv(128->64, k=3) + ReLU
     *      Flatten
     *      FC(flat_size -> latent_dim, relu=false)
     * -# Extract output from result[0][0] -> epsilon_pred [latent_dim].
     *
     * @param x_t  Noisy latent at timestep t. Length must equal latent_dim.
     * @param t    Diffusion timestep in [0, T-1].
     * @return     Predicted noise vector, length latent_dim, values in R.
     *             No activation applied — epsilon is unbounded.
     *
     * @throws std::invalid_argument if x_t.size() != latent_dim.
     * @throws std::runtime_error if NeuralNetwork produces unexpected output shape.
     *
     * @par Key Consideration - no output activation
     * The FC output layer is constructed with apply_relu=false. At early
     * timesteps (high t, low alpha_bar) individual epsilon samples from N(0,I)
     * can be +/-3 or beyond. Applying ReLU would clip negative predictions to
     * zero, biasing the reverse step toward positive values and degrading
     * sample quality.
     *
     * @par Key Consideration - NeuralNetwork output extraction
     * NeuralNetwork::forward() returns [1][1][out_dim] after Flatten+FC.
     * The epsilon prediction is extracted as result[0][0], which is the flat
     * output vector of length latent_dim.
     */
    std::vector<double> predictEpsilon(const std::vector<double>& x_t, int t);
    /**
     * @brief Batch prediction: run predictEpsilon() over N latents at the same t.
     *
     * @param batch  [N x latent_dim] noisy latent vectors.
     * @param t      Shared diffusion timestep for the entire batch.
     * @return       [N x latent_dim] predicted noise vectors.
     *
     * @note Calls are sequential. Each call uses the same sinusoidal t_emb.
     *       For true batch normalisation effects, a batched NeuralNetwork
     *       extension would be needed.
     *
     * @par Key Consideration - performance
     * Dominant cost is the FC forward: O(N * flat_size * latent_dim).
     * With latent_dim=64, grid_H=12, flat_size = 64*(12-6)*(12-6) = 2304,
     * this is ~9.4M multiply-adds per batch of 64. Fast enough for training
     * on CPU; a candidate for SIMD optimisation at larger batch sizes.
     */
    std::vector<std::vector<double>> predictBatch(
        const std::vector<std::vector<double>>& batch, int t);

    // ── Accessors for training loop ───────────────────────────────────────────

    /**
     * @brief Access the underlying NeuralNetwork for weight updates.
     *
     * @return Reference to the internal NeuralNetwork. Use to access
     *         fc_store_ weights/biases for gradient application.
     *
     * @note Use the FullyConnected::weights() and FullyConnected::biases()
     *       accessors exposed by NeuralNetworkLayers.hpp to retrieve the
     *       mutable weight vectors for SGD/Adam updates.
     */
    NeuralNetwork& network() { return nn_; }
    const NeuralNetwork& network() const { return nn_; }

    /**
     * @brief Infer the flat feature size fed into the FC layer.
     *
     * @details After three unpadded k=3 convolutions on a grid_H x grid_H
     * input, each conv reduces spatial size by (k-1)=2:
     *   after conv1: (grid_H - 2) x (grid_W - 2)
     *   after conv2: (grid_H - 4) x (grid_W - 4)
     *   after conv3: (grid_H - 6) x (grid_W - 6)
     * Output channels after conv3: 64.
     *
     * @return Total number of values in the flattened feature map.
     */
    int flat_size() const;

    // ── Checkpoint I/O ────────────────────────────────────────────────────────

    /**
     * @brief Save FC layer weights and biases to a binary checkpoint file.
     *
     * @param path  Destination file path.
     * @return      true if write succeeded, false on I/O error.
     *
     * @note Conv weights are NOT saved. They are deterministically re-initialised
     *       from seed on load. To add full conv weight checkpointing, extend
     *       ConvolutionalLayer in NeuralNetworkLayers.hpp with public kernel
     *       accessors and add conv save/load calls here.
     *
     * @par Binary layout
     * @code
     *   [int32 latent_dim][int32 embed_dim][int32 grid_H][int32 grid_W]
     *   [int32 fc_seed]
     *   [uint64 fc_weight_count][double... fc_weights]
     *   [uint64 fc_bias_count  ][double... fc_biases ]
     * @endcode
     */
    bool save(const std::string& path) const;

    /**
     * @brief Load FC weights and biases from a binary checkpoint file.
     *
     * @param path  Source file previously written by save().
     * @return      true on success, false if file could not be opened.
     *
     * @throws std::runtime_error on dimension mismatch.
     *
     * @note Conv layers are rebuilt deterministically from the original seed
     *       after the dimension header is validated. This keeps the checkpoint
     *       compact while still restoring the trained FC output projection.
     */
    bool load(const std::string& path);

private:

    /// Persistent NeuralNetwork — weights survive across all predictEpsilon() calls.
    NeuralNetwork nn_;

    // ── Network construction ──────────────────────────────────────────────────

    /**
     * @brief Build the NeuralNetwork layer stack.
     *
     * @details Called once from the constructor. Appends layers in the order
     * that NeuralNetwork::forward() will execute them:
     *
     *   Conv(1->64, k=3, s=1)     stage 0
     *   ReLU                       stage 1
     *   Conv(64->128, k=3, s=1)   stage 2
     *   ReLU                       stage 3
     *   Conv(128->64, k=3, s=1)   stage 4
     *   ReLU                       stage 5
     *   Flatten                    stage 6
     *   FC(flat_size->latent_dim, relu=false)  stage 7
     *
     * @note No pooling layers are added here. Pooling would further reduce
     *       spatial dimensions, requiring a larger initial grid to maintain
     *       sufficient feature map size after the FC. Add pooling here and
     *       adjust grid_H in the constructor if receptive field expansion
     *       is needed for larger latent dimensions.
     *
     * @param seed  Passed to ConvolutionalLayer for deterministic He init.
     *              Currently unused by NeuralNetwork::addConvolutionalLayer
     *              directly (ConvolutionalLayer derives seed from layer dims);
     *              retained as a parameter for future seeding extension.
     */
    void build_network(uint32_t /*seed*/) {
        // Three conv-relu stages, no padding (ConvolutionalLayer convention)
        nn_.addConvolutionalLayer(1,   64,  3, 1);  // stage 0
        nn_.addReluLayer();                          // stage 1
        nn_.addConvolutionalLayer(64,  128, 3, 1);  // stage 2
        nn_.addReluLayer();                          // stage 3
        nn_.addConvolutionalLayer(128, 64,  3, 1);  // stage 4
        nn_.addReluLayer();                          // stage 5
        nn_.addFlatten();                            // stage 6

        // FC: flat_size -> latent_dim, no ReLU on output
        // flat_size = 64 * (grid_H - 6) * (grid_W - 6)  (6 = 3 stages * 2 border px)
        int fs = flat_size();
        nn_.addFullyConnectedLayer(fs, latent_dim, /*apply_relu=*/false); // stage 7
    }

    // ── Binary I/O helpers ────────────────────────────────────────────────────

    static void write_vec(std::ofstream& f, const std::vector<double>& v) {
        uint64_t sz = v.size();
        f.write(reinterpret_cast<const char*>(&sz), 8);
        f.write(reinterpret_cast<const char*>(v.data()), sz * sizeof(double));
    }
    static void read_vec(std::ifstream& f, std::vector<double>& v) {
        uint64_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), 8);
        v.resize(sz);
        f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(double));
    }
};