#include "EpsilonPredictor.hpp"
#include "NeuralNetworkLayers.hpp"
#include <vector>

EpsilonPredictor::EpsilonPredictor(int latent = 64, int embed = 32, uint32_t seed = 42)
        : latent_dim(latent), embed_dim(embed)
    {
        // Grid: smallest square covering (latent + embed) values, +2 for conv border
        int total_in = latent + embed;
        int base = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(total_in))));
        grid_H = base + 2;  // absorbs 1-pixel border lost per conv stage (3 stages = 6px; +2 conservative)
        grid_W = grid_H;

        build_network(seed);
    }

std::vector<double> EpsilonPredictor::timestep_embedding(int t) const {
        std::vector<double> emb(embed_dim, 0.0);
        for (int i = 0; i < embed_dim / 2; ++i) {
            double freq = std::pow(10000.0, 2.0 * i / embed_dim);
            emb[2 * i]     = std::sin(t / freq);
            emb[2 * i + 1] = std::cos(t / freq);
        }
        return emb;
    }
std::vector<double> EpsilonPredictor::predictEpsilon(const std::vector<double>& x_t, int t) {
        if (static_cast<int>(x_t.size()) != latent_dim)
            throw std::invalid_argument(
                "EpsilonPredictor::predictEpsilon: x_t.size()=" +
                std::to_string(x_t.size()) + " != latent_dim=" +
                std::to_string(latent_dim));
        if (t < 0)
            throw std::invalid_argument(
                "EpsilonPredictor::predictEpsilon: t must be >= 0, got " +
                std::to_string(t));

        // Step 1: sinusoidal timestep embedding
        auto t_emb = timestep_embedding(t);

        // Step 2: concatenate x_t ++ t_emb
        std::vector<double> flat;
        flat.reserve(latent_dim + embed_dim);
        flat.insert(flat.end(), x_t.begin(),   x_t.end());
        flat.insert(flat.end(), t_emb.begin(), t_emb.end());

        // Step 3: zero-pad to grid_H * grid_W; reshape to [grid_H][grid_W]
        //         Cells beyond flat.size() remain 0.
        std::vector<std::vector<double>> grid(
            grid_H, std::vector<double>(grid_W, 0.0));
        for (int i = 0; i < static_cast<int>(flat.size()); ++i)
            grid[i / grid_W][i % grid_W] = flat[i];

        // Step 4: NeuralNetwork forward (Conv->ReLU->Conv->ReLU->Conv->ReLU->Flatten->FC)
        auto result = nn_.forward(grid);

        // Step 5: extract epsilon_pred from [1][1][latent_dim]
        if (result.empty() || result[0].empty() || result[0][0].empty())
            throw std::runtime_error(
                "EpsilonPredictor::predictEpsilon: NeuralNetwork returned empty output");

        return result[0][0];  // flat vector of length latent_dim
    }

std::vector<std::vector<double>> EpsilonPredictor::predictBatch(
        const std::vector<std::vector<double>>& batch, int t)
    {
        std::vector<std::vector<double>> out;
        out.reserve(batch.size());
        for (const auto& x_t : batch)
            out.push_back(predictEpsilon(x_t, t));
        return out;
    }
int EpsilonPredictor::flat_size() const {
        return 64 * (grid_H - 6) * (grid_W - 6);
    }

bool EpsilonPredictor::save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;

        int32_t ld = latent_dim, ed = embed_dim,
                gh = grid_H,     gw = grid_W;
        f.write(reinterpret_cast<const char*>(&ld), 4);
        f.write(reinterpret_cast<const char*>(&ed), 4);
        f.write(reinterpret_cast<const char*>(&gh), 4);
        f.write(reinterpret_cast<const char*>(&gw), 4);

        // FC weights and biases via NeuralNetwork accessor
        // nn_ exposes fc_store_ through network(); we reach fc[0] via the
        // stage-ordered store. For now access the single FC layer directly.
        const auto& fc_w = const_cast<NeuralNetwork&>(nn_).fc_weights(0);
        const auto& fc_b = const_cast<NeuralNetwork&>(nn_).fc_biases(0);
        write_vec(f, fc_w);
        write_vec(f, fc_b);

        return f.good();
    }

bool EpsilonPredictor::load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        int32_t ld, ed, gh, gw;
        f.read(reinterpret_cast<char*>(&ld), 4);
        f.read(reinterpret_cast<char*>(&ed), 4);
        f.read(reinterpret_cast<char*>(&gh), 4);
        f.read(reinterpret_cast<char*>(&gw), 4);

        if (ld != latent_dim || ed != embed_dim)
            throw std::runtime_error(
                "EpsilonPredictor::load -- dimension mismatch: "
                "file latent=" + std::to_string(ld) +
                " embed=" + std::to_string(ed) +
                ", instance latent=" + std::to_string(latent_dim) +
                " embed=" + std::to_string(embed_dim));

        auto& fc_w = nn_.fc_weights(0);
        auto& fc_b = nn_.fc_biases(0);
        read_vec(f, fc_w);
        read_vec(f, fc_b);

        return f.good();
    }