#pragma once

#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <vector>

namespace pai {

/// A small vision transformer over the last few camera frames, forward pass only.
///
/// The weights come from PyTorch (transformer/training/train.py exports them); the
/// game only runs the network, so no backpropagation is written here.
///
/// Input: `frames` grey images of height x width, oldest first, concatenated.
/// Each image is cut into patch x patch squares; every square is one token, so
/// a 20x20 camera with 5x5 patches over 4 frames gives 64 tokens.
///
///   token  = W_patch * pixels + b_patch + position[t]
///   layer  : x += Attention(LayerNorm(x));  x += MLP(LayerNorm(x))   (pre-norm)
///   MLP    : d -> ff (GELU, erf form) -> d
///   output : LayerNorm, mean over tokens, linear -> actions
///
/// Attention is standard multi-head softmax(q k^T / sqrt(d_head)) v.
class Transformer {
public:
    struct Shape {
        std::size_t frames = 4, height = 20, width = 20, patch = 5;
        std::size_t model = 32, heads = 2, layers = 2, feedForward = 64, outputs = 4;

        std::size_t patchesPerFrame() const { return (height / patch) * (width / patch); }
        std::size_t tokens() const { return frames * patchesPerFrame(); }
        std::size_t patchPixels() const { return patch * patch; }
        std::size_t inputs() const { return frames * height * width; }
    };

    /// Reads a weight file written by the PyTorch exporter; throws on any mismatch.
    explicit Transformer(std::istream& in);

    const Shape& shape() const { return shape_; }

    /// Action scores (before softmax) for one input of length shape().inputs().
    std::vector<float> logits(const std::vector<float>& input) const;
    int predict(const std::vector<float>& input) const;

private:
    struct Layer {
        std::vector<float> ln1Gain, ln1Bias, qkvW, qkvB, outW, outB;
        std::vector<float> ln2Gain, ln2Bias, ff1W, ff1B, ff2W, ff2B;
    };

    Shape shape_;
    std::vector<float> patchW_, patchB_, position_;
    std::vector<Layer> layers_;
    std::vector<float> lnGain_, lnBias_, headW_, headB_;
};

/// Loads a transformer from a file; returns nullptr and fills *error on failure.
std::unique_ptr<Transformer> loadTransformer(const std::string& path, std::string* error);

}  // namespace pai
