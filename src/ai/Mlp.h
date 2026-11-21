#pragma once

#include <cstddef>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

namespace pai {

/// One training example: an observation and the index of the action taken.
struct Sample {
    std::vector<float> input;
    int label = 0;
};

/// Two-layer perceptron, the same network as in 2016:
///   input -> sigmoid hidden layer -> softmax over actions,
/// trained with cross-entropy and stochastic gradient descent.
/// Weights live in contiguous row-major arrays (one allocation per layer).
class Mlp {
public:
    Mlp(std::size_t inputs, std::size_t hidden, std::size_t outputs, unsigned seed);

    std::size_t inputs() const { return inputs_; }
    std::size_t hidden() const { return hidden_; }
    std::size_t outputs() const { return outputs_; }

    /// Action probabilities for one input (length inputs()).
    std::vector<float> probabilities(const std::vector<float>& input) const;
    /// Most probable action.
    int predict(const std::vector<float>& input) const;

    /// One gradient step on a minibatch; returns the batch's mean cross-entropy
    /// before the step.
    float trainBatch(const std::vector<const Sample*>& batch, float learningRate);

    void save(std::ostream& out) const;
    /// Replaces the weights with those in the stream; throws if the shape differs.
    void load(std::istream& in);

private:
    void forward(const float* input, float* hiddenOut, float* probsOut) const;

    std::size_t inputs_;
    std::size_t hidden_;
    std::size_t outputs_;
    std::vector<float> w1_;  // hidden x inputs
    std::vector<float> b1_;  // hidden
    std::vector<float> w2_;  // outputs x hidden
    std::vector<float> b2_;  // outputs
};

}  // namespace pai
