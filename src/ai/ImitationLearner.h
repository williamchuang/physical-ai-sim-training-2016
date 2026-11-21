#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "ai/Mlp.h"

namespace pai {

/// Fixed-capacity ring buffer of recent demonstrations. Training draws random
/// minibatches from it, so consecutive, nearly identical camera frames do not
/// dominate a gradient step.
class ReplayBuffer {
public:
    explicit ReplayBuffer(std::size_t capacity) : capacity_(capacity) { samples_.reserve(capacity); }

    void push(Sample s);
    std::size_t size() const { return samples_.size(); }
    std::size_t capacity() const { return capacity_; }
    /// Up to n distinct samples chosen uniformly at random.
    std::vector<const Sample*> minibatch(std::size_t n, std::mt19937& rng) const;

private:
    std::size_t capacity_;
    std::size_t next_ = 0;
    std::vector<Sample> samples_;
};

/// Behaviour cloning: records (what the car's camera saw, what the human did)
/// and fits the policy network to it online.
class ImitationLearner {
public:
    struct Settings {
        float learningRate = 0.05f;
        std::size_t batchSize = 5;
        std::size_t replayCapacity = 2048;
        unsigned seed = 2016;
    };

    ImitationLearner(Mlp& policy, const Settings& settings);

    /// Store one demonstration and take one training step.
    void record(std::vector<float> observation, int action);

    float averageLoss() const { return averageLoss_; }  ///< exponential moving average
    std::size_t steps() const { return steps_; }
    std::size_t samples() const { return buffer_.size(); }

private:
    Mlp& policy_;
    Settings settings_;
    ReplayBuffer buffer_;
    std::mt19937 rng_;
    float averageLoss_ = 0.0f;
    std::size_t steps_ = 0;
};

/// Save and load a policy as a small text file.
bool savePolicy(const Mlp& policy, const std::string& path);
bool loadPolicy(Mlp& policy, const std::string& path, std::string* error = nullptr);

}  // namespace pai
