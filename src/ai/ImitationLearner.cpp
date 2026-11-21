#include "ai/ImitationLearner.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace pai {

void ReplayBuffer::push(Sample s) {
    if (samples_.size() < capacity_) {
        samples_.push_back(std::move(s));
    } else {
        samples_[next_] = std::move(s);
    }
    next_ = (next_ + 1) % capacity_;
}

std::vector<const Sample*> ReplayBuffer::minibatch(std::size_t n, std::mt19937& rng) const {
    // Floyd's algorithm: n distinct indices out of size() in O(n^2) time and O(n)
    // memory, independent of the buffer size (n is a handful, the buffer thousands).
    const std::size_t total = samples_.size();
    n = std::min(n, total);
    std::vector<std::size_t> chosen;
    chosen.reserve(n);
    for (std::size_t j = total - n; j < total; ++j) {
        const std::size_t t = std::uniform_int_distribution<std::size_t>(0, j)(rng);
        const bool seen = std::find(chosen.begin(), chosen.end(), t) != chosen.end();
        chosen.push_back(seen ? j : t);
    }
    std::vector<const Sample*> batch;
    batch.reserve(n);
    for (std::size_t i : chosen) batch.push_back(&samples_[i]);
    return batch;
}

ImitationLearner::ImitationLearner(Mlp& policy, const Settings& settings)
    : policy_(policy), settings_(settings), buffer_(settings.replayCapacity), rng_(settings.seed) {}

void ImitationLearner::record(std::vector<float> observation, int action) {
    buffer_.push(Sample{std::move(observation), action});
    const float loss = policy_.trainBatch(buffer_.minibatch(settings_.batchSize, rng_), settings_.learningRate);
    averageLoss_ = steps_ == 0 ? loss : 0.98f * averageLoss_ + 0.02f * loss;
    ++steps_;
}

bool savePolicy(const Mlp& policy, const std::string& path) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    const std::filesystem::path tmp = p.string() + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) return false;
        policy.save(out);
        if (!out) return false;
    }
    std::filesystem::rename(tmp, p);  // never leaves a half-written policy behind
    return true;
}

bool loadPolicy(Mlp& policy, const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "no policy file at " + path;
        return false;
    }
    try {
        policy.load(in);
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

}  // namespace pai
