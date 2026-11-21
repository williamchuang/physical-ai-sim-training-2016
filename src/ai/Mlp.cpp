#include "ai/Mlp.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace pai {
namespace {

float sigmoid(float x) {
    // split form avoids overflow in exp for large |x|
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

void softmaxInPlace(float* z, std::size_t n) {
    const float maxZ = *std::max_element(z, z + n);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        z[i] = std::exp(z[i] - maxZ);
        sum += z[i];
    }
    for (std::size_t i = 0; i < n; ++i) z[i] /= sum;
}

void glorotUniform(std::vector<float>& w, std::size_t fanIn, std::size_t fanOut, std::mt19937& rng) {
    const float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& x : w) x = dist(rng);
}

void writeArray(std::ostream& out, const std::vector<float>& a) {
    for (std::size_t i = 0; i < a.size(); ++i) out << (i ? " " : "") << a[i];
    out << '\n';
}

void readArray(std::istream& in, std::vector<float>& a) {
    for (float& x : a)
        if (!(in >> x)) throw std::runtime_error("policy file: not enough weights");
}

}  // namespace

Mlp::Mlp(std::size_t inputs, std::size_t hidden, std::size_t outputs, unsigned seed)
    : inputs_(inputs),
      hidden_(hidden),
      outputs_(outputs),
      w1_(hidden * inputs),
      b1_(hidden, 0.0f),
      w2_(outputs * hidden),
      b2_(outputs, 0.0f) {
    if (inputs == 0 || hidden == 0 || outputs < 2) throw std::invalid_argument("Mlp: bad layer sizes");
    std::mt19937 rng(seed);
    glorotUniform(w1_, inputs, hidden, rng);
    glorotUniform(w2_, hidden, outputs, rng);
}

void Mlp::forward(const float* input, float* hiddenOut, float* probsOut) const {
    for (std::size_t h = 0; h < hidden_; ++h) {
        const float* row = &w1_[h * inputs_];
        float a = b1_[h];
        for (std::size_t i = 0; i < inputs_; ++i) a += row[i] * input[i];
        hiddenOut[h] = sigmoid(a);
    }
    for (std::size_t o = 0; o < outputs_; ++o) {
        const float* row = &w2_[o * hidden_];
        float z = b2_[o];
        for (std::size_t h = 0; h < hidden_; ++h) z += row[h] * hiddenOut[h];
        probsOut[o] = z;
    }
    softmaxInPlace(probsOut, outputs_);
}

std::vector<float> Mlp::probabilities(const std::vector<float>& input) const {
    if (input.size() != inputs_) throw std::invalid_argument("Mlp: input has the wrong size");
    std::vector<float> hiddenAct(hidden_), probs(outputs_);
    forward(input.data(), hiddenAct.data(), probs.data());
    return probs;
}

int Mlp::predict(const std::vector<float>& input) const {
    const std::vector<float> p = probabilities(input);
    return static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
}

float Mlp::trainBatch(const std::vector<const Sample*>& batch, float learningRate) {
    if (batch.empty()) return 0.0f;
    std::vector<float> gw1(w1_.size(), 0.0f), gb1(b1_.size(), 0.0f);
    std::vector<float> gw2(w2_.size(), 0.0f), gb2(b2_.size(), 0.0f);
    std::vector<float> hiddenAct(hidden_), probs(outputs_), dHidden(hidden_);
    float loss = 0.0f;

    for (const Sample* s : batch) {
        if (s->input.size() != inputs_) throw std::invalid_argument("Mlp: sample has the wrong size");
        if (s->label < 0 || static_cast<std::size_t>(s->label) >= outputs_)
            throw std::invalid_argument("Mlp: label out of range");
        forward(s->input.data(), hiddenAct.data(), probs.data());
        loss -= std::log(std::max(probs[s->label], 1e-12f));

        // softmax + cross-entropy: dL/dz = p - onehot(label)
        probs[s->label] -= 1.0f;
        std::fill(dHidden.begin(), dHidden.end(), 0.0f);
        for (std::size_t o = 0; o < outputs_; ++o) {
            const float dz = probs[o];
            gb2[o] += dz;
            float* gRow = &gw2[o * hidden_];
            const float* wRow = &w2_[o * hidden_];
            for (std::size_t h = 0; h < hidden_; ++h) {
                gRow[h] += dz * hiddenAct[h];
                dHidden[h] += dz * wRow[h];
            }
        }
        for (std::size_t h = 0; h < hidden_; ++h) {
            const float da = dHidden[h] * hiddenAct[h] * (1.0f - hiddenAct[h]);  // sigmoid'
            gb1[h] += da;
            float* gRow = &gw1[h * inputs_];
            for (std::size_t i = 0; i < inputs_; ++i) gRow[i] += da * s->input[i];
        }
    }

    const float step = learningRate / static_cast<float>(batch.size());
    auto apply = [step](std::vector<float>& w, const std::vector<float>& g) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] -= step * g[i];
    };
    apply(w1_, gw1);
    apply(b1_, gb1);
    apply(w2_, gw2);
    apply(b2_, gb2);
    return loss / static_cast<float>(batch.size());
}

void Mlp::save(std::ostream& out) const {
    out << "mlp 1 " << inputs_ << ' ' << hidden_ << ' ' << outputs_ << '\n' << std::setprecision(9);
    writeArray(out, w1_);
    writeArray(out, b1_);
    writeArray(out, w2_);
    writeArray(out, b2_);
}

void Mlp::load(std::istream& in) {
    std::string tag;
    int version = 0;
    std::size_t i = 0, h = 0, o = 0;
    if (!(in >> tag >> version >> i >> h >> o) || tag != "mlp" || version != 1)
        throw std::runtime_error("policy file: missing 'mlp 1' header");
    if (i != inputs_ || h != hidden_ || o != outputs_)
        throw std::runtime_error("policy file: network shape differs from the configuration");
    readArray(in, w1_);
    readArray(in, b1_);
    readArray(in, w2_);
    readArray(in, b2_);
}

}  // namespace pai
