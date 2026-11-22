#include "Transformer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace pai {
namespace {

constexpr float kLayerNormEpsilon = 1e-5f;

std::vector<float> readTensor(std::istream& in, const std::string& name, std::size_t count) {
    std::string tag;
    std::size_t n = 0;
    if (!(in >> tag >> n)) throw std::runtime_error("transformer file: missing tensor " + name);
    if (tag != name) throw std::runtime_error("transformer file: expected " + name + ", found " + tag);
    if (n != count)
        throw std::runtime_error("transformer file: " + name + " has " + std::to_string(n) + " values, expected " +
                                 std::to_string(count));
    std::vector<float> v(count);
    for (float& x : v)
        if (!(in >> x)) throw std::runtime_error("transformer file: not enough values in " + name);
    return v;
}

// y[r] = b[r] + sum_c W[r][c] x[c]   (W is rows x cols, row-major, as PyTorch stores Linear weights)
void linear(const std::vector<float>& w, const std::vector<float>& b, const float* x, std::size_t cols, float* y) {
    const std::size_t rows = b.size();
    for (std::size_t r = 0; r < rows; ++r) {
        const float* wr = w.data() + r * cols;
        float s = b[r];
        for (std::size_t c = 0; c < cols; ++c) s += wr[c] * x[c];
        y[r] = s;
    }
}

void layerNorm(const float* x, std::size_t n, const std::vector<float>& gain, const std::vector<float>& bias,
               float* y) {
    float mean = 0.0f;
    for (std::size_t i = 0; i < n; ++i) mean += x[i];
    mean /= static_cast<float>(n);
    float var = 0.0f;
    for (std::size_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= static_cast<float>(n);
    const float inv = 1.0f / std::sqrt(var + kLayerNormEpsilon);
    for (std::size_t i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * gain[i] + bias[i];
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

}  // namespace

Transformer::Transformer(std::istream& in) {
    std::string tag;
    int version = 0;
    Shape& s = shape_;
    if (!(in >> tag >> version >> s.frames >> s.height >> s.width >> s.patch >> s.model >> s.heads >> s.layers >>
          s.feedForward >> s.outputs) ||
        tag != "transformer" || version != 1)
        throw std::runtime_error("transformer file: bad header");
    if (s.patch == 0 || s.height % s.patch || s.width % s.patch || s.heads == 0 || s.model % s.heads || s.frames == 0 ||
        s.outputs == 0)
        throw std::runtime_error("transformer file: inconsistent shape");

    const std::size_t d = s.model;
    patchW_ = readTensor(in, "patch.weight", d * s.patchPixels());
    patchB_ = readTensor(in, "patch.bias", d);
    position_ = readTensor(in, "position", s.tokens() * d);
    for (std::size_t l = 0; l < s.layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        Layer L;
        L.ln1Gain = readTensor(in, p + "ln1.weight", d);
        L.ln1Bias = readTensor(in, p + "ln1.bias", d);
        L.qkvW = readTensor(in, p + "qkv.weight", 3 * d * d);
        L.qkvB = readTensor(in, p + "qkv.bias", 3 * d);
        L.outW = readTensor(in, p + "out.weight", d * d);
        L.outB = readTensor(in, p + "out.bias", d);
        L.ln2Gain = readTensor(in, p + "ln2.weight", d);
        L.ln2Bias = readTensor(in, p + "ln2.bias", d);
        L.ff1W = readTensor(in, p + "ff1.weight", s.feedForward * d);
        L.ff1B = readTensor(in, p + "ff1.bias", s.feedForward);
        L.ff2W = readTensor(in, p + "ff2.weight", d * s.feedForward);
        L.ff2B = readTensor(in, p + "ff2.bias", d);
        layers_.push_back(std::move(L));
    }
    lnGain_ = readTensor(in, "norm.weight", d);
    lnBias_ = readTensor(in, "norm.bias", d);
    headW_ = readTensor(in, "head.weight", s.outputs * d);
    headB_ = readTensor(in, "head.bias", s.outputs);
}

std::vector<float> Transformer::logits(const std::vector<float>& input) const {
    const Shape& s = shape_;
    if (input.size() != s.inputs()) throw std::invalid_argument("transformer: wrong input size");
    const std::size_t d = s.model, T = s.tokens(), P = s.patch, cols = s.width / P;
    const std::size_t heads = s.heads, dh = d / heads;

    // Tokens: one per patch, frames oldest first; pixels inside a patch row by row.
    std::vector<float> x(T * d), pixels(s.patchPixels());
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t f = t / s.patchesPerFrame(), k = t % s.patchesPerFrame();
        const std::size_t row0 = (k / cols) * P, col0 = (k % cols) * P;
        const float* image = input.data() + f * s.height * s.width;
        for (std::size_t dy = 0; dy < P; ++dy)
            for (std::size_t dx = 0; dx < P; ++dx) pixels[dy * P + dx] = image[(row0 + dy) * s.width + col0 + dx];
        linear(patchW_, patchB_, pixels.data(), s.patchPixels(), &x[t * d]);
        for (std::size_t i = 0; i < d; ++i) x[t * d + i] += position_[t * d + i];
    }

    std::vector<float> h(T * d), qkv(T * 3 * d), mixed(T * d), scores(T), proj(d), hidden(s.feedForward);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (const Layer& L : layers_) {
        // Attention block.
        for (std::size_t t = 0; t < T; ++t) {
            layerNorm(&x[t * d], d, L.ln1Gain, L.ln1Bias, &h[t * d]);
            linear(L.qkvW, L.qkvB, &h[t * d], d, &qkv[t * 3 * d]);
        }
        for (std::size_t hd = 0; hd < heads; ++hd) {
            for (std::size_t t = 0; t < T; ++t) {
                const float* q = &qkv[t * 3 * d + hd * dh];
                float best = -1e30f;
                for (std::size_t u = 0; u < T; ++u) {
                    const float* k = &qkv[u * 3 * d + d + hd * dh];
                    float dot = 0.0f;
                    for (std::size_t i = 0; i < dh; ++i) dot += q[i] * k[i];
                    scores[u] = dot * scale;
                    best = std::max(best, scores[u]);
                }
                float total = 0.0f;
                for (std::size_t u = 0; u < T; ++u) total += scores[u] = std::exp(scores[u] - best);
                float* out = &mixed[t * d + hd * dh];
                std::fill(out, out + dh, 0.0f);
                for (std::size_t u = 0; u < T; ++u) {
                    const float* v = &qkv[u * 3 * d + 2 * d + hd * dh];
                    const float a = scores[u] / total;
                    for (std::size_t i = 0; i < dh; ++i) out[i] += a * v[i];
                }
            }
        }
        for (std::size_t t = 0; t < T; ++t) {
            linear(L.outW, L.outB, &mixed[t * d], d, proj.data());
            for (std::size_t i = 0; i < d; ++i) x[t * d + i] += proj[i];
        }
        // Feed-forward block.
        for (std::size_t t = 0; t < T; ++t) {
            layerNorm(&x[t * d], d, L.ln2Gain, L.ln2Bias, h.data());
            linear(L.ff1W, L.ff1B, h.data(), d, hidden.data());
            for (float& v : hidden) v = gelu(v);
            linear(L.ff2W, L.ff2B, hidden.data(), s.feedForward, proj.data());
            for (std::size_t i = 0; i < d; ++i) x[t * d + i] += proj[i];
        }
    }

    std::vector<float> pooled(d, 0.0f);
    for (std::size_t t = 0; t < T; ++t) {
        layerNorm(&x[t * d], d, lnGain_, lnBias_, h.data());
        for (std::size_t i = 0; i < d; ++i) pooled[i] += h[i] / static_cast<float>(T);
    }
    std::vector<float> out(s.outputs);
    linear(headW_, headB_, pooled.data(), d, out.data());
    return out;
}

int Transformer::predict(const std::vector<float>& input) const {
    const std::vector<float> z = logits(input);
    return static_cast<int>(std::max_element(z.begin(), z.end()) - z.begin());
}

std::unique_ptr<Transformer> loadTransformer(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "cannot open " + path;
        return nullptr;
    }
    try {
        return std::make_unique<Transformer>(in);
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return nullptr;
    }
}

}  // namespace pai
