#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

#include "Check.h"
#include "ai/ImitationLearner.h"
#include "ai/Mlp.h"

using namespace pai;

namespace {

/// A stand-in for camera images: the brightest of four regions names the action.
Sample syntheticFrame(std::mt19937& rng) {
    std::uniform_real_distribution<float> noise(0.0f, 0.3f);
    std::uniform_int_distribution<int> pick(0, 3);
    Sample s;
    s.input.resize(9);
    for (float& x : s.input) x = noise(rng);
    s.label = pick(rng);
    static const int bright[4][2] = {{1, 4}, {5, 8}, {3, 6}, {0, 2}};
    for (int p : bright[s.label]) s.input[static_cast<std::size_t>(p)] += 0.7f;
    return s;
}

double accuracy(const Mlp& net, std::mt19937& rng, int n) {
    int right = 0;
    for (int i = 0; i < n; ++i) {
        const Sample s = syntheticFrame(rng);
        right += net.predict(s.input) == s.label;
    }
    return double(right) / n;
}

}  // namespace

TEST(mlp_outputs_a_distribution) {
    const Mlp net(9, 6, 4, 1);
    const auto p = net.probabilities(std::vector<float>(9, 0.5f));
    CHECK(p.size() == 4);
    CHECK_NEAR(std::accumulate(p.begin(), p.end(), 0.0), 1.0, 1e-5);
    for (float x : p) CHECK(x > 0);
}

TEST(mlp_training_lowers_loss) {
    Mlp net(9, 6, 4, 3);
    std::mt19937 rng(11);
    std::vector<Sample> data;
    for (int i = 0; i < 64; ++i) data.push_back(syntheticFrame(rng));
    std::vector<const Sample*> batch;
    for (const Sample& s : data) batch.push_back(&s);
    const float first = net.trainBatch(batch, 0.5f);
    float last = first;
    for (int i = 0; i < 300; ++i) last = net.trainBatch(batch, 0.5f);
    CHECK(last < first * 0.5f);
}

TEST(mlp_save_load_round_trip) {
    Mlp a(9, 6, 4, 5);
    std::stringstream buf;
    a.save(buf);
    Mlp b(9, 6, 4, 99);
    b.load(buf);
    const std::vector<float> x{0.1f, 0.9f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.2f, 0.0f};
    const auto pa = a.probabilities(x), pb = b.probabilities(x);
    for (std::size_t i = 0; i < pa.size(); ++i) CHECK_NEAR(pa[i], pb[i], 1e-6);
}

TEST(mlp_load_rejects_other_shapes) {
    Mlp a(9, 6, 4, 5);
    std::stringstream buf;
    a.save(buf);
    Mlp b(16, 6, 4, 5);
    bool threw = false;
    try {
        b.load(buf);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(replay_buffer_overwrites_oldest) {
    ReplayBuffer buf(3);
    for (int i = 0; i < 5; ++i) buf.push(Sample{{float(i)}, i});
    CHECK(buf.size() == 3);
    std::mt19937 rng(1);
    std::set<int> labels;
    for (const Sample* s : buf.minibatch(3, rng)) labels.insert(s->label);
    CHECK((labels == std::set<int>{2, 3, 4}));
}

TEST(replay_minibatch_is_distinct_and_uniform) {
    ReplayBuffer buf(50);
    for (int i = 0; i < 50; ++i) buf.push(Sample{{}, i});
    std::mt19937 rng(2);
    std::vector<int> hits(50, 0);
    for (int round = 0; round < 4000; ++round) {
        std::set<int> seen;
        for (const Sample* s : buf.minibatch(5, rng)) {
            CHECK(seen.insert(s->label).second);
            ++hits[static_cast<std::size_t>(s->label)];
        }
    }
    for (int h : hits) CHECK(h > 300 && h < 500);  // expected 400 each
}

TEST(imitation_learner_clones_the_demonstrator) {
    Mlp net(9, 6, 4, 2016);
    ImitationLearner::Settings settings;
    settings.learningRate = 0.3f;
    ImitationLearner learner(net, settings);
    std::mt19937 rng(42);
    const double before = accuracy(net, rng, 500);
    for (int i = 0; i < 6000; ++i) {
        Sample s = syntheticFrame(rng);
        learner.record(std::move(s.input), s.label);
    }
    const double after = accuracy(net, rng, 1000);
    CHECK(before < 0.5);
    CHECK(after > 0.95);
    CHECK(learner.steps() == 6000);
}

TEST(policy_file_round_trip) {
    const auto path = (std::filesystem::temp_directory_path() / "pai_test_models" / "policy.txt").string();
    Mlp a(9, 6, 4, 8);
    CHECK(savePolicy(a, path));
    Mlp b(9, 6, 4, 9);
    std::string error;
    CHECK(loadPolicy(b, path, &error));
    CHECK(!loadPolicy(b, path + ".missing", &error));
    CHECK(!error.empty());
    std::filesystem::remove_all(std::filesystem::path(path).parent_path());
}

namespace {

/// All parameters of a network, in save() order (w1, b1, w2, b2).
std::vector<double> parameters(const Mlp& net) {
    std::stringstream buf;
    net.save(buf);
    std::string tag;
    int version;
    std::size_t i, h, o;
    buf >> tag >> version >> i >> h >> o;
    std::vector<double> p;
    for (double x; buf >> x;) p.push_back(x);
    return p;
}

void setParameters(Mlp& net, const std::vector<double>& p) {
    std::stringstream buf;
    buf << "mlp 1 " << net.inputs() << ' ' << net.hidden() << ' ' << net.outputs() << '\n';
    buf.precision(17);
    for (double x : p) buf << x << ' ';
    net.load(buf);
}

}  // namespace

// Backpropagation check: the step trainBatch takes must equal -lr times the
// gradient measured by central finite differences, for every parameter.
TEST(mlp_backprop_matches_finite_differences) {
    std::mt19937 rng(5);
    std::vector<Sample> data;
    for (int i = 0; i < 8; ++i) data.push_back(syntheticFrame(rng));
    std::vector<const Sample*> batch;
    for (const Sample& s : data) batch.push_back(&s);

    Mlp net(9, 6, 4, 21);
    const std::vector<double> theta = parameters(net);
    const float lr = 1.0f;  // the step is lr times the gradient at theta; a large lr keeps float rounding small
    Mlp stepped = net;
    stepped.trainBatch(batch, lr);
    const std::vector<double> after = parameters(stepped);
    CHECK(after.size() == theta.size());

    const double eps = 1e-2;
    double worst = 0;
    for (std::size_t k = 0; k < theta.size(); ++k) {
        std::vector<double> plus = theta, minus = theta;
        plus[k] += eps;
        minus[k] -= eps;
        Mlp a = net, b = net;
        setParameters(a, plus);
        setParameters(b, minus);
        const double numeric = (a.trainBatch(batch, 0.0f) - b.trainBatch(batch, 0.0f)) / (2 * eps);
        const double analytic = (theta[k] - after[k]) / lr;
        worst = std::max(worst, std::abs(numeric - analytic) / std::max(1e-3, std::abs(numeric) + std::abs(analytic)));
    }
    CHECK(worst < 1e-2);  // float arithmetic; a wrong gradient gives errors near 1
    std::cout << "        largest relative gradient error " << worst << " over " << theta.size() << " parameters\n";
}
