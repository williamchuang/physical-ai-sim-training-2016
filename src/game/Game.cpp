#include "game/Game.h"

#include <algorithm>
#include <array>
#include <cmath>

#include "sim/Autopilot.h"

namespace pai {
namespace {

constexpr double kPhysicsRate = 60.0;       // physics steps per second, at least
constexpr double kStartCoefficient = 10.0;  // speed 0.9 rad/s, as in 2016
constexpr std::array<double, 7> kLevelThresholds{2, 4, 8, 16, 32, 64, 128};
// Starting lanes from 2016: the cube you drive, then bots 1..3.
constexpr double kHumanLane = 0.3;
constexpr std::array<double, 3> kBotLanes{0.0, 0.45, -0.2};

}  // namespace

Game::Game(const GameConfig& config, BodySizes sizes)
    : config_(config),
      sizes_(sizes),
      world_(MobiusStrip{}, CarLimits{}, sizes, config.seed),
      policy_(static_cast<std::size_t>(config.sensorPixels()), static_cast<std::size_t>(config.hiddenUnits),
              kActionCount, config.seed),
      observation_(static_cast<std::size_t>(config.sensorPixels())),
      rng_(config.seed + 1),
      collisions_(config.collisions) {
    std::string error;
    if (loadPolicy(policy_, config_.policyFile, &error)) {
        policyStatus_ = "policy loaded from " + config_.policyFile;
    } else {
        policyStatus_ = "untrained policy (" + error + ")";
    }
    if (config_.mode == Mode::Train) {
        ImitationLearner::Settings s;
        s.learningRate = static_cast<float>(config_.learningRate);
        s.batchSize = static_cast<std::size_t>(config_.batchSize);
        s.replayCapacity = static_cast<std::size_t>(config_.replayCapacity);
        s.seed = config_.seed;
        learner_ = std::make_unique<ImitationLearner>(policy_, s);
    }
    populate();
}

void Game::populate() {
    world_ = World(MobiusStrip{}, CarLimits{}, sizes_, config_.seed);
    humanCar_.reset();
    if (config_.mode != Mode::Demo) {
        humanCar_ = world_.cars().size();
        world_.addCar("you", Driver::Human, 0.0, kHumanLane, kStartCoefficient);
    }

    const int bots = config_.mode == Mode::Train ? 0 : config_.neuralAgents;
    for (int i = 0; i < bots; ++i) {
        const double lane = i < static_cast<int>(kBotLanes.size()) ? kBotLanes[i] : 0.15 * (i % 5 - 2);
        world_.addCar("bot" + std::to_string(i + 1), Driver::Neural, -0.05 * (i + 1), lane, kStartCoefficient);
    }

    // Scripted cubes keep their lane at a constant speed, spread around the track: obstacles.
    const int scripted = config_.scriptedAgents;
    const double halfWidth = world_.strip().halfWidth();
    for (int i = 0; i < scripted; ++i) {
        const double lane = scripted > 1 ? -0.8 * halfWidth + 1.6 * halfWidth * i / (scripted - 1) : 0.0;
        const double u = MobiusStrip::kPeriod * (i + 1) / (scripted + 1);
        world_.addCar("cube" + std::to_string(i + 1), Driver::Scripted, u, lane, kStartCoefficient);
    }
    world_.scatterCoins(static_cast<std::size_t>(config_.coins));

    sensingCars_.clear();
    for (std::size_t i = 0; i < world_.cars().size(); ++i) {
        const Driver d = world_.cars()[i].driver();
        if (d == Driver::Neural || (d == Driver::Human && config_.mode == Mode::Train)) sensingCars_.push_back(i);
    }
    decisionTimer_ = 0.0;
    roundOver_ = false;
}

void Game::start() {
    populate();
    clock_.reset();
    clock_.resume();
}

void Game::resume() {
    if (!roundOver_) clock_.resume();
}

void Game::pause() {
    clock_.pause();
}

bool Game::savePolicyNow() {
    const bool ok = savePolicy(policy_, config_.policyFile);
    policyStatus_ = ok ? "policy saved to " + config_.policyFile : "could not write " + config_.policyFile;
    return ok;
}

std::optional<std::size_t> Game::firstBot() const {
    for (std::size_t i = 0; i < world_.cars().size(); ++i)
        if (world_.cars()[i].driver() == Driver::Neural) return i;
    return std::nullopt;
}

void Game::decide(Perception* perception, const DriveInput& humanInput) {
    if (!perception) return;
    for (std::size_t index : sensingCars_) {
        if (!perception->observe(index, observation_)) continue;
        Car& car = world_.cars()[index];
        if (car.driver() == Driver::Neural) {
            car.heldInput = DriveInput::from(static_cast<Action>(policy_.predict(observation_)));
        } else if (learner_) {
            // A frame with no key pressed carries no label, so it is not learned from.
            if (auto action = humanInput.toAction()) learner_->record(observation_, static_cast<int>(*action));
            // DAgger: until the next decision, the network may hold the wheel.
            policyDrives_ = config_.demonstrator == Demonstrator::Autopilot && daggerCoin_(rng_) < config_.daggerShare;
            if (policyDrives_) car.heldInput = DriveInput::from(static_cast<Action>(policy_.predict(observation_)));
        }
    }
}

void Game::update(double dt, const DriveInput& keys, Perception* perception) {
    if (!clock_.running() || roundOver_) return;
    clock_.tick(dt);

    DriveInput humanInput = keys;
    if (humanCar_ && config_.mode == Mode::Train && config_.demonstrator == Demonstrator::Autopilot)
        humanInput = autopilot(world_, *humanCar_);
    decisionTimer_ += dt;
    if (decisionTimer_ >= config_.decisionInterval) {
        decisionTimer_ = std::min(decisionTimer_ - config_.decisionInterval, config_.decisionInterval);
        decide(perception, humanInput);
    }
    // Physics in sub-steps of at most 1/60 s, so a slow frame cannot carry a cube
    // past a coin or another cube between two collision checks.
    const int substeps = std::max(1, static_cast<int>(std::ceil(dt * kPhysicsRate)));
    const double h = dt / substeps;
    for (int k = 0; k < substeps; ++k) {
        if (humanCar_) world_.drive(*humanCar_, policyDrives_ ? world_.cars()[*humanCar_].heldInput : humanInput, h);
        for (std::size_t i = 0; i < world_.cars().size(); ++i)
            if (world_.cars()[i].driver() == Driver::Neural) world_.drive(i, world_.cars()[i].heldInput, h);
        world_.step(h, collisions_, config_.respawnCoins);
    }

    if (learner_) {
        autosaveTimer_ += dt;
        if (config_.autosaveSeconds > 0 && autosaveTimer_ >= config_.autosaveSeconds) {
            autosaveTimer_ = 0.0;
            savePolicyNow();
        }
    }
    if (clock_.elapsed() >= config_.roundSeconds || (config_.coins > 0 && world_.coins().empty())) {
        roundOver_ = true;
        clock_.pause();
        if (learner_) savePolicyNow();
    }
}

double Game::secondsLeft() const {
    return std::max(0.0, config_.roundSeconds - clock_.elapsed());
}

int Game::level() const {
    const int s = score(focusCar());
    return static_cast<int>(
        std::count_if(kLevelThresholds.begin(), kLevelThresholds.end(), [s](double t) { return s > t; }));
}

int Game::bestBotScore() const {
    int best = 0;
    bool any = false;
    for (const Car& c : world_.cars()) {
        if (c.driver() != Driver::Neural) continue;
        best = any ? std::max(best, c.score) : c.score;
        any = true;
    }
    return best;
}

}  // namespace pai
