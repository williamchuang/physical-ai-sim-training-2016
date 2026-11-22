#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "ai/ImitationLearner.h"
#include "ai/Mlp.h"
#include "Transformer.h"  // transformer/cpp
#include "core/Config.h"
#include "core/GameClock.h"
#include "sim/World.h"

namespace pai {

/// Supplies what a cube's on-board camera sees, as sensorPixels() grey values in [0, 1].
class Perception {
public:
    virtual ~Perception() = default;
    virtual bool observe(std::size_t carIndex, std::vector<float>& out) = 0;
};

/// Rules and control loop, independent of rendering.
///   Train:   keys (or the autopilot) drive your cube; (its camera image, the key) -> imitation learner.
///   Demo:    the bot's camera image -> policy -> the bot's keys.
///   Compare: as Demo, plus your cube on the keys, scored side by side.
class Game {
public:
    explicit Game(const GameConfig& config, BodySizes sizes = {});

    void update(double dt, const DriveInput& keys, Perception* perception);

    // Menu and console commands
    void start();   ///< new round from the start line
    void resume();  ///< continue a paused round
    void pause();
    void resetClock() { clock_.reset(); }
    void setCollisions(bool on) { collisions_ = on; }
    bool collisions() const { return collisions_; }
    bool savePolicyNow();

    const GameConfig& config() const { return config_; }
    World& world() { return world_; }
    const World& world() const { return world_; }
    const GameClock& clock() const { return clock_; }

    /// The cube on the keyboard (Train and Compare), if any.
    std::optional<std::size_t> humanCar() const { return humanCar_; }
    std::optional<std::size_t> firstBot() const;
    /// The cube the camera follows: yours if you drive one, otherwise the first bot.
    std::size_t focusCar() const { return humanCar_ ? *humanCar_ : firstBot().value_or(0); }
    /// Cubes whose camera the game reads: the bots, plus yours while training.
    const std::vector<std::size_t>& sensingCars() const { return sensingCars_; }

    double secondsLeft() const;
    bool roundOver() const { return roundOver_; }
    int level() const;  ///< of the followed cube: one level per threshold 2, 4, 8, ..., 128 passed
    int score(std::size_t car) const { return world_.cars()[car].score; }
    int bestBotScore() const;
    std::size_t coinsLeft() const { return world_.coins().size(); }
    float trainingLoss() const { return learner_ ? learner_->averageLoss() : 0.0f; }
    std::size_t trainingSamples() const { return learner_ ? learner_->samples() : 0; }
    const std::string& policyStatus() const { return policyStatus_; }

private:
    void populate();
    void decide(Perception* perception, const DriveInput& humanInput);
    /// Adds the newest camera frame to sensing slot `slot`'s history.
    void remember(std::size_t slot);
    /// The configured network's action for sensing slot `slot`.
    int act(std::size_t slot) const;
    void record(std::size_t slot, std::size_t car, int label, bool policyDrove);

    GameConfig config_;
    BodySizes sizes_;
    World world_;
    GameClock clock_;
    Mlp policy_;
    std::unique_ptr<ImitationLearner> learner_;
    std::unique_ptr<Transformer> transformer_;
    /// Per sensing car: the last historyFrames camera frames, oldest first.
    std::vector<std::vector<float>> histories_;
    std::vector<bool> historyStarted_;
    std::ofstream recording_;
    int episode_ = -1;
    std::optional<std::size_t> humanCar_;
    std::vector<std::size_t> sensingCars_;
    std::vector<float> observation_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> daggerCoin_{0.0, 1.0};
    bool policyDrives_ = false;
    double decisionTimer_ = 0.0;
    double autosaveTimer_ = 0.0;
    bool collisions_ = true;
    bool roundOver_ = false;
    std::string policyStatus_;
};

}  // namespace pai
