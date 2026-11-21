#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include "Check.h"
#include "game/Game.h"
#include "sim/Autopilot.h"
#include "sim/World.h"

using namespace pai;

TEST(drive_input_round_trips_actions) {
    for (int a = 0; a < kActionCount; ++a) {
        const Action action = static_cast<Action>(a);
        CHECK(DriveInput::from(action).toAction() == action);
    }
    CHECK(!DriveInput{}.toAction().has_value());
    CHECK((DriveInput{+1, +1}.toAction() == Action::Faster));  // 2016 key priority
}

TEST(car_moves_smoothly_and_respects_limits) {
    const CarLimits limits;
    Car car("c", Driver::Human, 0.0, 0.0, 10.0);
    CHECK_NEAR(car.speed(), 0.9, 1e-12);
    car.advance(0.5);
    CHECK_NEAR(car.u(), 0.45, 1e-12);

    for (int i = 0; i < 100; ++i) car.drive({+1, +1}, 0.1, limits, 0.5);
    CHECK_NEAR(car.v(), 0.5, 0);
    CHECK_NEAR(car.speedCoefficient(), limits.maxCoefficient, 0);
    for (int i = 0; i < 100; ++i) car.drive({-1, -1}, 0.1, limits, 0.5);
    CHECK_NEAR(car.v(), -0.5, 0);
    CHECK_NEAR(car.speed(), 0.0, 0);  // coefficient 1 stops the car
}

namespace {

World emptyWorld() {
    return World(MobiusStrip{}, CarLimits{}, BodySizes{}, 1);
}

}  // namespace

TEST(coin_pickup_scores_once_and_respawns) {
    World world = emptyWorld();
    world.scatterCoins(1);
    const Coin before = world.coins()[0];
    world.addCar("a", Driver::Human, before.u, before.v, 1.0);  // parked on the coin
    const StepEvents e = world.step(0.01, true, true);
    CHECK(e.pickups.size() == 1);
    CHECK(world.cars()[0].score == 1);
    CHECK(world.coins()[0].u != before.u);
}

TEST(crash_costs_one_point_per_contact) {
    World world = emptyWorld();
    world.addCar("a", Driver::Human, 2.0, 0.0, 1.0);
    world.addCar("b", Driver::Scripted, 2.0, 0.005, 1.0);  // 5 units apart, boxes 20 wide
    StepEvents e = world.step(0.01, true);
    CHECK(e.crashes.size() == 2);
    CHECK(world.cars()[0].score == -1);
    CHECK(world.cars()[1].score == -1);
    CHECK(world.cars()[0].v() < world.cars()[1].v());  // pushed apart across the track

    // Still touching or not, the same contact is not charged twice.
    world.step(0.01, true);
    CHECK(world.cars()[0].score == -1);
}

TEST(without_collisions_ignores_contact) {
    World world = emptyWorld();
    world.addCar("a", Driver::Human, 2.0, 0.0, 1.0);
    world.addCar("b", Driver::Scripted, 2.0, 0.0, 1.0);
    const StepEvents e = world.step(0.01, false);
    CHECK(e.crashes.empty());
    CHECK(world.cars()[0].score == 0);
}

namespace {

/// Synthetic perception: reports the car's lane as a bright column, so a policy
/// can learn "steer back to the centre".
class LanePerception : public Perception {
public:
    explicit LanePerception(const Game& g) : game_(g) {}
    bool observe(std::size_t car, std::vector<float>& out) override {
        const double v = game_.world().cars()[car].v();
        out.assign(9, 0.1f);
        const int col = v > 0.1 ? 0 : (v < -0.1 ? 2 : 1);
        for (int row = 0; row < 3; ++row) out[static_cast<std::size_t>(row * 3 + col)] = 0.9f;
        ++calls;
        return true;
    }
    int calls = 0;

private:
    const Game& game_;
};

/// The demonstrator: keep to the centre lane, otherwise speed up.
DriveInput centreKeeper(double v) {
    if (v > 0.1) return {-1, 0};
    if (v < -0.1) return {+1, 0};
    return {0, +1};
}

}  // namespace

TEST(game_headless_training_then_demo) {
    const auto dir = std::filesystem::temp_directory_path() / "pai_test_game";
    std::filesystem::remove_all(dir);

    // The synthetic camera has 3 x 3 pixels, as in 2016.
    GameConfig train;
    train.sensorWidth = train.sensorHeight = 3;
    train.hiddenUnits = 6;
    train.mode = Mode::Train;
    train.scriptedAgents = 0;
    train.coins = 0;
    train.roundSeconds = 240;
    train.learningRate = 0.3;
    train.policyFile = (dir / "policy.txt").string();
    Game trainer(train);
    CHECK(trainer.sensingCars().size() == 1);  // only your cube is watched while training

    LanePerception eyes(trainer);
    trainer.start();
    // Wander across the track so every lane gets demonstrations.
    double t = 0;
    while (!trainer.roundOver()) {
        const double v = trainer.world().cars()[0].v();
        DriveInput in = centreKeeper(v);
        if (std::fmod(t, 6.0) < 1.0) in = {std::fmod(t, 12.0) < 6.0 ? +1 : -1, 0};  // drift away now and then
        trainer.update(1.0 / 60, in, &eyes);
        t += 1.0 / 60;
    }
    CHECK(trainer.trainingSamples() > 1000);
    CHECK(trainer.trainingLoss() < 0.7f);  // chance is ln 4 = 1.39; drift phases keep it above 0
    CHECK(std::filesystem::exists(train.policyFile));

    GameConfig demo = train;
    demo.mode = Mode::Demo;
    demo.neuralAgents = 1;
    demo.scriptedAgents = 0;
    demo.coins = 0;
    demo.roundSeconds = 30;
    demo.policyFile = train.policyFile;
    Game game(demo);
    CHECK(!game.humanCar().has_value());  // the demo has no cube on the keyboard
    CHECK(game.policyStatus().find("loaded") != std::string::npos);
    LanePerception agentEyes(game);
    game.start();
    const std::size_t agent = *game.firstBot();
    game.world().cars()[agent].setLane(0.45, 0.5);  // start the agent near the edge
    for (int i = 0; i < 60 * 5; ++i) game.update(1.0 / 60, {}, &agentEyes);
    CHECK(std::abs(game.world().cars()[agent].v()) < 0.2);  // it steered back like the demonstrator
    CHECK(agentEyes.calls > 0);

    std::filesystem::remove_all(dir);
}

TEST(game_round_timer_and_levels) {
    GameConfig c;
    c.mode = Mode::Compare;
    c.neuralAgents = 0;
    c.scriptedAgents = 0;
    c.coins = 0;
    c.roundSeconds = 2;
    c.policyFile = "does-not-exist.txt";
    Game g(c);
    g.update(1.0, {}, nullptr);
    CHECK_NEAR(g.clock().elapsed(), 0.0, 0);  // the game opens paused
    g.start();
    for (int i = 0; i < 300; ++i) g.update(1.0 / 60, {}, nullptr);
    CHECK(g.roundOver());
    CHECK_NEAR(g.secondsLeft(), 0.0, 0);
    CHECK(g.level() == 0);
    g.world().cars()[0].score = 9;
    CHECK(g.level() == 3);  // above 2, 4 and 8
}

TEST(collected_coins_disappear_unless_respawn) {
    for (bool respawn : {false, true}) {
        World world(MobiusStrip{}, CarLimits{}, BodySizes{}, 4);
        world.scatterCoins(5);
        const Coin c = world.coins()[2];
        world.addCar("a", Driver::Human, c.u, c.v, 1.0);
        world.step(0.01, true, respawn);
        CHECK(world.cars()[0].score == 1);
        CHECK(world.coins().size() == (respawn ? 5u : 4u));
    }
}

TEST(round_ends_when_every_coin_is_taken) {
    GameConfig c;
    c.mode = Mode::Compare;
    c.neuralAgents = 0;
    c.coins = 1;
    c.roundSeconds = 600;
    c.policyFile = "does-not-exist.txt";
    Game g(c);
    g.start();
    const Coin coin = g.world().coins()[0];
    g.world().cars()[*g.humanCar()] = Car("you", Driver::Human, coin.u, coin.v, 1.0);  // parked on it
    g.update(1.0 / 60, {}, nullptr);
    CHECK(g.roundOver());
    CHECK(g.coinsLeft() == 0);
    CHECK(g.score(*g.humanCar()) == 1);
    CHECK(g.secondsLeft() > 500);
}

TEST(autopilot_collects_coins_and_avoids_cars) {
    auto run = [](bool pilot) {
        World world(MobiusStrip{}, CarLimits{}, BodySizes{}, 3);
        world.addCar("p", Driver::Human, 0.0, 0.0, 10.0);
        for (int i = 0; i < 16; ++i)
            world.addCar("s" + std::to_string(i), Driver::Scripted, 0.8 * (i + 1), -0.4 + 0.05 * i, 10.0);
        world.scatterCoins(74);
        int crashes = 0;
        for (int step = 0; step < 60 * 90; ++step) {  // a 90 s round at 60 frames per second
            if (pilot) world.drive(0, autopilot(world, 0), 1.0 / 60);
            const StepEvents e = world.step(1.0 / 60, true, true);
            for (std::size_t c : e.crashes) crashes += c == 0;
        }
        return std::make_pair(world.cars()[0].score + crashes, crashes);  // coins, crashes
    };
    const auto [idleCoins, idleCrashes] = run(false);
    const auto [pilotCoins, pilotCrashes] = run(true);
    std::cout << "        90 s round: idle " << idleCoins << " coins / " << idleCrashes << " crashes, autopilot "
              << pilotCoins << " coins / " << pilotCrashes << " crashes\n";
    CHECK(pilotCoins > 3 * std::max(idleCoins, 1));
    CHECK(pilotCrashes <= idleCrashes);
}
