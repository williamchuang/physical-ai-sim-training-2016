#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/MobiusStrip.h"
#include "physics/Collision.h"
#include "sim/Car.h"

namespace pai {

struct Coin {
    double u = 0.0;
    double v = 0.0;
};

/// Physical size of the bodies, taken from the meshes by the renderer.
struct BodySizes {
    Vec3 carHalfExtents{10, 10, 10};
    double carLift = 10.0;  ///< height of a car's centre above the track
    Vec3 coinHalfExtents{12, 12, 3};
    double coinLift = 12.0;
};

struct StepEvents {
    std::vector<std::size_t> pickups;  ///< cars that collected a coin this step
    std::vector<std::size_t> crashes;  ///< cars that started touching another car
};

/// The cars and coins on the Möbius strip, and the rules that move them.
class World {
public:
    World(MobiusStrip strip, CarLimits limits, BodySizes sizes, unsigned seed);

    Car& addCar(const std::string& name, Driver driver, double u, double v, double coefficient);
    void scatterCoins(std::size_t count);

    /// Advance every car by dt, then resolve car-car pushes and coin pickups.
    /// A collected coin reappears elsewhere if respawnCoins, otherwise it is gone.
    StepEvents step(double dt, bool collisions, bool respawnCoins = false);

    void drive(std::size_t car, const DriveInput& input, double dt);

    std::vector<Car>& cars() { return cars_; }
    const std::vector<Car>& cars() const { return cars_; }
    const std::vector<Coin>& coins() const { return coins_; }
    const MobiusStrip& strip() const { return strip_; }
    const CarLimits& limits() const { return limits_; }
    const BodySizes& sizes() const { return sizes_; }
    void setSizes(const BodySizes& sizes) { sizes_ = sizes; }

    Car* find(const std::string& name);

    // World-space placement, shared by physics and rendering.
    Vec3 carPosition(const Car& c) const;
    Frame carFrame(const Car& c) const;
    Vec3 coinPosition(const Coin& c) const;
    Frame coinFrame(const Coin& c) const;

private:
    Obb carBox(const Car& c) const;
    Obb coinBox(const Coin& c) const;
    Coin randomCoin();

    MobiusStrip strip_;
    CarLimits limits_;
    BodySizes sizes_;
    std::mt19937 rng_;
    std::vector<Car> cars_;
    std::vector<Coin> coins_;
    std::unordered_set<std::uint64_t> contacts_;  ///< car pairs touching last step
};

}  // namespace pai
