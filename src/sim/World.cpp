#include "sim/World.h"

#include <utility>

namespace pai {
namespace {

std::uint64_t pairKey(std::size_t a, std::size_t b) {
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

}  // namespace

World::World(MobiusStrip strip, CarLimits limits, BodySizes sizes, unsigned seed)
    : strip_(strip), limits_(limits), sizes_(sizes), rng_(seed) {}

Car& World::addCar(const std::string& name, Driver driver, double u, double v, double coefficient) {
    cars_.emplace_back(name, driver, u, strip_.clampV(v), coefficient);
    cars_.back().setSpeedCoefficient(coefficient, limits_);
    return cars_.back();
}

Coin World::randomCoin() {
    std::uniform_real_distribution<double> alongTrack(0.0, MobiusStrip::kPeriod);
    std::uniform_real_distribution<double> acrossTrack(-0.8 * strip_.halfWidth(), 0.8 * strip_.halfWidth());
    return Coin{alongTrack(rng_), acrossTrack(rng_)};
}

void World::scatterCoins(std::size_t count) {
    coins_.clear();
    coins_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) coins_.push_back(randomCoin());
}

void World::drive(std::size_t car, const DriveInput& input, double dt) {
    cars_.at(car).drive(input, dt, limits_, strip_.halfWidth());
}

Car* World::find(const std::string& name) {
    for (Car& c : cars_)
        if (c.name() == name) return &c;
    return nullptr;
}

Vec3 World::carPosition(const Car& c) const {
    return strip_.position(c.u(), c.v()) + strip_.normal(c.u(), c.v()) * sizes_.carLift;
}
Frame World::carFrame(const Car& c) const {
    return strip_.frame(c.u(), c.v());
}
Vec3 World::coinPosition(const Coin& c) const {
    return strip_.position(c.u, c.v) + strip_.normal(c.u, c.v) * sizes_.coinLift;
}
Frame World::coinFrame(const Coin& c) const {
    return strip_.frame(c.u, c.v);
}

Obb World::carBox(const Car& c) const {
    return Obb::fromFrame(carPosition(c), carFrame(c), sizes_.carHalfExtents);
}
Obb World::coinBox(const Coin& c) const {
    return Obb::fromFrame(coinPosition(c), coinFrame(c), sizes_.coinHalfExtents);
}

StepEvents World::step(double dt, bool collisions, bool respawnCoins) {
    for (Car& c : cars_) c.advance(dt);

    StepEvents events;
    if (!collisions) {
        contacts_.clear();
        return events;
    }

    // Bodies 0..cars-1 are cars, the rest are coins.
    const std::size_t nCars = cars_.size();
    std::vector<Obb> boxes;
    std::vector<Aabb> bounds;
    boxes.reserve(nCars + coins_.size());
    bounds.reserve(nCars + coins_.size());
    for (const Car& c : cars_) boxes.push_back(carBox(c));
    for (const Coin& c : coins_) boxes.push_back(coinBox(c));
    for (const Obb& b : boxes) bounds.push_back(b.bounds());

    std::unordered_set<std::uint64_t> touching;
    std::vector<bool> coinTaken(coins_.size(), false);
    for (auto [i, j] : overlappingPairs(bounds)) {
        if (i >= nCars) continue;  // coin-coin pairs never matter
        Vec3 mtd;
        if (!intersects(boxes[i], boxes[j], &mtd)) continue;

        if (j < nCars) {
            // Cars push each other apart across the track, half each.
            Car& a = cars_[i];
            Car& b = cars_[j];
            a.nudgeLateral(0.5 * strip_.lateralComponent(a.u(), a.v(), mtd), strip_.halfWidth());
            b.nudgeLateral(-0.5 * strip_.lateralComponent(b.u(), b.v(), mtd), strip_.halfWidth());
            const std::uint64_t key = pairKey(i, j);
            touching.insert(key);
            if (!contacts_.count(key)) {  // penalise a new crash once, not every frame
                --a.score;
                --b.score;
                events.crashes.push_back(i);
                events.crashes.push_back(j);
            }
        } else {
            const std::size_t coin = j - nCars;
            if (coinTaken[coin]) continue;
            coinTaken[coin] = true;
            ++cars_[i].score;
            events.pickups.push_back(i);
            if (respawnCoins) coins_[coin] = randomCoin();
        }
    }
    contacts_ = std::move(touching);
    if (!respawnCoins) {
        // Remove the collected coins: swap each with the last one, O(1) per coin.
        for (std::size_t k = coins_.size(); k-- > 0;) {
            if (!coinTaken[k]) continue;
            coins_[k] = coins_.back();
            coins_.pop_back();
        }
    }
    return events;
}

}  // namespace pai
