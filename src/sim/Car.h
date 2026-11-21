#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace pai {

/// The four actions of the 2016 network, in its output order.
enum class Action : std::uint8_t { Faster = 0, Right = 1, Left = 2, Slower = 3 };
constexpr int kActionCount = 4;

/// Steering and throttle intents, each -1, 0 or +1.
struct DriveInput {
    int steer = 0;     ///< +1 towards +v (left), -1 towards -v (right)
    int throttle = 0;  ///< +1 faster, -1 slower

    static DriveInput from(Action a);
    /// The single action a demonstration is labelled with, if any key was held.
    /// Priority follows the 2016 key order: faster, right, left, slower.
    std::optional<Action> toAction() const;
};

enum class Driver { Human, Neural, Scripted };

struct CarLimits {
    double minCoefficient = 1.0;  ///< speed = 1 - 1/coefficient (rad/s along the strip)
    double maxCoefficient = 20.0;
    double coefficientRate = 5.0;  ///< coefficient change per second of throttle
    double lateralRate = 0.8;      ///< change in v per second of steering
};

/// A car is a point (u, v) on the strip plus its speed setting. Motion is
/// integrated from the frame time, so changing speed never makes a car jump.
class Car {
public:
    Car(std::string name, Driver driver, double u, double v, double speedCoefficient);

    void drive(const DriveInput& input, double dt, const CarLimits& limits, double halfWidth);
    void advance(double dt) { u_ += speed() * dt; }
    void nudgeLateral(double dv, double halfWidth);

    double speed() const { return 1.0 - 1.0 / coefficient_; }

    const std::string& name() const { return name_; }
    Driver driver() const { return driver_; }
    double u() const { return u_; }
    double v() const { return v_; }
    double speedCoefficient() const { return coefficient_; }
    void setLane(double v, double halfWidth);
    void setSpeedCoefficient(double c, const CarLimits& limits);

    int score = 0;
    DriveInput heldInput;  ///< for neural drivers: the last decision, held until the next one

private:
    std::string name_;
    Driver driver_;
    double u_;
    double v_;
    double coefficient_;
};

}  // namespace pai
