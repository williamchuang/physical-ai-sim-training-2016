#include "sim/Car.h"

#include <algorithm>

namespace pai {

DriveInput DriveInput::from(Action a) {
    switch (a) {
        case Action::Faster: return {0, +1};
        case Action::Right: return {-1, 0};
        case Action::Left: return {+1, 0};
        case Action::Slower: return {0, -1};
    }
    return {};
}

std::optional<Action> DriveInput::toAction() const {
    if (throttle > 0) return Action::Faster;
    if (steer < 0) return Action::Right;
    if (steer > 0) return Action::Left;
    if (throttle < 0) return Action::Slower;
    return std::nullopt;  // nothing pressed: no demonstration to learn from
}

Car::Car(std::string name, Driver driver, double u, double v, double speedCoefficient)
    : name_(std::move(name)), driver_(driver), u_(u), v_(v), coefficient_(speedCoefficient) {}

void Car::drive(const DriveInput& input, double dt, const CarLimits& limits, double halfWidth) {
    setLane(v_ + input.steer * limits.lateralRate * dt, halfWidth);
    setSpeedCoefficient(coefficient_ + input.throttle * limits.coefficientRate * dt, limits);
}

void Car::nudgeLateral(double dv, double halfWidth) {
    setLane(v_ + dv, halfWidth);
}

void Car::setLane(double v, double halfWidth) {
    v_ = std::clamp(v, -halfWidth, halfWidth);
}

void Car::setSpeedCoefficient(double c, const CarLimits& limits) {
    coefficient_ = std::clamp(c, limits.minCoefficient, limits.maxCoefficient);
}

}  // namespace pai
