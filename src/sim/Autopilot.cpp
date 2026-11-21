#include "sim/Autopilot.h"

#include <cmath>
#include <limits>

namespace pai {
namespace {

/// Distance ahead along the car's own face of the strip, in [0, 4 pi).
double ahead(double fromU, double toU) {
    const double d = std::fmod(toU - fromU, MobiusStrip::kPeriod);
    return d < 0 ? d + MobiusStrip::kPeriod : d;
}

}  // namespace

DriveInput autopilot(const World& world, std::size_t carIndex, const AutopilotSettings& s) {
    const Car& me = world.cars()[carIndex];
    const double hw = world.strip().halfWidth();
    const double speed = std::max(me.speed(), 0.05);
    double targetV = me.v();

    // 1. Avoid the nearest car close ahead in this lane.
    double nearestCar = s.carGap;
    for (std::size_t i = 0; i < world.cars().size(); ++i) {
        if (i == carIndex) continue;
        const Car& other = world.cars()[i];
        const double du = ahead(me.u(), other.u());
        if (du < nearestCar && std::abs(other.v() - me.v()) < s.carWidth) {
            nearestCar = du;
            const double away = other.v() >= me.v() ? -1.0 : 1.0;
            targetV = me.v() + away * s.carWidth;
            if (std::abs(targetV) > 0.9 * hw) targetV = me.v() - away * s.carWidth;  // no room: pass on the other side
        }
    }

    // 2. Otherwise head for the nearest coin ahead that can still be reached.
    if (nearestCar >= s.carGap) {
        double best = std::numeric_limits<double>::max();
        const Vec3 eye = world.carPosition(me);
        const Frame f = world.carFrame(me);
        for (const Coin& c : world.coins()) {
            const Vec3 d = world.coinPosition(c) - eye;
            const double forward = d.dot(f.forward);
            if (forward < 20.0 || forward > s.viewDistance) continue;                      // behind, or too far
            if (std::abs(d.dot(f.right)) > forward * std::tan(s.viewHalfAngle)) continue;  // out of view
            if (std::abs(d.dot(f.up)) > 60.0) continue;  // the other face of the strip
            const double du = forward / world.strip().radius();
            const double reach = world.limits().lateralRate * du / speed;
            if (std::abs(c.v - me.v()) > reach + 0.01) continue;
            if (forward < best) {
                best = forward;
                targetV = c.v;
            }
        }
    }

    // 3. Nothing to chase: hold the lane, but come back from the edge.
    if (targetV == me.v() && std::abs(me.v()) > 0.8 * hw) targetV = 0.5 * me.v();

    DriveInput in;
    const double error = targetV - me.v();
    if (error > s.deadband)
        in.steer = +1;
    else if (error < -s.deadband)
        in.steer = -1;
    else if (me.speedCoefficient() < s.cruiseCoefficient - 0.5)
        in.throttle = +1;
    else if (me.speedCoefficient() > s.cruiseCoefficient + 0.5)
        in.throttle = -1;
    return in;
}

}  // namespace pai
