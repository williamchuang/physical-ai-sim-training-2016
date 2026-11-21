#include "core/MobiusStrip.h"

#include <algorithm>
#include <cmath>

namespace pai {

Vec3 MobiusStrip::position(double u, double v) const {
    const double ring = 1.0 + v * std::cos(u / 2.0);
    return Vec3{ring * std::cos(u), ring * std::sin(u), v * std::sin(u / 2.0)} * radius_;
}

Vec3 MobiusStrip::tangentU(double u, double v) const {
    const double c = std::cos(u / 2.0);
    const double s = std::sin(u / 2.0);
    const double ring = 1.0 + v * c;
    const double dRing = -0.5 * v * s;  // d(ring)/du
    return Vec3{dRing * std::cos(u) - ring * std::sin(u), dRing * std::sin(u) + ring * std::cos(u), 0.5 * v * c} *
           radius_;
}

Vec3 MobiusStrip::tangentV(double u, double /*v*/) const {
    const double c = std::cos(u / 2.0);
    return Vec3{c * std::cos(u), c * std::sin(u), std::sin(u / 2.0)} * radius_;
}

Vec3 MobiusStrip::normal(double u, double v) const {
    return tangentU(u, v).cross(tangentV(u, v)).normalized();
}

Frame MobiusStrip::frame(double u, double v) const {
    Frame f;
    f.right = tangentV(u, v).normalized();
    f.up = normal(u, v);
    f.forward = f.right.cross(f.up);  // unit, orthogonal to both, along +dX/du
    return f;
}

double MobiusStrip::lateralComponent(double u, double v, const Vec3& d) const {
    const Vec3 tv = tangentV(u, v);
    return d.dot(tv) / tv.dot(tv);
}

double MobiusStrip::wrapU(double u) const {
    const double w = std::fmod(u, kPeriod);
    return w < 0.0 ? w + kPeriod : w;
}

double MobiusStrip::clampV(double v) const {
    return std::clamp(v, -halfWidth_, halfWidth_);
}

}  // namespace pai
