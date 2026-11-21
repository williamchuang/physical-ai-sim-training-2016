#pragma once

#include "core/Vec3.h"

namespace pai {

/// The track: a Möbius strip of radius R parametrised by
///   X(u, v) = R ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2)),
/// with u the position along the track and v in [-halfWidth, halfWidth] the
/// position across it. One lap on the same side takes u from 0 to 4*pi.
class MobiusStrip {
public:
    static constexpr double kPi = 3.14159265358979323846;
    static constexpr double kPeriod = 4.0 * kPi;

    explicit MobiusStrip(double radius = 1000.0, double halfWidth = 0.5) : radius_(radius), halfWidth_(halfWidth) {}

    double radius() const { return radius_; }
    double halfWidth() const { return halfWidth_; }

    Vec3 position(double u, double v) const;
    /// dX/du: direction of travel (not normalised).
    Vec3 tangentU(double u, double v) const;
    /// dX/dv: direction across the track (not normalised).
    Vec3 tangentV(double u, double v) const;
    /// Unit surface normal, (dX/du x dX/dv) normalised.
    Vec3 normal(double u, double v) const;
    /// Orthonormal frame for an object on the strip: right across the track,
    /// up along the normal, forward along the direction of travel.
    Frame frame(double u, double v) const;

    /// Change in v produced by a small world-space displacement d at (u, v).
    double lateralComponent(double u, double v, const Vec3& d) const;

    double wrapU(double u) const;
    double clampV(double v) const;

private:
    double radius_;
    double halfWidth_;
};

}  // namespace pai
