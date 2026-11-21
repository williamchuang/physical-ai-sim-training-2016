#pragma once

#include <cstddef>

#include "sim/World.h"

namespace pai {

/// A scripted expert that presses the same four keys a person would: steer
/// towards the nearest reachable coin in view, steer away from a car close ahead,
/// otherwise hold a cruising speed. Used to demonstrate driving in training mode
/// when no one sits at the keyboard. One key at a time, so every press is a clean label.
/// It only chases coins the cube's camera could see, so the network that copies it
/// is never asked to react to something outside its own view.
struct AutopilotSettings {
    double viewDistance = 700.0;      ///< coins farther ahead than this are ignored (world units)
    double viewHalfAngle = 0.45;      ///< ... and those more than this many radians off to the side
    double deadband = 0.02;           ///< lane error ignored, in v
    double cruiseCoefficient = 10.0;  ///< speed setting to hold
    double carGap = 0.12;             ///< a car closer than this ahead (in u) ...
    double carWidth = 0.05;           ///< ... and within this across (in v) is avoided
};

DriveInput autopilot(const World& world, std::size_t carIndex, const AutopilotSettings& settings = {});

}  // namespace pai
