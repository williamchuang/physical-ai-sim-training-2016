#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/Vec3.h"

namespace pai {

struct Aabb {
    Vec3 min;
    Vec3 max;
};

/// Oriented bounding box: centre, three orthonormal axes and the half size along each.
struct Obb {
    Vec3 center;
    std::array<Vec3, 3> axes{Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};
    Vec3 halfExtents{1, 1, 1};

    static Obb fromFrame(const Vec3& center, const Frame& f, const Vec3& halfExtents) {
        return Obb{center, {f.right, f.up, f.forward}, halfExtents};
    }
    Aabb bounds() const;
};

/// Separating-axis test over the 15 candidate axes of two boxes.
/// On overlap returns true and, if mtd is given, the shortest translation that
/// moves box a out of box b.
bool intersects(const Obb& a, const Obb& b, Vec3* mtd = nullptr);

/// Sweep and prune: sorts boxes along x and reports every pair whose AABBs
/// overlap, in O(n log n + k) instead of testing all n^2 pairs.
std::vector<std::pair<std::size_t, std::size_t>> overlappingPairs(const std::vector<Aabb>& boxes);

}  // namespace pai
