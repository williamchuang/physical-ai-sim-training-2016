#include "physics/Collision.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace pai {
namespace {

constexpr double kParallelEpsilon = 1e-9;

double halfComponent(const Obb& box, int k) {
    // extent of the box along world axis k
    const double* h = &box.halfExtents.x;
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double* axis = &box.axes[i].x;
        sum += std::abs(axis[k]) * h[i];
    }
    return sum;
}

double projectedRadius(const Obb& box, const Vec3& axis) {
    return box.halfExtents.x * std::abs(box.axes[0].dot(axis)) + box.halfExtents.y * std::abs(box.axes[1].dot(axis)) +
           box.halfExtents.z * std::abs(box.axes[2].dot(axis));
}

}  // namespace

Aabb Obb::bounds() const {
    const Vec3 e{halfComponent(*this, 0), halfComponent(*this, 1), halfComponent(*this, 2)};
    return {center - e, center + e};
}

bool intersects(const Obb& a, const Obb& b, Vec3* mtd) {
    const Vec3 between = b.center - a.center;
    double bestOverlap = std::numeric_limits<double>::infinity();
    Vec3 bestAxis;

    auto testAxis = [&](Vec3 axis) {
        const double len = axis.length();
        if (len < kParallelEpsilon) return true;  // degenerate cross product: skip
        axis = axis / len;
        const double distance = between.dot(axis);
        const double overlap = projectedRadius(a, axis) + projectedRadius(b, axis) - std::abs(distance);
        if (overlap <= 0.0) return false;
        if (overlap < bestOverlap) {
            bestOverlap = overlap;
            bestAxis = distance > 0.0 ? -axis : axis;  // point from b towards a
        }
        return true;
    };

    for (const Vec3& axis : a.axes)
        if (!testAxis(axis)) return false;
    for (const Vec3& axis : b.axes)
        if (!testAxis(axis)) return false;
    for (const Vec3& ea : a.axes)
        for (const Vec3& eb : b.axes)
            if (!testAxis(ea.cross(eb))) return false;

    if (mtd) *mtd = bestAxis * bestOverlap;
    return true;
}

std::vector<std::pair<std::size_t, std::size_t>> overlappingPairs(const std::vector<Aabb>& boxes) {
    std::vector<std::size_t> order(boxes.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t i, std::size_t j) { return boxes[i].min.x < boxes[j].min.x; });

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    std::vector<std::size_t> active;
    for (std::size_t i : order) {
        const Aabb& box = boxes[i];
        active.erase(
            std::remove_if(active.begin(), active.end(), [&](std::size_t j) { return boxes[j].max.x < box.min.x; }),
            active.end());
        for (std::size_t j : active) {
            const Aabb& other = boxes[j];
            if (box.min.y <= other.max.y && other.min.y <= box.max.y && box.min.z <= other.max.z &&
                other.min.z <= box.max.z) {
                pairs.emplace_back(std::min(i, j), std::max(i, j));
            }
        }
        active.push_back(i);
    }
    return pairs;
}

}  // namespace pai
