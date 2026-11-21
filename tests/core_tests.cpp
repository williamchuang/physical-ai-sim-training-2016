#include <random>
#include <sstream>

#include "Check.h"
#include "core/Config.h"
#include "core/GameClock.h"
#include "core/MobiusStrip.h"
#include "physics/Collision.h"

using namespace pai;

namespace {

void checkVecNear(const Vec3& a, const Vec3& b, double tol) {
    CHECK_NEAR(a.x, b.x, tol);
    CHECK_NEAR(a.y, b.y, tol);
    CHECK_NEAR(a.z, b.z, tol);
}

}  // namespace

TEST(strip_matches_parametrisation) {
    const MobiusStrip s(1000.0, 0.5);
    const double u = 1.1, v = 0.3;
    const double ring = 1 + v * std::cos(u / 2);
    checkVecNear(s.position(u, v),
                 Vec3{1000 * ring * std::cos(u), 1000 * ring * std::sin(u), 1000 * v * std::sin(u / 2)}, 1e-9);
}

TEST(strip_tangents_are_derivatives) {
    const MobiusStrip s;
    const double h = 1e-6;
    for (double u : {0.0, 0.7, 2.5, 4.0, 9.0}) {
        for (double v : {-0.4, 0.0, 0.25}) {
            const Vec3 du = (s.position(u + h, v) - s.position(u - h, v)) / (2 * h);
            const Vec3 dv = (s.position(u, v + h) - s.position(u, v - h)) / (2 * h);
            checkVecNear(s.tangentU(u, v), du, 1e-3);
            checkVecNear(s.tangentV(u, v), dv, 1e-3);
        }
    }
}

TEST(strip_has_one_side) {
    // Half a lap further on, the same point is reached from the other face.
    const MobiusStrip s;
    const double u = 0.9, v = 0.2;
    checkVecNear(s.position(u + MobiusStrip::kPeriod / 2, -v), s.position(u, v), 1e-6);
    checkVecNear(s.normal(u + MobiusStrip::kPeriod / 2, -v), -s.normal(u, v), 1e-9);
}

TEST(strip_frame_is_orthonormal_and_right_handed) {
    const MobiusStrip s;
    for (double u = 0; u < MobiusStrip::kPeriod; u += 0.37) {
        const Frame f = s.frame(u, 0.3);
        CHECK_NEAR(f.right.length(), 1, 1e-9);
        CHECK_NEAR(f.up.length(), 1, 1e-9);
        CHECK_NEAR(f.forward.length(), 1, 1e-9);
        CHECK_NEAR(f.right.dot(f.up), 0, 1e-9);
        checkVecNear(f.right.cross(f.up), f.forward, 1e-9);
        CHECK(f.forward.dot(s.tangentU(u, 0.3)) > 0);  // forward is the direction of travel
    }
}

TEST(strip_lateral_component_inverts_tangent) {
    const MobiusStrip s;
    const double u = 2.0, v = -0.1;
    CHECK_NEAR(s.lateralComponent(u, v, s.tangentV(u, v) * 0.05), 0.05, 1e-12);
    CHECK_NEAR(s.lateralComponent(u, v, s.normal(u, v) * 7.0), 0.0, 1e-9);
}

TEST(strip_wraps_and_clamps) {
    const MobiusStrip s;
    CHECK_NEAR(s.wrapU(MobiusStrip::kPeriod + 1.0), 1.0, 1e-12);
    CHECK_NEAR(s.wrapU(-1.0), MobiusStrip::kPeriod - 1.0, 1e-12);
    CHECK_NEAR(s.clampV(2.0), 0.5, 0);
    CHECK_NEAR(s.clampV(-2.0), -0.5, 0);
}

TEST(clock_advances_only_while_running) {
    GameClock c;
    c.tick(1.0);
    CHECK_NEAR(c.elapsed(), 0.0, 0);
    c.resume();
    c.tick(0.5);
    c.pause();
    c.tick(3.0);
    CHECK_NEAR(c.elapsed(), 0.5, 1e-12);
    c.reset();
    CHECK_NEAR(c.elapsed(), 0.0, 0);
}

TEST(config_parses_keys_comments_and_defaults) {
    std::istringstream in(
        "# comment\n"
        "Mode = Train   # trailing comment\n"
        "neural_agents = 1\n"
        "mirror = off\n"
        "policy_file = models/x.txt\n"
        "not a setting\n");
    const GameConfig c = GameConfig::from(KeyValueFile::parse(in));
    CHECK(c.mode == Mode::Train);
    CHECK(c.neuralAgents == 1);
    CHECK(!c.mirror);
    CHECK(c.policyFile == "models/x.txt");
    CHECK(c.scriptedAgents == 0);  // default kept
}

TEST(config_rejects_unknown_mode) {
    std::istringstream in("mode = fly\n");
    bool threw = false;
    try {
        GameConfig::from(KeyValueFile::parse(in));
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(obb_overlap_and_separation) {
    const Frame identity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const Obb a = Obb::fromFrame({0, 0, 0}, identity, {1, 1, 1});
    CHECK(intersects(a, Obb::fromFrame({1.5, 0, 0}, identity, {1, 1, 1})));
    CHECK(!intersects(a, Obb::fromFrame({2.5, 0, 0}, identity, {1, 1, 1})));

    // A box turned 45 degrees about y reaches sqrt(2) along x.
    const double r = std::sqrt(0.5);
    const Frame turned{{r, 0, -r}, {0, 1, 0}, {r, 0, r}};
    CHECK(intersects(a, Obb::fromFrame({2.3, 0, 0}, turned, {1, 1, 1})));
    CHECK(!intersects(a, Obb::fromFrame({2.5, 0, 0}, turned, {1, 1, 1})));
}

TEST(obb_translation_separates) {
    const Frame identity{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Obb a = Obb::fromFrame({0.2, 0.1, 0}, identity, {1, 1, 1});
    const Obb b = Obb::fromFrame({1.5, 0, 0}, identity, {1, 1, 1});
    Vec3 mtd;
    CHECK(intersects(a, b, &mtd));
    CHECK(mtd.x < 0);  // pushes a away from b
    a.center = a.center + mtd * 1.001;
    CHECK(!intersects(a, b));
}

TEST(sweep_and_prune_matches_brute_force) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> pos(0, 100), size(0.5, 8);
    std::vector<Aabb> boxes;
    for (int i = 0; i < 200; ++i) {
        const Vec3 c{pos(rng), pos(rng), pos(rng)}, h{size(rng), size(rng), size(rng)};
        boxes.push_back({c - h, c + h});
    }
    auto pairs = overlappingPairs(boxes);
    std::sort(pairs.begin(), pairs.end());

    std::vector<std::pair<std::size_t, std::size_t>> expected;
    for (std::size_t i = 0; i < boxes.size(); ++i)
        for (std::size_t j = i + 1; j < boxes.size(); ++j) {
            const Aabb &p = boxes[i], &q = boxes[j];
            if (p.min.x <= q.max.x && q.min.x <= p.max.x && p.min.y <= q.max.y && q.min.y <= p.max.y &&
                p.min.z <= q.max.z && q.min.z <= p.max.z)
                expected.emplace_back(i, j);
        }
    CHECK(pairs == expected);
    CHECK(!expected.empty());
}
