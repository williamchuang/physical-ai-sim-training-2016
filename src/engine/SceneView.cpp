#include "engine/SceneView.h"

#include <OgreEntity.h>
#include <OgreManualObject.h>
#include <OgreRenderWindow.h>
#include <OgreSceneNode.h>

#include <cmath>
#include <string>

#include "engine/AgentCameras.h"
#include "engine/OgreConvert.h"

namespace pai {
namespace {

constexpr int kTrackSegments = 512;  // along u, over one turn (0..2 pi)
constexpr int kTrackLanes = 16;      // across v
constexpr int kStripeSegments = 8;   // segments per light or dark stripe
constexpr double kCoinRadius = 12.0;
constexpr double kCoinThickness = 6.0;
constexpr Ogre::Real kPrefabCubeSize = 100.0f;  // Ogre's PT_CUBE edge length
constexpr Ogre::Real kPrefabSphereRadius = 50.0f;

const char* materialFor(Driver d) {
    switch (d) {
        case Driver::Human: return "Pai/Human";
        case Driver::Neural: return "Pai/Neural";
        case Driver::Scripted: return "Pai/Scripted";
    }
    return "Pai/Scripted";
}

}  // namespace

BodySizes SceneView::bodySizes() {
    BodySizes s;
    s.carHalfExtents = Vec3{10, 10, 10};
    s.carLift = 11.0;  // a small gap above the surface keeps the two faces of the strip apart
    s.coinHalfExtents = Vec3{kCoinRadius, kCoinRadius, kCoinThickness / 2};
    s.coinLift = kCoinRadius + 1.0;
    return s;
}

SceneView::SceneView(Ogre::SceneManager& scene, Ogre::RenderWindow& window, const GameConfig& config)
    : scene_(scene), window_(window), config_(config) {
    scene_.setAmbientLight(Ogre::ColourValue(0.6f, 0.6f, 0.6f));
    Ogre::Light* light = scene_.createLight("Pai/SunLight");
    light->setType(Ogre::Light::LT_DIRECTIONAL);
    light->setDirection(Ogre::Vector3(-0.3f, -1.0f, -0.4f).normalisedCopy());

    // The sun at the centre of the ring, as in 2016.
    Ogre::Entity* sun = scene_.createEntity("Pai/Sun", Ogre::SceneManager::PT_SPHERE);
    sun->setMaterialName("Pai/Sun");
    Ogre::SceneNode* sunNode = scene_.getRootSceneNode()->createChildSceneNode();
    sunNode->attachObject(sun);
    sunNode->setScale(Ogre::Vector3(150.0f / kPrefabSphereRadius));

    camera_ = scene_.createCamera("Pai/Main");
    camera_->setNearClipDistance(1.0f);
    camera_->setFarClipDistance(20000.0f);
    camera_->setFixedYawAxis(false);
    viewport_ = window_.addViewport(camera_, 0);
    viewport_->setBackgroundColour(Ogre::ColourValue(0.55f, 0.7f, 0.9f));

    if (config_.mirror) {
        mirrorCamera_ = scene_.createCamera("Pai/Mirror");
        mirrorCamera_->setNearClipDistance(1.0f);
        mirrorCamera_->setFarClipDistance(20000.0f);
        mirrorCamera_->setFixedYawAxis(false);
        mirrorViewport_ = window_.addViewport(mirrorCamera_, 1, 0.68f, 0.13f, 0.3f, 0.18f);
        mirrorViewport_->setBackgroundColour(Ogre::ColourValue(0.55f, 0.7f, 0.9f));
        mirrorViewport_->setOverlaysEnabled(false);
    }
    buildCoinMesh();
    onResize();
}

void SceneView::onResize() {
    camera_->setAspectRatio(Ogre::Real(viewport_->getActualWidth()) / Ogre::Real(viewport_->getActualHeight()));
    if (mirrorCamera_) {
        mirrorCamera_->setAspectRatio(Ogre::Real(mirrorViewport_->getActualWidth()) /
                                      Ogre::Real(mirrorViewport_->getActualHeight()));
    }
}

void SceneView::setMenuMode(bool menu) {
    if (!mirrorViewport_) return;
    // Behind the menu the mirror neither clears nor draws, so the menu shows through.
    mirrorViewport_->setVisibilityMask(menu ? 0u : 0xFFFFFFFFu);
    mirrorViewport_->setClearEveryFrame(!menu);
}

void SceneView::buildTrack(const MobiusStrip& strip) {
    Ogre::ManualObject* track = scene_.createManualObject("Pai/Track");
    track->begin("Pai/Track", Ogre::RenderOperation::OT_TRIANGLE_LIST);
    const double hw = strip.halfWidth();
    for (int i = 0; i <= kTrackSegments; ++i) {
        const double u = MobiusStrip::kPeriod / 2 * i / kTrackSegments;  // one turn covers the surface
        const bool dark = (i / kStripeSegments) % 2 == 1;
        for (int j = 0; j <= kTrackLanes; ++j) {
            const double v = -hw + 2 * hw * j / kTrackLanes;
            const bool edge = j == 0 || j == kTrackLanes;
            const float shade = dark ? 0.32f : 0.48f;
            track->position(toOgre(strip.position(u, v)));
            if (edge)
                track->colour(0.12f, 0.1f, 0.1f);  // dark kerbs: the edge stays visible in grey
            else
                track->colour(shade, shade, shade);
        }
    }
    const int row = kTrackLanes + 1;
    for (int i = 0; i < kTrackSegments; ++i) {
        for (int j = 0; j < kTrackLanes; ++j) {
            const Ogre::uint32 a = i * row + j, b = a + 1, c = a + row, d = c + 1;
            track->triangle(a, c, b);
            track->triangle(b, c, d);
        }
    }
    track->end();
    scene_.getRootSceneNode()->createChildSceneNode()->attachObject(track);
    trackBuilt_ = true;
}

void SceneView::buildCoinMesh() {
    // A flat disc in the local x-y plane, faces along z.
    constexpr int kSides = 24;
    Ogre::ManualObject* coin = scene_.createManualObject("Pai/CoinShape");
    coin->begin("Pai/Coin", Ogre::RenderOperation::OT_TRIANGLE_LIST);
    const Ogre::Real r = Ogre::Real(kCoinRadius), h = Ogre::Real(kCoinThickness / 2);
    for (int face = 0; face < 2; ++face) {
        const Ogre::Real z = face == 0 ? h : -h;
        coin->position(0, 0, z);
        coin->normal(0, 0, face == 0 ? 1.0f : -1.0f);
        for (int k = 0; k <= kSides; ++k) {
            const Ogre::Real a = Ogre::Math::TWO_PI * k / kSides;
            coin->position(r * std::cos(a), r * std::sin(a), z);
            coin->normal(0, 0, face == 0 ? 1.0f : -1.0f);
        }
    }
    const Ogre::uint32 ring = kSides + 2;
    for (Ogre::uint32 k = 1; k <= kSides; ++k) {
        coin->triangle(0, k, k + 1);
        coin->triangle(ring, ring + k + 1, ring + k);
        coin->quad(k, ring + k, ring + k + 1, k + 1);
    }
    coin->end();
    coin->convertToMesh("Pai/Coin.mesh");
    scene_.destroyManualObject(coin);
}

void SceneView::rebuildBodies(const World& world) {
    auto destroyAll = [this](std::vector<Ogre::SceneNode*>& nodes) {
        for (Ogre::SceneNode* n : nodes) {
            auto* e = static_cast<Ogre::Entity*>(n->detachObject(static_cast<unsigned short>(0)));
            scene_.destroyEntity(e);
            scene_.destroySceneNode(n);
        }
        nodes.clear();
    };
    destroyAll(carNodes_);
    destroyAll(coinNodes_);

    const BodySizes& sizes = world.sizes();
    const Ogre::Vector3 carScale = toOgre(sizes.carHalfExtents * 2.0) / kPrefabCubeSize;
    for (const Car& car : world.cars()) {
        Ogre::Entity* e = scene_.createEntity(Ogre::SceneManager::PT_CUBE);
        e->setMaterialName(materialFor(car.driver()));
        Ogre::SceneNode* n = scene_.getRootSceneNode()->createChildSceneNode();
        n->attachObject(e);
        n->setScale(carScale);
        carNodes_.push_back(n);
    }
    for (std::size_t i = 0; i < world.coins().size(); ++i) {
        Ogre::SceneNode* n = scene_.getRootSceneNode()->createChildSceneNode();
        n->attachObject(scene_.createEntity("Pai/Coin.mesh"));
        coinNodes_.push_back(n);
    }
}

void SceneView::sync(const World& world, std::size_t focusCar, std::size_t viewCar) {
    if (!trackBuilt_) buildTrack(world.strip());
    if (carNodes_.size() != world.cars().size() || coinNodes_.size() != world.coins().size()) rebuildBodies(world);

    for (std::size_t i = 0; i < carNodes_.size(); ++i) {
        const Car& c = world.cars()[i];
        carNodes_[i]->setPosition(toOgre(world.carPosition(c)));
        carNodes_[i]->setOrientation(bodyOrientation(world.carFrame(c)));
    }
    for (std::size_t i = 0; i < coinNodes_.size(); ++i) {
        const Coin& c = world.coins()[i];
        coinNodes_[i]->setPosition(toOgre(world.coinPosition(c)));
        coinNodes_[i]->setOrientation(bodyOrientation(world.coinFrame(c)));
    }
    placeCameras(world, focusCar, viewCar);
}

void SceneView::placeCameras(const World& world, std::size_t focusCar, std::size_t viewCar) {
    const Car& followed = world.cars()[focusCar];
    const Frame f = world.carFrame(followed);
    const Vec3 p = world.carPosition(followed);

    if (viewCar != focusCar) {
        // What an agent's camera sees, at window resolution.
        AgentCameras::placeEye(*camera_, world, world.cars()[viewCar]);
    } else {
        camera_->setPosition(toOgre(p - f.forward * 110.0 + f.up * 40.0));
        camera_->setOrientation(cameraOrientation(f));
        camera_->pitch(Ogre::Degree(-12));
    }
    if (mirrorCamera_) {
        const Frame back{-f.right, f.up, -f.forward};
        mirrorCamera_->setPosition(toOgre(p + f.up * 18.0 - f.forward * 12.0));
        mirrorCamera_->setOrientation(cameraOrientation(back));
        mirrorCamera_->pitch(Ogre::Degree(-8));
    }
}

}  // namespace pai
