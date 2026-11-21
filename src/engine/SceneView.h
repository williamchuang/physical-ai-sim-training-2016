#pragma once

#include <OgreCamera.h>
#include <OgreSceneManager.h>
#include <OgreViewport.h>

#include <cstddef>
#include <vector>

#include "core/Config.h"
#include "sim/World.h"

namespace pai {

/// Draws a World: the strip, the cars, the coins, the main camera and the
/// rear-view mirror. Owns scene nodes only; the world itself stays in the simulation.
class SceneView {
public:
    SceneView(Ogre::SceneManager& scene, Ogre::RenderWindow& window, const GameConfig& config);

    /// The physical sizes the renderer uses, for the simulation's collision boxes.
    static BodySizes bodySizes();

    /// Match the scene to the world: rebuilds car and coin nodes if their count
    /// changed, then places every node. Called once per frame.
    void sync(const World& world, std::size_t focusCar, std::size_t viewCar);

    void setMenuMode(bool menu);
    void onResize();

private:
    void buildTrack(const MobiusStrip& strip);
    void buildCoinMesh();
    void rebuildBodies(const World& world);
    void placeCameras(const World& world, std::size_t focusCar, std::size_t viewCar);

    Ogre::SceneManager& scene_;
    Ogre::RenderWindow& window_;
    GameConfig config_;
    Ogre::Camera* camera_ = nullptr;
    Ogre::Camera* mirrorCamera_ = nullptr;
    Ogre::Viewport* viewport_ = nullptr;
    Ogre::Viewport* mirrorViewport_ = nullptr;
    std::vector<Ogre::SceneNode*> carNodes_;
    std::vector<Ogre::SceneNode*> coinNodes_;
    bool trackBuilt_ = false;
};

}  // namespace pai
