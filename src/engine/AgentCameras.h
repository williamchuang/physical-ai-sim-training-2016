#pragma once

#include <OgreCamera.h>
#include <OgreRenderTexture.h>
#include <OgreSceneManager.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "game/Game.h"

namespace pai {

/// Each sensing car carries a tiny camera rendered into its own texture
/// (sensorWidth x sensorHeight, 3 x 3 in 2016). A reading is that image in
/// grey levels scaled to [0, 1], read back straight from the render target.
class AgentCameras : public Perception {
public:
    AgentCameras(Ogre::SceneManager& scene, const World& world, int width, int height);
    ~AgentCameras() override;
    AgentCameras(const AgentCameras&) = delete;
    AgentCameras& operator=(const AgentCameras&) = delete;

    bool observe(std::size_t carIndex, std::vector<float>& out) override;

    /// Put a camera where a car's sensor sits: on the front of the car, tilted to the road.
    static void placeEye(Ogre::Camera& camera, const World& world, const Car& car);

private:
    struct Eye {
        Ogre::Camera* camera = nullptr;
        Ogre::TexturePtr texture;
        Ogre::RenderTexture* target = nullptr;
    };
    Eye& eyeFor(std::size_t carIndex);

    Ogre::SceneManager& scene_;
    const World& world_;
    int width_;
    int height_;
    std::unordered_map<std::size_t, Eye> eyes_;
    std::vector<std::uint8_t> pixels_;
};

}  // namespace pai
