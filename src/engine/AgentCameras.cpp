#include "engine/AgentCameras.h"

#include <OgreHardwarePixelBuffer.h>
#include <OgreTextureManager.h>
#include <OgreViewport.h>

#include <string>

#include "engine/OgreConvert.h"

namespace pai {
namespace {

constexpr int kOversample = 4;                // rendered pixels per sensor pixel, each way
constexpr double kFieldOfViewDegrees = 60.0;  // vertical
constexpr double kPitchDegrees = -6.0;        // tilted a little towards the road

}  // namespace

AgentCameras::AgentCameras(Ogre::SceneManager& scene, const World& world, int width, int height)
    : scene_(scene),
      world_(world),
      width_(width),
      height_(height),
      pixels_(static_cast<std::size_t>(width) * height * kOversample * kOversample * 3) {}

AgentCameras::~AgentCameras() {
    for (auto& [index, eye] : eyes_) {
        eye.target->removeAllViewports();
        Ogre::TextureManager::getSingleton().remove(eye.texture->getHandle());
        scene_.destroyCamera(eye.camera);
    }
}

AgentCameras::Eye& AgentCameras::eyeFor(std::size_t carIndex) {
    auto it = eyes_.find(carIndex);
    if (it != eyes_.end()) return it->second;

    const std::string name = "Pai/Eye/" + std::to_string(carIndex);
    Eye eye;
    eye.camera = scene_.createCamera(name);
    eye.camera->setNearClipDistance(1.0f);
    eye.camera->setFarClipDistance(20000.0f);
    eye.camera->setFixedYawAxis(false);
    eye.camera->setAspectRatio(Ogre::Real(width_) / Ogre::Real(height_));
    eye.camera->setFOVy(Ogre::Degree(kFieldOfViewDegrees));
    eye.texture = Ogre::TextureManager::getSingleton().createManual(
        name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, Ogre::TEX_TYPE_2D,
        static_cast<Ogre::uint>(width_ * kOversample), static_cast<Ogre::uint>(height_ * kOversample), 0,
        Ogre::PF_R8G8B8, Ogre::TU_RENDERTARGET);
    eye.target = eye.texture->getBuffer()->getRenderTarget();
    eye.target->setAutoUpdated(false);
    Ogre::Viewport* vp = eye.target->addViewport(eye.camera);
    vp->setClearEveryFrame(true);
    vp->setBackgroundColour(Ogre::ColourValue(0.55f, 0.7f, 0.9f));
    vp->setOverlaysEnabled(false);
    return eyes_.emplace(carIndex, eye).first->second;
}

void AgentCameras::placeEye(Ogre::Camera& camera, const World& world, const Car& car) {
    const Frame f = world.carFrame(car);
    const double frontOfCar = world.sizes().carHalfExtents.z + 2.0;
    camera.setPosition(toOgre(world.carPosition(car) + f.up * 8.0 + f.forward * frontOfCar));
    camera.setOrientation(cameraOrientation(f));
    camera.pitch(Ogre::Degree(kPitchDegrees));
}

bool AgentCameras::observe(std::size_t carIndex, std::vector<float>& out) {
    if (carIndex >= world_.cars().size()) return false;
    Eye& eye = eyeFor(carIndex);
    placeEye(*eye.camera, world_, world_.cars()[carIndex]);
    eye.target->update();

    const std::size_t w = static_cast<std::size_t>(width_) * kOversample;
    const std::size_t h = static_cast<std::size_t>(height_) * kOversample;
    Ogre::PixelBox box(w, h, 1, Ogre::PF_BYTE_RGB, pixels_.data());
    eye.target->copyContentsToMemory(box, Ogre::RenderTarget::FB_AUTO);

    // Each sensor pixel is the mean grey of a kOversample x kOversample block, as a
    // real sensor integrates light: a coin smaller than a pixel still brightens it.
    out.assign(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), 0.0f);
    constexpr float kScale = 1.0f / (255.0f * kOversample * kOversample);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            const std::uint8_t* p = &pixels_[3 * (y * w + x)];
            out[(y / kOversample) * static_cast<std::size_t>(width_) + x / kOversample] +=
                (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) * kScale;
        }
    }
    return true;
}

}  // namespace pai
