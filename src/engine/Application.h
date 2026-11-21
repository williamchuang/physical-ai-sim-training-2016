#pragma once

#include <OgreFrameListener.h>
#include <OgreWindowEventUtilities.h>

#include <memory>
#include <string>

#include "core/Config.h"

namespace OIS {
class KeyEvent;
}

namespace Ogre {
class OverlaySystem;
class RenderWindow;
class Root;
class SceneManager;
}  // namespace Ogre

namespace pai {

class AgentCameras;
class DebugConsole;
class Game;
class Hud;
class Input;
class LuaBridge;
class Menu;
class SceneView;

/// Owns the engine objects and runs one frame at a time:
/// input -> scripts -> game rules -> scene -> HUD.
class Application : public Ogre::FrameListener, public Ogre::WindowEventListener {
public:
    explicit Application(std::string configPath);
    ~Application() override;

    /// Returns false if the user cancelled the render-system dialog.
    bool setup();
    void run();

private:
    bool frameStarted(const Ogre::FrameEvent& evt) override;
    void windowResized(Ogre::RenderWindow* window) override;
    void windowClosed(Ogre::RenderWindow* window) override;

    void loadResources();
    void onKey(const OIS::KeyEvent& e);
    void showMenu(bool show);

    std::string configPath_;
    std::string dataRoot_;
    GameConfig config_;
    std::unique_ptr<Ogre::Root> root_;
    std::unique_ptr<Ogre::OverlaySystem> overlaySystem_;
    Ogre::RenderWindow* window_ = nullptr;
    Ogre::SceneManager* scene_ = nullptr;

    std::unique_ptr<Game> game_;
    std::unique_ptr<SceneView> view_;
    std::unique_ptr<AgentCameras> eyes_;
    std::unique_ptr<Input> input_;
    std::unique_ptr<Hud> hud_;
    std::unique_ptr<Menu> menu_;
    std::unique_ptr<DebugConsole> console_;
    std::unique_ptr<LuaBridge> lua_;
    bool quit_ = false;
};

}  // namespace pai
