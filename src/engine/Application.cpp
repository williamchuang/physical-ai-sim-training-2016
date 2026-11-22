#include "engine/Application.h"

#include <OIS.h>
#include <Ogre.h>
#include <OgreOverlaySystem.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "engine/AgentCameras.h"
#include "engine/Input.h"
#include "engine/LuaBridge.h"
#include "engine/SceneView.h"
#include "engine/Ui.h"
#include "game/Game.h"

#ifndef PAI_PLUGINS_CFG
#define PAI_PLUGINS_CFG "plugins.cfg"
#endif

namespace pai {
namespace {

constexpr double kMaxFrameSeconds = 0.1;  // a stalled frame must not teleport the cars
constexpr const char* kTitle = "Physical AI Simulation Training (2016)";

/// The project folder: the nearest folder above the config file that holds media/.
std::string findDataRoot(const std::string& configPath) {
    std::filesystem::path dir = std::filesystem::absolute(configPath).parent_path();
    for (std::filesystem::path d = dir; !d.empty(); d = d.parent_path()) {
        if (std::filesystem::is_directory(d / "media")) return d.string();
        if (d == d.parent_path()) break;
    }
    return dir.string();
}

/// Paths in the config are relative to the project folder.
std::string resolve(const std::string& root, const std::string& path) {
    const std::filesystem::path p(path);
    return p.is_absolute() ? path : (std::filesystem::path(root) / p).string();
}

/// Prefer OpenGL in a 1280x720 window when there is no saved ogre.cfg yet.
bool chooseRenderSystem(Ogre::Root& root) {
    if (root.restoreConfig()) return true;
    for (Ogre::RenderSystem* rs : root.getAvailableRenderers()) {
        if (rs->getName().find("OpenGL") == Ogre::String::npos) continue;
        rs->setConfigOption("Full Screen", "No");
        rs->setConfigOption("Video Mode", "1280 x 720");
        root.setRenderSystem(rs);
        root.saveConfig();
        return true;
    }
    return root.showConfigDialog();
}

}  // namespace

Application::Application(std::string configPath) : configPath_(std::move(configPath)) {
    dataRoot_ = findDataRoot(configPath_);
    config_ = GameConfig::from(KeyValueFile::load(configPath_));
    config_.policyFile = resolve(dataRoot_, config_.policyFile);
    config_.transformerFile = resolve(dataRoot_, config_.transformerFile);
    if (!config_.recordFile.empty()) config_.recordFile = resolve(dataRoot_, config_.recordFile);
}

Application::~Application() {
    if (window_) Ogre::WindowEventUtilities::removeWindowEventListener(window_, this);
    // Engine-side objects first, then the overlay system, then Root (member order).
    lua_.reset();
    console_.reset();
    menu_.reset();
    hud_.reset();
    input_.reset();
    eyes_.reset();
    view_.reset();
    game_.reset();
}

void Application::loadResources() {
    auto& groups = Ogre::ResourceGroupManager::getSingleton();
    const std::string media = resolve(dataRoot_, "media");
    groups.addResourceLocation(media + "/materials", "FileSystem");
    groups.addResourceLocation(media + "/fonts", "FileSystem");
    groups.initialiseAllResourceGroups();
}

bool Application::setup() {
    root_ = std::make_unique<Ogre::Root>(PAI_PLUGINS_CFG, "ogre.cfg", "ogre.log");
    overlaySystem_ = std::make_unique<Ogre::OverlaySystem>();
    if (!chooseRenderSystem(*root_)) return false;

    window_ = root_->initialise(true, kTitle);
    scene_ = root_->createSceneManager(Ogre::ST_GENERIC);
    scene_->addRenderQueueListener(overlaySystem_.get());
    loadResources();

    game_ = std::make_unique<Game>(config_, SceneView::bodySizes());
    view_ = std::make_unique<SceneView>(*scene_, *window_, config_);
    eyes_ = std::make_unique<AgentCameras>(*scene_, game_->world(), config_.sensorWidth, config_.sensorHeight);

    std::size_t handle = 0;
    window_->getCustomAttribute("WINDOW", &handle);
    input_ = std::make_unique<Input>(handle);
    input_->onPress([this](const OIS::KeyEvent& e) { onKey(e); });

    hud_ = std::make_unique<Hud>();
    auto runLua = [this](const std::string& code) { lua_->run(code); };
    menu_ = std::make_unique<Menu>(runLua);
    console_ = std::make_unique<DebugConsole>(runLua);

    ScriptHost host;
    host.start = [this] {
        game_->start();
        showMenu(false);
    };
    host.resume = [this] { showMenu(false); };
    host.quit = [this] { quit_ = true; };
    host.print = [this](const std::string& text) {
        console_->print(text);
        std::cout << text << std::endl;  // also on the terminal, for scripted runs
    };
    host.clearOutput = [this] { console_->clear(); };
    host.setMessageSeconds = [this](double s) { console_->setMessageSeconds(s); };
    lua_ = std::make_unique<LuaBridge>(*game_, std::move(host));

    Ogre::WindowEventUtilities::addWindowEventListener(window_, this);
    root_->addFrameListener(this);
    console_->print(game_->policyStatus());
    showMenu(true);
    lua_->runFile(resolve(dataRoot_, "scripts/ScriptManager.lua"));
    lua_->runFile(resolve(dataRoot_, config_.startupScript));  // may start a round itself
    return true;
}

void Application::run() {
    root_->startRendering();
}

void Application::showMenu(bool show) {
    menu_->setVisible(show);
    view_->setMenuMode(show);
    show ? game_->pause() : game_->resume();
}

void Application::onKey(const OIS::KeyEvent& e) {
    if (e.key == OIS::KC_ESCAPE) {
        if (console_->active())
            console_->setActive(false);
        else
            quit_ = true;
        return;
    }
    if (e.key == OIS::KC_GRAVE) {
        console_->setActive(!console_->active());
        return;
    }
    if (console_->active()) {
        console_->handle(e);
        return;
    }
    if (e.key == OIS::KC_P) {
        showMenu(!menu_->visible());
        return;
    }
    if (menu_->visible()) {
        menu_->handle(e);
        return;
    }
    // Test keys from 2016: 6/7 collisions off/on, 8 pause, 9 resume, 0 reset the clock.
    switch (e.key) {
        case OIS::KC_6: game_->setCollisions(false); break;
        case OIS::KC_7: game_->setCollisions(true); break;
        case OIS::KC_8: game_->pause(); break;
        case OIS::KC_9: game_->resume(); break;
        case OIS::KC_0: game_->resetClock(); break;
        default: break;
    }
}

bool Application::frameStarted(const Ogre::FrameEvent& evt) {
    if (quit_ || window_->isClosed()) return false;
    const double dt = config_.fixedFrameSeconds > 0
                          ? config_.fixedFrameSeconds
                          : std::min(static_cast<double>(evt.timeSinceLastFrame), kMaxFrameSeconds);

    input_->capture();
    console_->update(dt);
    lua_->think(dt);

    const bool driving = !menu_->visible() && !console_->active();
    game_->update(dt, driving ? input_->driving() : DriveInput{}, eyes_.get());

    const std::size_t focus = game_->focusCar();
    const std::size_t viewCar = config_.cameraView == CameraView::Agent ? game_->firstBot().value_or(focus) : focus;
    view_->sync(game_->world(), focus, viewCar);
    hud_->update(*game_);
    return !quit_;
}

void Application::windowResized(Ogre::RenderWindow*) {
    view_->onResize();
}

void Application::windowClosed(Ogre::RenderWindow*) {
    quit_ = true;
}

}  // namespace pai
