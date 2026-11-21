#pragma once

#include <OIS.h>
#include <OgreOverlay.h>
#include <OgreOverlayContainer.h>
#include <OgreTextAreaOverlayElement.h>

#include <array>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "game/Game.h"

namespace pai {

/// Score, time, level and training status across the top of the window.
class Hud {
public:
    Hud();
    void update(const Game& game);
    void setVisible(bool visible);

private:
    Ogre::Overlay* overlay_;
    std::array<Ogre::TextAreaOverlayElement*, 3> lines_{};
};

/// The pause menu: Start, Continue, Quit. Up and Down choose, Enter runs the
/// choice as a Lua command, so the menu and the console share one code path.
class Menu {
public:
    using Run = std::function<void(const std::string& luaCode)>;

    explicit Menu(Run run);
    void handle(const OIS::KeyEvent& e);
    void setVisible(bool visible);
    bool visible() const { return visible_; }

private:
    void refresh();

    Run run_;
    Ogre::Overlay* overlay_;
    std::vector<Ogre::TextAreaOverlayElement*> items_;
    std::size_t selected_ = 0;
    bool visible_ = true;
};

/// The Lua console, opened with the ` key. Messages fade after a set time.
class DebugConsole {
public:
    using Run = std::function<void(const std::string& luaCode)>;

    explicit DebugConsole(Run run);
    void handle(const OIS::KeyEvent& e);
    void update(double dt);

    void print(const std::string& text);
    void clear() { messages_.clear(); }
    void setMessageSeconds(double seconds) { messageSeconds_ = seconds; }

    void setActive(bool active);
    bool active() const { return active_; }

private:
    struct Message {
        std::string text;
        double secondsLeft;
    };
    void refresh();

    Run run_;
    Ogre::Overlay* overlay_;
    Ogre::TextAreaOverlayElement* text_;
    std::deque<Message> messages_;
    std::vector<std::string> history_;
    std::size_t historyCursor_ = 0;
    std::string line_;
    double messageSeconds_ = 8.0;
    bool active_ = false;
};

}  // namespace pai
