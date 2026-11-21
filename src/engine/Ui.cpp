#include "engine/Ui.h"

#include <OgreOverlayManager.h>

#include <cmath>
#include <cstdio>
#include <sstream>

namespace pai {
namespace {

constexpr std::size_t kMaxMessages = 14;
constexpr const char* kFont = "Pai/Font";

Ogre::OverlayManager& overlays() {
    return Ogre::OverlayManager::getSingleton();
}

Ogre::OverlayContainer* panel(const std::string& name, Ogre::Real left, Ogre::Real top, Ogre::Real width,
                              Ogre::Real height, const std::string& material) {
    auto* p = static_cast<Ogre::OverlayContainer*>(overlays().createOverlayElement("Panel", name));
    p->setMetricsMode(Ogre::GMM_RELATIVE);
    p->setPosition(left, top);
    p->setDimensions(width, height);
    if (!material.empty()) p->setMaterialName(material);
    return p;
}

Ogre::TextAreaOverlayElement* text(Ogre::OverlayContainer* parent, const std::string& name, Ogre::Real left,
                                   Ogre::Real top, Ogre::Real height) {
    auto* t = static_cast<Ogre::TextAreaOverlayElement*>(overlays().createOverlayElement("TextArea", name));
    t->setMetricsMode(Ogre::GMM_RELATIVE);
    t->setPosition(left, top);
    t->setCharHeight(height);
    t->setFontName(kFont);
    t->setColour(Ogre::ColourValue::White);
    parent->addChild(t);
    return t;
}

std::string format(const char* fmt, double a, double b = 0, double c = 0, double d = 0) {
    char buf[160];
    std::snprintf(buf, sizeof buf, fmt, a, b, c, d);
    return buf;
}

}  // namespace

// ---------------------------------------------------------------- Hud

Hud::Hud() : overlay_(overlays().create("Pai/Hud")) {
    Ogre::OverlayContainer* p = panel("Pai/Hud/Panel", 0, 0, 1, 0.11f, "Pai/HudPanel");
    for (std::size_t i = 0; i < lines_.size(); ++i)
        lines_[i] = text(p, "Pai/Hud/Line" + std::to_string(i), 0.01f, 0.005f + 0.035f * i, 0.032f);
    overlay_->add2D(p);
    overlay_->setZOrder(100);
    overlay_->show();
}

void Hud::setVisible(bool visible) {
    visible ? overlay_->show() : overlay_->hide();
}

void Hud::update(const Game& game) {
    const Mode mode = game.config().mode;
    std::string first;
    if (mode == Mode::Compare) {
        first =
            "You " + std::to_string(game.score(*game.humanCar())) + "    Bot " + std::to_string(game.bestBotScore());
    } else {
        first = "Score " + std::to_string(game.score(game.focusCar()));
    }
    first += "    Level " + std::to_string(game.level()) + "    Coins left " + std::to_string(game.coinsLeft()) +
             format("    Time %.0f s", std::ceil(game.secondsLeft()));
    if (!game.collisions()) first += "    collisions off";
    lines_[0]->setCaption(first);

    std::ostringstream second;
    if (mode == Mode::Train) {
        second << (game.config().demonstrator == Demonstrator::Autopilot ? "Training from the autopilot"
                                                                         : "Training from your keys")
               << "    samples " << game.trainingSamples()
               << format("    loss %.3f", static_cast<double>(game.trainingLoss()));
    } else {
        for (const Car& c : game.world().cars())
            if (c.driver() == Driver::Neural) second << c.name() << ' ' << c.score << "    ";
    }
    lines_[1]->setCaption(second.str());

    std::string third = game.policyStatus();
    if (game.roundOver()) {
        third = game.coinsLeft() == 0 ? "Round over: every coin collected." : "Round over: time is up.";
        if (mode == Mode::Compare) {
            const int you = game.score(*game.humanCar()), bot = game.bestBotScore();
            third += you > bot ? "  You scored higher." : (you < bot ? "  The bot scored higher." : "  Equal scores.");
        }
        third += "  P for the menu.";
    }
    lines_[2]->setCaption(third);
}

// ---------------------------------------------------------------- Menu

Menu::Menu(Run run) : run_(std::move(run)), overlay_(overlays().create("Pai/Menu")) {
    Ogre::OverlayContainer* p = panel("Pai/Menu/Panel", 0, 0, 1, 1, "Pai/MenuPanel");
    text(p, "Pai/Menu/Title", 0.1f, 0.12f, 0.07f)->setCaption("Physical AI Simulation Training (2016)");
    const std::array<const char*, 3> labels{"Start", "Continue", "Quit"};
    for (std::size_t i = 0; i < labels.size(); ++i) {
        auto* t = text(p, std::string("Pai/Menu/") + labels[i], 0.2f, 0.32f + 0.12f * i, 0.09f);
        t->setCaption(labels[i]);
        items_.push_back(t);
    }
    text(p, "Pai/Menu/Help", 0.1f, 0.82f, 0.035f)
        ->setCaption("Up/Down choose, Enter select.  Drive: T faster, G slower, F left, H right.  P menu, ` console.");
    overlay_->add2D(p);
    overlay_->setZOrder(300);
    overlay_->show();
    refresh();
}

void Menu::refresh() {
    for (std::size_t i = 0; i < items_.size(); ++i) {
        items_[i]->setColour(i == selected_ ? Ogre::ColourValue(1.0f, 0.85f, 0.2f) : Ogre::ColourValue::White);
    }
}

void Menu::setVisible(bool visible) {
    visible_ = visible;
    visible ? overlay_->show() : overlay_->hide();
}

void Menu::handle(const OIS::KeyEvent& e) {
    if (e.key == OIS::KC_UP) {
        selected_ = (selected_ + items_.size() - 1) % items_.size();
    } else if (e.key == OIS::KC_DOWN) {
        selected_ = (selected_ + 1) % items_.size();
    } else if (e.key == OIS::KC_RETURN || e.key == OIS::KC_NUMPADENTER) {
        run_(items_[selected_]->getCaption().asUTF8() + "()");
    }
    refresh();
}

// ---------------------------------------------------------------- DebugConsole

DebugConsole::DebugConsole(Run run) : run_(std::move(run)), overlay_(overlays().create("Pai/Console")) {
    Ogre::OverlayContainer* p = panel("Pai/Console/Panel", 0, 0.55f, 1, 0.45f, "Pai/ConsolePanel");
    text_ = text(p, "Pai/Console/Text", 0.01f, 0.01f, 0.028f);
    overlay_->add2D(p);
    overlay_->setZOrder(200);
    refresh();
}

void DebugConsole::setActive(bool active) {
    active_ = active;
    refresh();
}

void DebugConsole::print(const std::string& message) {
    std::istringstream lines(message);
    for (std::string line; std::getline(lines, line);) messages_.push_back({line, messageSeconds_});
    while (messages_.size() > kMaxMessages) messages_.pop_front();
    refresh();
}

void DebugConsole::update(double dt) {
    bool changed = false;
    for (Message& m : messages_) m.secondsLeft -= dt;
    while (!messages_.empty() && messages_.front().secondsLeft <= 0) {
        messages_.pop_front();
        changed = true;
    }
    if (changed) refresh();
}

void DebugConsole::handle(const OIS::KeyEvent& e) {
    switch (e.key) {
        case OIS::KC_RETURN:
        case OIS::KC_NUMPADENTER:
            if (!line_.empty()) {
                print("> " + line_);
                history_.push_back(line_);
                historyCursor_ = history_.size();
                run_(line_);
                line_.clear();
            }
            break;
        case OIS::KC_BACK:
            if (!line_.empty()) line_.pop_back();
            break;
        case OIS::KC_UP:
            if (historyCursor_ > 0) line_ = history_[--historyCursor_];
            break;
        case OIS::KC_DOWN:
            if (historyCursor_ + 1 < history_.size()) {
                line_ = history_[++historyCursor_];
            } else {
                historyCursor_ = history_.size();
                line_.clear();
            }
            break;
        default:
            if (e.text >= 32 && e.text < 127) line_.push_back(static_cast<char>(e.text));
            break;
    }
    refresh();
}

void DebugConsole::refresh() {
    if (!active_ && messages_.empty()) {
        overlay_->hide();
        return;
    }
    std::string all;
    for (const Message& m : messages_) all += m.text + "\n";
    if (active_) all += "lua> " + line_ + "_";
    text_->setCaption(all);
    overlay_->show();
}

}  // namespace pai
