#pragma once

namespace pai {

/// Game time that advances only while running. Time comes from the frame
/// delta, so a paused menu, a slow frame or a reset never moves objects.
class GameClock {
public:
    void tick(double dt) {
        if (running_) elapsed_ += dt;
    }
    void pause() { running_ = false; }
    void resume() { running_ = true; }
    void reset() { elapsed_ = 0.0; }

    bool running() const { return running_; }
    double elapsed() const { return elapsed_; }

private:
    double elapsed_ = 0.0;
    bool running_ = false;  // the game opens on the menu
};

}  // namespace pai
