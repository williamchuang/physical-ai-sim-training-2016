#pragma once

#include <OIS.h>

#include <cstddef>
#include <functional>
#include <string>

#include "sim/Car.h"

namespace pai {

/// Keyboard through OIS. Held keys are polled; presses arrive once each, as events,
/// so a toggle key never flips twice because it stayed down for two frames.
class Input : public OIS::KeyListener {
public:
    using PressHandler = std::function<void(const OIS::KeyEvent&)>;

    explicit Input(std::size_t windowHandle);
    ~Input() override;
    Input(const Input&) = delete;
    Input& operator=(const Input&) = delete;

    void capture() { keyboard_->capture(); }
    void onPress(PressHandler handler) { onPress_ = std::move(handler); }

    bool isDown(OIS::KeyCode key) const { return keyboard_->isKeyDown(key); }

    /// The driving keys for your cube: T faster, G slower, F left, H right (as in 2016),
    /// with the arrow keys as an alternative.
    DriveInput driving() const;

private:
    bool keyPressed(const OIS::KeyEvent& e) override;
    bool keyReleased(const OIS::KeyEvent&) override { return true; }

    OIS::InputManager* manager_ = nullptr;
    OIS::Keyboard* keyboard_ = nullptr;
    PressHandler onPress_;
};

}  // namespace pai
