#include "engine/Input.h"

namespace pai {

Input::Input(std::size_t windowHandle) {
    OIS::ParamList params;
    params.insert({"WINDOW", std::to_string(windowHandle)});
#if defined(OIS_LINUX_PLATFORM)
    params.insert({"x11_keyboard_grab", "false"});
    params.insert({"XAutoRepeatOn", "true"});
#endif
    manager_ = OIS::InputManager::createInputSystem(params);
    keyboard_ = static_cast<OIS::Keyboard*>(manager_->createInputObject(OIS::OISKeyboard, true));
    keyboard_->setTextTranslation(OIS::Keyboard::Unicode);
    keyboard_->setEventCallback(this);
}

Input::~Input() {
    if (manager_) {
        manager_->destroyInputObject(keyboard_);
        OIS::InputManager::destroyInputSystem(manager_);
    }
}

bool Input::keyPressed(const OIS::KeyEvent& e) {
    if (onPress_) onPress_(e);
    return true;
}

DriveInput Input::driving() const {
    DriveInput in;
    if (isDown(OIS::KC_T) || isDown(OIS::KC_UP)) in.throttle += 1;
    if (isDown(OIS::KC_G) || isDown(OIS::KC_DOWN)) in.throttle -= 1;
    if (isDown(OIS::KC_F) || isDown(OIS::KC_LEFT)) in.steer += 1;
    if (isDown(OIS::KC_H) || isDown(OIS::KC_RIGHT)) in.steer -= 1;
    return in;
}

}  // namespace pai
