#include <exception>
#include <iostream>

#include "engine/Application.h"

#ifndef PAI_DEFAULT_CONFIG
#define PAI_DEFAULT_CONFIG "config/game.cfg"
#endif

int main(int argc, char** argv) {
    const char* config = argc > 1 ? argv[1] : PAI_DEFAULT_CONFIG;
    try {
        pai::Application app(config);
        if (!app.setup()) return 0;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
