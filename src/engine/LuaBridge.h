#pragma once

#include <functional>
#include <string>

struct lua_State;

namespace pai {

class Game;

/// What scripts may ask the application to do, beyond reading and moving cars.
struct ScriptHost {
    std::function<void()> start;
    std::function<void()> resume;
    std::function<void()> quit;
    std::function<void(const std::string&)> print;
    std::function<void()> clearOutput;
    std::function<void(double)> setMessageSeconds;
};

/// Lua 5.2 scripting. Registers the game's commands as Lua functions, runs
/// console lines and menu actions, and calls ScriptManagerThink(dt) each frame
/// when a script defines it.
class LuaBridge {
public:
    LuaBridge(Game& game, ScriptHost host);
    ~LuaBridge();
    LuaBridge(const LuaBridge&) = delete;
    LuaBridge& operator=(const LuaBridge&) = delete;

    /// Run a chunk of Lua; errors go to the console.
    bool run(const std::string& code);
    /// Run a script file if it exists.
    bool runFile(const std::string& path);
    void think(double dt);

    Game& game() { return game_; }
    ScriptHost& host() { return host_; }

private:
    void report(int status);

    Game& game_;
    ScriptHost host_;
    lua_State* L_;
};

}  // namespace pai
