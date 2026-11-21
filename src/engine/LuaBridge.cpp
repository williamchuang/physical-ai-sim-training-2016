#include "engine/LuaBridge.h"

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "game/Game.h"

namespace pai {
namespace {

LuaBridge& bridge(lua_State* L) {
    return *static_cast<LuaBridge*>(lua_touserdata(L, lua_upvalueindex(1)));
}

Car& carArg(lua_State* L, int index) {
    const char* name = luaL_checkstring(L, index);
    if (Car* car = bridge(L).game().world().find(name)) return *car;
    luaL_error(L, "no car named '%s' (see cars())", name);  // raises a Lua error and does not return
    std::abort();
}

// Each command is a plain C function; the bridge arrives as an upvalue.

int luaStart(lua_State* L) {
    bridge(L).host().start();
    return 0;
}
int luaContinue(lua_State* L) {
    bridge(L).host().resume();
    return 0;
}
int luaQuit(lua_State* L) {
    bridge(L).host().quit();
    return 0;
}
int luaClear(lua_State* L) {
    bridge(L).host().clearOutput();
    return 0;
}

int luaPrint(lua_State* L) {
    std::ostringstream line;
    const int n = lua_gettop(L);
    for (int i = 1; i <= n; ++i) {
        if (i > 1) line << '\t';
        line << luaL_tolstring(L, i, nullptr);
        lua_pop(L, 1);
    }
    bridge(L).host().print(line.str());
    return 0;
}

int luaSetDebugTime(lua_State* L) {
    bridge(L).host().setMessageSeconds(luaL_checknumber(L, 1));
    return 0;
}

int luaGetPosition(lua_State* L) {
    World& world = bridge(L).game().world();
    const Vec3 p = world.carPosition(carArg(L, 1));
    lua_pushnumber(L, p.x);
    lua_pushnumber(L, p.y);
    lua_pushnumber(L, p.z);
    return 3;
}

int luaGetTrackPosition(lua_State* L) {
    const Car& c = carArg(L, 1);
    lua_pushnumber(L, c.u());
    lua_pushnumber(L, c.v());
    return 2;
}

int luaSetLane(lua_State* L) {
    World& world = bridge(L).game().world();
    carArg(L, 1).setLane(luaL_checknumber(L, 2), world.strip().halfWidth());
    return 0;
}

int luaSetSpeed(lua_State* L) {
    World& world = bridge(L).game().world();
    carArg(L, 1).setSpeedCoefficient(luaL_checknumber(L, 2), world.limits());
    return 0;
}

int luaCars(lua_State* L) {
    const auto& cars = bridge(L).game().world().cars();
    lua_createtable(L, static_cast<int>(cars.size()), 0);
    for (std::size_t i = 0; i < cars.size(); ++i) {
        lua_pushstring(L, cars[i].name().c_str());
        lua_rawseti(L, -2, static_cast<int>(i + 1));
    }
    return 1;
}

int luaSetCollisions(lua_State* L) {
    bridge(L).game().setCollisions(lua_toboolean(L, 1) != 0);
    return 0;
}

int luaScore(lua_State* L) {
    lua_pushinteger(L, carArg(L, 1).score);
    return 1;
}

int luaRoundOver(lua_State* L) {
    lua_pushboolean(L, bridge(L).game().roundOver());
    return 1;
}

int luaCoinsLeft(lua_State* L) {
    lua_pushinteger(L, static_cast<lua_Integer>(bridge(L).game().coinsLeft()));
    return 1;
}

int luaSavePolicy(lua_State* L) {
    lua_pushboolean(L, bridge(L).game().savePolicyNow());
    return 1;
}

int luaHelp(lua_State* L) {
    bridge(L).host().print(
        "Start()  Continue()  Quit()  help()\n"
        "cars()  getPosition(name)  getTrackPosition(name)  setLane(name, v)  setSpeed(name, c)\n"
        "score(name)  coinsLeft()  roundOver()  setCollisions(on)  savePolicy()\n"
        "debugOutput(...)  clearDebugOutput()  setDebugTime(seconds)");
    return 0;
}

const luaL_Reg kCommands[] = {
    {"Start", luaStart},
    {"Continue", luaContinue},
    {"Quit", luaQuit},
    {"help", luaHelp},
    {"showFunctions", luaHelp},
    {"print", luaPrint},
    {"debugOutput", luaPrint},
    {"clearDebugOutput", luaClear},
    {"setDebugTime", luaSetDebugTime},
    {"cars", luaCars},
    {"getPosition", luaGetPosition},
    {"getTrackPosition", luaGetTrackPosition},
    {"setLane", luaSetLane},
    {"setSpeed", luaSetSpeed},
    {"setCollisions", luaSetCollisions},
    {"savePolicy", luaSavePolicy},
    {"score", luaScore},
    {"roundOver", luaRoundOver},
    {"coinsLeft", luaCoinsLeft},
    {nullptr, nullptr},
};

}  // namespace

LuaBridge::LuaBridge(Game& game, ScriptHost host) : game_(game), host_(std::move(host)), L_(luaL_newstate()) {
    luaL_openlibs(L_);
    lua_pushglobaltable(L_);
    lua_pushlightuserdata(L_, this);
    luaL_setfuncs(L_, kCommands, 1);
    lua_pop(L_, 1);
}

LuaBridge::~LuaBridge() {
    lua_close(L_);
}

void LuaBridge::report(int status) {
    if (status == LUA_OK) return;
    host_.print(std::string("lua: ") + (lua_tostring(L_, -1) ? lua_tostring(L_, -1) : "error"));
    lua_pop(L_, 1);
}

bool LuaBridge::run(const std::string& code) {
    int status = luaL_loadbuffer(L_, code.data(), code.size(), "console");
    if (status == LUA_OK) status = lua_pcall(L_, 0, 0, 0);
    report(status);
    return status == LUA_OK;
}

bool LuaBridge::runFile(const std::string& path) {
    if (!std::ifstream(path)) return false;
    const int status = luaL_dofile(L_, path.c_str());
    report(status);
    return status == LUA_OK;
}

void LuaBridge::think(double dt) {
    lua_getglobal(L_, "ScriptManagerThink");
    if (!lua_isfunction(L_, -1)) {
        lua_pop(L_, 1);
        return;
    }
    lua_pushnumber(L_, dt);
    report(lua_pcall(L_, 1, 0, 0));
}

}  // namespace pai
