cmake_minimum_required(VERSION 3.16)
project(physical_ai_sim_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PAI_BUILD_GAME "Build the Ogre application (needs Ogre 1.9, OIS and Lua 5.2)" ON)
option(PAI_BUILD_TESTS "Build the unit tests (no engine needed)" ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Simulation core: track, physics, cars, learning, game rules. No engine types.
add_library(pai_core STATIC
  src/core/Config.cpp
  src/core/MobiusStrip.cpp
  src/physics/Collision.cpp
  src/ai/Mlp.cpp
  transformer/cpp/Transformer.cpp
  src/ai/ImitationLearner.cpp
  src/sim/Car.cpp
  src/sim/World.cpp
  src/sim/Autopilot.cpp
  src/game/Game.cpp)
target_include_directories(pai_core PUBLIC src transformer/cpp)

if(PAI_BUILD_TESTS)
  enable_testing()
  add_executable(pai_tests tests/test_main.cpp tests/core_tests.cpp tests/ai_tests.cpp tests/sim_tests.cpp
    transformer/tests/transformer_tests.cpp)
  target_include_directories(pai_tests PRIVATE tests)
  target_link_libraries(pai_tests PRIVATE pai_core)
  target_compile_definitions(pai_tests PRIVATE PAI_TRANSFORMER_TEST_DATA="${CMAKE_SOURCE_DIR}/transformer/tests")
  add_test(NAME pai_tests COMMAND pai_tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# ---- The game: Ogre 1.9 rendering, OIS input, Lua 5.2 scripting.
if(PAI_BUILD_GAME)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(OGRE REQUIRED IMPORTED_TARGET OGRE>=1.9 OGRE-Overlay>=1.9)
  pkg_check_modules(OIS REQUIRED IMPORTED_TARGET OIS)
  pkg_search_module(LUA REQUIRED IMPORTED_TARGET lua5.2 lua-5.2 lua52 lua>=5.2)
  pkg_get_variable(OGRE_PLUGIN_DIR OGRE plugindir)

  # plugins.cfg points Ogre at the plugin folder of this installation.
  configure_file(config/plugins.cfg.in ${CMAKE_BINARY_DIR}/plugins.cfg @ONLY)

  add_executable(pai_sim
    src/main.cpp
    src/engine/Application.cpp
    src/engine/AgentCameras.cpp
    src/engine/Input.cpp
    src/engine/LuaBridge.cpp
    src/engine/SceneView.cpp
    src/engine/Ui.cpp)
  target_link_libraries(pai_sim PRIVATE pai_core PkgConfig::OGRE PkgConfig::OIS PkgConfig::LUA)
  target_compile_definitions(pai_sim PRIVATE
    PAI_PLUGINS_CFG="${CMAKE_BINARY_DIR}/plugins.cfg"
    PAI_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/game.cfg")
endif()
