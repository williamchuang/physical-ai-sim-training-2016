# `src/engine/` — Ogre rendering, cameras, input, UI and Lua

The only code that uses Ogre 1.9, OIS and Lua 5.2. It draws the simulation and
supplies camera images; the rules stay in `src/game` and `src/sim`.

| File | |
|---|---|
| `Application.h/.cpp` | owns the engine objects; finds the project folder; runs input → scripts → game → scene → HUD each frame |
| `SceneView.h/.cpp` | generates the strip, cube and coin meshes; places them each frame; main camera and rear-view mirror |
| `AgentCameras.h/.cpp` | one small camera per bot, rendered into its own texture at 4x and averaged down to the sensor size, read back as grey values |
| `Input.h/.cpp` | keyboard through OIS: held keys for driving, single presses for toggles |
| `Ui.h/.cpp` | HUD (score, time, level, training status), pause menu, Lua console |
| `LuaBridge.h/.cpp` | the Lua commands available to scripts and the console |
| `OgreConvert.h` | conversions between the simulation's frames and Ogre's |
