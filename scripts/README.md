# `scripts/` — Lua scripts run by the game

The game runs Lua 5.2. A script is a table with a `Think(self, dt)` function,
called every frame once added with `addScript`.

| File | |
|---|---|
| `ScriptManager.lua` | the small script system: `Script.create()`, `addScript`, and the per-frame loop |
| `startup.lua` | runs at start-up (key `startup_script`); holds an example logger, inactive |
| `evaluate.lua` | starts a round at once, prints every cube's score when it ends and quits; used by the evaluation and recording settings |

In the game, `` ` `` opens the Lua console and `help()` lists the commands
(`Start`, `cars`, `getTrackPosition`, `setLane`, `setSpeed`, `score`, `coinsLeft`, `savePolicy`, ...).
