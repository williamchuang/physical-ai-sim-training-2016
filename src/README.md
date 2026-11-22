# `src/` — C++17 source of the restored simulation

The simulation does not depend on the engine: everything except `engine/` builds
and is tested without Ogre (`cmake -S . -B build -DPAI_BUILD_GAME=OFF`).

| Folder | | Depends on |
|---|---|---|
| [`core/`](core/) | Möbius strip geometry, settings, game clock, vectors | standard library |
| [`physics/`](physics/) | oriented boxes: separating-axis test, sweep-and-prune | core |
| [`ai/`](ai/) | the perceptron, replay buffer, imitation learner | standard library |
| [`sim/`](sim/) | cubes, coins, their rules, the autopilot | core, physics |
| [`game/`](game/) | modes, decisions, training, recording, scoring | sim, ai, transformer/cpp |
| [`engine/`](engine/) | Ogre scene, cameras, OIS input, HUD, menu, Lua | everything above, Ogre, OIS, Lua |
| `main.cpp` | starts the application with a settings file | engine |

One frame: input → Lua scripts → game rules (decisions every `decision_interval`,
physics in steps of at most 1/60 s) → scene → HUD.
