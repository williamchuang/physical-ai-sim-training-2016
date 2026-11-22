# `src/sim/` — Cubes, coins, rules and the autopilot

| File | |
|---|---|
| `Car.h/.cpp` | a cube: place on the strip (u, v), speed, score; the four actions and how a key press moves it |
| `World.h/.cpp` | the cubes and coins on the strip: driving, collisions, coin pickup (a collected coin is gone unless `respawn_coins`) |
| `Autopilot.h/.cpp` | a scripted expert pressing the same keys a person would: steer to the nearest reachable coin in view, away from a cube close ahead, back from the edge |

The autopilot only uses what the bot's camera could see (coins ahead, within the
camera's angle), so a network copying it is not asked to know the invisible.
