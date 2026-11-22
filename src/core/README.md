# `src/core/` — Track geometry, settings and clock

| File | |
|---|---|
| `MobiusStrip.h/.cpp` | the track X(u, v), its tangents and local frame (across, up, forward) |
| `Vec3.h` | a small 3-vector and right-handed frames |
| `Config.h/.cpp` | the `key = value` reader and `GameConfig`, every setting with its default |
| `GameClock.h` | game time that advances only while a round runs |

Nothing here knows about rendering, so the tests can use it directly.
