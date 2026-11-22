# `src/physics/` — Oriented-box collisions

| File | |
|---|---|
| `Collision.h/.cpp` | oriented bounding boxes; the separating-axis test over the 15 candidate axes; a sweep-and-prune broadphase; the minimum push that separates two boxes |

A crash costs a point once per contact, not once per frame.
