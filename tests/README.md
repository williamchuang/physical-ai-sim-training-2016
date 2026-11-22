# `tests/` — Unit tests of the simulation core

A small harness (`Check.h`) with no dependencies beyond the standard library.

```sh
cmake -S . -B build && cmake --build build && ./build/pai_tests
```

| File | Checks |
|---|---|
| `core_tests.cpp` | strip geometry and frames, settings parsing, the game clock, oriented-box collisions and sweep-and-prune |
| `ai_tests.cpp` | the perceptron learns and clones a demonstrator; backpropagation matches finite differences; replay sampling; policy files |
| `sim_tests.cpp` | driving limits, coin pickup, a crash costs one point per contact, round rules, training then demo without the engine, the autopilot collects coins and avoids cubes |
| `test_main.cpp`, `Check.h` | the runner |

`pai_tests` also runs the transformer's check against PyTorch from
[`transformer/tests/`](../transformer/tests/); 33 tests in total.
