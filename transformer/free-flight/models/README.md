# `transformer/free-flight/models/` — Trained free-flight transformer and records

| File | |
|---|---|
| `flight_transformer.pt` | the trained transformer, memory of 8 observations (PyTorch state dict, with its kind, memory, loss weights and world settings) |
| `flight_transformer.history.json` | per training episode: loss, coins, surface and bot hits, speed, time outside the corridor |
| `flight_transformer.eval.json` | the comparison with the free-flight network and with a memory of 1 observation (seeds 7, 8, 9) |

Trained 100 episodes with the same loss, seed and schedule as `free-flight/models/flight_policy.pt`.
