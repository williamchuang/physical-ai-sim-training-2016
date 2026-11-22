# `free-flight/models/` — Trained flight policy and its records

| File | |
|---|---|
| `flight_policy.pt` | the trained network 23 → 64 → 64 → 3 (PyTorch state dict, with the loss weights and world settings) |
| `flight_policy.history.json` | per training episode: loss, coins, surface and bot hits, speed, time outside the corridor |
| `flight_policy.eval.json` | trained against untrained on the same worlds (`python evaluate.py`) |

The free-flight transformer and its records are in
[`transformer/free-flight/models/`](../../transformer/free-flight/models/).
