# `config/presets/` — Ready-made settings for each mode

Each preset changes a few keys of `config/game.cfg`; everything else keeps its default.

| Preset | What it runs |
|---|---|
| `compare.cfg` | you and the trained bot on the same strip, scores side by side |
| `train.cfg` | you drive with the keys while the network learns from the cube's camera (the 2016 way) |
| `train-autopilot.cfg` | the autopilot drives and labels, the network drives half the time (DAgger); used for `models/policy.txt` |
| `demo-three-bots.cfg` | three trained bots among sixteen obstacle cubes, as in the largest 2016 build |
| `agent-view.cfg` | the demo seen through the bot's own camera |
| `evaluate.cfg` | one unattended 90 s round that prints the scores and quits; used for the result tables |

```sh
./build/pai_sim config/presets/train.cfg
```
