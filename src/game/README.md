# `src/game/` — Modes, decisions, training and scoring

`Game` runs the rules of one round independently of rendering:

- **demo** and **compare**: every `decision_interval` each bot's camera image goes
  to its policy (the perceptron or the transformer) and the chosen key is held
  until the next decision;
- **train**: you or the autopilot drive; each (image, key) goes to the imitation
  learner; with `dagger_share` the network drives part of the time;
- **recording**: with `record_file`, each camera frame and the demonstrator's key
  is written for PyTorch training ([`transformer/training/`](../../transformer/training/));
- physics advances in steps of at most 1/60 s; scores, levels and the round timer.

| File | |
|---|---|
| `Game.h/.cpp` | the `Game` class and the `Perception` interface the engine implements |
