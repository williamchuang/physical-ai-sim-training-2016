# Physical AI Simulation Training (2016)

Training a robot inside a virtual world before it ever touches the real one.

A bot in this project looks at the world through its own small camera, and a
neural network turns what it sees into driving commands. The network learns
from demonstrations inside a 3D graphics engine with physics, not on a road,
so a learning machine can make its mistakes where no vehicle is damaged and no
person is hurt: in bad weather, in traffic, in a crash.

**The cubes, coins and scores are not the point.** The project began in 2016 in
a Game Engineering course whose rule was that every project had to be a game,
so the idea was given a game's clothes: a cube for the robot, coins for goals,
a score for how well it does. Underneath, the subject is physical AI: robots and
self-driving vehicles trained in simulation (VR) and judged by what they do there.

The project grew out of an idea William H. Chuang brought to the course in 2016,
and it became real through the work of the whole team, Enrique Gonzalez and
Prof. David Galles, to whom he remains grateful. At the time, the team knew of no
comparable work; it was new to them, and it is shared here in that spirit.

![A trained bot (blue) on the Möbius strip, with coins ahead](docs/demo.png)

## Two variations

| | `/` Restored 2016 simulation | [`free-flight/`](free-flight/) |
|---|---|---|
| Body | a cube on the surface of the strip | a sphere moving freely in 3D above it |
| Senses | its own camera: a 20 x 20 grey image | the strip's frame, the nearest coins and the nearest other bot |
| Learns by | imitation: a demonstrator presses the keys, the network copies them | backpropagation through a differentiable simulator |
| Engine | C++17, Ogre 1.9, OIS, Lua 5.2 (as in 2016) | Python, PyTorch |
| Modes | train, demo, compare | train, demo, compare |

Both variations ship with a trained policy, so the demos run at once.

### The track

Every variation drives on a Möbius strip of radius R = 1000,

    X(u, v) = R ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2)),   |v| <= 1/2,

a surface with one side: half a lap later the bot is on the "other" face, upside
down to an outside observer, so the controller must work in the strip's local
frame (across, up, forward) instead of a fixed world frame.

## 2016 in context

The project began in February–March 2016; its first commits date from April, and
the work ran through September. Public milestones around it, for orientation:

- 2015: DeepDriving (Princeton) trains a driving network inside the TORCS racing game.
- April 2016: NVIDIA's end-to-end network learns steering from camera images of real
  driving, tested in simulation and on a car; OpenAI releases Gym.
- August 2016: "Playing for Data" (TU Darmstadt, Intel Labs) labels frames from
  Grand Theft Auto V to train vision models.
- December 2016: OpenAI Universe trains agents on games and applications.
- February 2017: Microsoft AirSim, a drone and car simulator built on Unreal Engine.
- May 2017: NVIDIA announces the Isaac robot simulator, built on video-game and
  graphics technology.
- June 2017: the Transformer architecture is published; June 2018: the first GPT.
- March 2018: NVIDIA DRIVE Constellation, simulation for testing self-driving cars.

The team did not know of any of the 2015 and 2016 work above at the time. Neural
networks were already trained on GPUs, but computer vision was far younger than
today, and the Transformer and GPT did not yet exist. As far as William H. Chuang,
Enrique Gonzalez and Prof. David Galles knew in 2016, earlier neural-network work had
not joined the two fields this project brought together: training models, and
building games with graphics and physics engines.

William H. Chuang had attended the Bay Area deep learning meetup he recalls as the
oldest and largest since its first meeting, before 2016, and later a second one. Few
in software took neural networks seriously then; many engineers he met saw them as
black boxes without a future. In 2016 he brought up, in discussions at these meetups,
the idea of training bots in game environments, and later the project he was working
on. As he recalls, most attention there went to chatbots rather than to robots,
vehicles or anything physical, and the engineers he spoke with, among them at least
two or three from NVIDIA, were not yet convinced that a game engine could serve as a
training ground. About a year later, NVIDIA introduced its Isaac robot simulator,
built on video-game and graphics technology.

## Variation 1: the restored 2016 simulation

The bot cube rides on the strip. Its camera image goes through the 2016 network,
a two-layer perceptron (pixels → sigmoid hidden layer → softmax over four keys:
faster, right, left, slower), written from scratch without a neural-network library.

- **demo**: the trained bot drives alone and collects the coins; a collected coin is gone.
- **compare**: the same strip with a second cube on the keyboard, scores side by side.
- **train**: a demonstrator drives the cube while the network learns from its camera.
  The demonstrator is a person at the keys (the 2016 way) or the built-in autopilot.

### Results

A 90-second demo round: one bot, 74 coins, coins do not come back, three seeds.

| Controller | Seed 101 | Seed 202 | Seed 303 | Mean |
|---|---|---|---|---|
| untrained network | 0 | 0 | 2 | 0.7 |
| driving straight, never steering | 3 | 6 | 4 | 4.3 |
| **trained bot** (`models/policy.txt`) | **26** | **17** | **28** | **23.7** |
| the autopilot it learned from | 62 | 51 | 71 | 61.3 |

The bot sees only its 20 x 20 grey camera image. The shipped policy learned in
the rendered engine from the autopilot (`config/presets/train-autopilot.cfg`):
about 18 minutes of driving, about 20 decisions a second, with the network at the
wheel half the time (DAgger). It was chosen by this evaluation among the saved
checkpoints; training on at the same learning rate became unstable. The bot and
untrained rows come from the rendered engine (`config/presets/evaluate.cfg`); the
straight and autopilot rows from the same rules without rendering.

### Build and run

Linux (Ubuntu) packages: `cmake g++ pkg-config libogre-1.9-dev libois-dev liblua5.2-dev`.
On macOS and Windows, any Ogre 1.9 / OIS / Lua 5.2 installation that pkg-config can find works.

```sh
cmake -S . -B build
cmake --build build
./build/pai_tests                                  # unit tests, no window
./build/pai_sim                                    # demo (config/game.cfg)
./build/pai_sim config/presets/compare.cfg         # you and the bot
./build/pai_sim config/presets/train.cfg           # teach it with your keys
```

`cmake -S . -B build -DPAI_BUILD_GAME=OFF` builds the simulation core and its
tests without Ogre.

### Controls

| Key | |
|---|---|
| T / G (or ↑ / ↓) | faster / slower |
| F / H (or ← / →) | left / right |
| P | menu (Start, Continue, Quit) |
| ` | Lua console; `help()` lists the commands |
| 6 / 7 | collisions off / on |
| 8 / 9 / 0 | pause / resume / reset the clock |
| Esc | close the console, or quit |

### Configuration

`config/game.cfg` holds every setting; `config/presets/` holds ready-made ones.

| Key | Default | |
|---|---|---|
| `mode` | demo | `train`, `demo` or `compare` |
| `demonstrator` | keyboard | train mode: `keyboard` or `autopilot` |
| `dagger_share` | 0 | with the autopilot: share of decisions the network drives while the autopilot labels |
| `camera_view` | chase | `chase` (behind the cube) or `agent` (through the bot's camera) |
| `neural_agents`, `scripted_agents` | 1, 0 | bots, and lane-keeping obstacle cubes |
| `coins`, `respawn_coins` | 74, false | collected coins vanish unless `respawn_coins` |
| `sensor_width`, `sensor_height`, `hidden_units` | 20, 20, 32 | camera and network size (2016: 3 x 3 and 6) |
| `decision_interval` | 0.1 | seconds between decisions |
| `learning_rate`, `batch_size`, `replay_capacity` | 0.05, 5, 2048 | training |
| `policy_file`, `autosave_seconds` | models/policy.txt, 15 | where the policy is kept |
| `startup_script` | scripts/startup.lua | Lua run at start-up |

Presets: `train.cfg`, `train-autopilot.cfg`, `compare.cfg`, `demo-three-bots.cfg`
(three bots among sixteen obstacle cubes, as in the largest 2016 build),
`agent-view.cfg`, `evaluate.cfg` (one unattended round that prints the scores).

### How the code is organised

```
src/core      Möbius strip geometry, settings, game clock       (no engine types)
src/physics   oriented boxes: separating-axis test, sweep-and-prune broadphase
src/ai        the perceptron, replay buffer, imitation learner, policy files
src/sim       cubes, coins and their rules; the autopilot
src/game      modes, decisions, training, scoring
src/engine    Ogre scene, cameras read back from render textures, OIS input, HUD, menu, Lua
tests         unit tests, including a finite-difference check of backpropagation
```

The simulation does not depend on the engine, so it is tested on its own,
and the engine only draws it and supplies camera images. What changed from 2016:

- one program with settings in place of five copies of the source tree;
- positions advance from the frame time, so a change of speed never makes a cube jump;
- collisions use oriented boxes and a sweep-and-prune broadphase, and a crash
  costs a point once per contact instead of once per frame;
- each camera renders into its own small texture, read back in memory instead of
  through image files;
- training samples go into a ring buffer and minibatches are drawn from it
  (Floyd's sampling), weights sit in contiguous arrays, policies are saved atomically;
- the track and coin shapes are generated from the same formula the physics uses;
- DAgger: while the autopilot teaches, the network also drives part of the time
  and the autopilot labels what it should have done, so the network also learns
  to recover from its own mistakes.

## Variation 2: free flight

See [`free-flight/README.md`](free-flight/README.md). Nothing holds the bots on the
strip: the surface is solid, a bot that touches it is stopped and the contact is
counted, and a policy trained in PyTorch has to follow the strip, collect the coins
and keep clear of the surface and of the other bots.

Over 60 s, the trained bots collect 4.1 coins per minute each, about six times a
hand-written cruise controller, with no contact with the surface or with each
other ([details](free-flight/README.md#results)).

## History

The commits from April to September 2016 are the original course project. In 2016 the team trained and compared camera inputs of 3×3, 8×8,
10×10, 20×20 and 80×80 pixels; the loss and parameter records of these runs were
committed in September 2016. Larger inputs, up to 256×256 and 1024-sized, were later
run on local machines and shown in demos to Prof. David Galles. The code was kept as
one program, since these runs differ only in the camera size, a single setting; the
restored code exposes it as `sensor_width` and `sensor_height`. The
restoration in 2026 rewrote the C++ code for clarity with the help of an AI coding
assistant, keeping the same libraries, and added the free-flight variation.

## Credits

2016 authors: William H. Chuang, Enrique Gonzalez and Prof. David Galles.

The DejaVu Sans font in `media/fonts/` has its own license (`media/fonts/LICENSE-DejaVu.txt`).
Everything else: MIT License, see `LICENSE`.
