"""Demo and compare views for the free-flight world (matplotlib, follow camera).

    python viewer.py demo                 # the trained bots fly alone; coins vanish when taken
    python viewer.py compare              # add a bot you fly with the keyboard, scored side by side
    python viewer.py demo --save demo.mp4 # write a video instead of opening a window (needs ffmpeg)

Keys in compare mode (your bot is red):
    W / S  thrust forward / back      A / D  thrust left / right
    R / F  thrust up / down           Space  pause
"""

import argparse
from pathlib import Path

import matplotlib

import numpy as np
import torch

from evaluate import load_policy
from mobius import Mobius
from world import World, WorldConfig

VIEW = 220.0  # half size of the box around the followed bot, in world units


class Session:
    def __init__(self, mode: str, policy_path: Path, bots: int, coins: int, seed: int):
        self.mode = mode
        self.config = WorldConfig(worlds=1, bots=bots + (1 if mode == "compare" else 0), coins=coins)
        self.world = World(self.config, seed=seed)
        self.policy = load_policy(policy_path)
        self.human = 0 if mode == "compare" else None
        self.keys = set()
        self.scores = np.zeros(self.config.bots, dtype=int)
        self.paused = False
        self.trail = []

    def human_action(self) -> torch.Tensor:
        k = self.keys
        across = (1.0 if "a" in k else 0.0) - (1.0 if "d" in k else 0.0)  # +across points to the left
        up = (1.0 if "r" in k else 0.0) - (1.0 if "f" in k else 0.0)
        forward = (1.0 if "w" in k else 0.0) - (1.0 if "s" in k else 0.0)
        return torch.tensor([across, up, forward])

    @torch.no_grad()
    def step(self):
        if self.paused:
            return
        action = self.policy(self.world.observe())
        if self.human is not None:
            action[0, self.human] = self.human_action()
        terms = self.world.step(action)
        self.scores += terms["collected"][0].numpy().astype(int)

    def focus(self) -> int:
        return self.human if self.human is not None else 0


def _screen_angle(ax, at, direction) -> float:
    """Angle on screen, in degrees, of a short world-space arrow starting at `at`."""
    from mpl_toolkits.mplot3d import proj3d

    m = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(*at, m)
    x1, y1, _ = proj3d.proj_transform(*(at + 10.0 * direction), m)
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def draw(ax, session: Session, strip: Mobius):
    w = session.world
    f = session.focus()
    centre = w.pos[0, f].numpy()
    u0, v0 = float(w.u[0, f]), float(w.v[0, f])
    frame = strip.frame(w.u[0, f], w.v[0, f]).numpy()  # rows: across, up, forward
    session.trail.append(centre.copy())
    ax.cla()

    # the strip around the followed bot, striped across so that motion shows
    uu, vv = torch.meshgrid(torch.linspace(u0 - 0.25, u0 + 0.45, 57), torch.linspace(-0.5, 0.5, 9), indexing="ij")
    surface = strip.position(uu, vv).numpy()
    band = ((uu / 0.025).floor() % 2).numpy()  # one entry per grid point; matplotlib uses [:-1, :-1]
    colours = np.where(band[..., None] > 0, [0.45, 0.45, 0.47, 1.0], [0.62, 0.62, 0.64, 1.0])
    colours[:, 0] = colours[:, -2] = [0.15, 0.12, 0.12, 1.0]  # dark kerbs on both edges
    ax.plot_surface(surface[..., 0], surface[..., 1], surface[..., 2], facecolors=colours, linewidth=0,
                    shade=False, zorder=1)

    coins = w.coins[0][w.alive[0]].numpy()
    near = np.abs(coins - centre).max(-1) < 2 * VIEW
    ax.scatter(*coins[near].T, color="gold", edgecolor="darkgoldenrod", s=70, depthshade=False, zorder=3)

    trail = np.array(session.trail[-60:])
    ax.plot(*trail.T, color="red" if session.human is not None else "royalblue", linewidth=1.5, alpha=0.6, zorder=2)
    for i, p in enumerate(w.pos[0].numpy()):
        colour = "red" if i == session.human else "royalblue"
        ax.scatter(*p, color=colour, edgecolor="black", s=110, depthshade=False, zorder=4)

    # look from behind and above the followed bot, along its direction of travel
    eye = -frame[2] + 0.55 * frame[1]
    eye /= np.linalg.norm(eye)
    elev, azim = np.degrees(np.arcsin(eye[2])), np.degrees(np.arctan2(eye[1], eye[0]))
    look = centre + 0.35 * VIEW * frame[2]
    ax.set_xlim(look[0] - VIEW, look[0] + VIEW)
    ax.set_ylim(look[1] - VIEW, look[1] + VIEW)
    ax.set_zlim(look[2] - VIEW, look[2] + VIEW)
    ax.set_box_aspect((1, 1, 1))
    # Roll the view so the strip's "up" points up on screen, on either face of the strip.
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.view_init(elev=elev, azim=azim, roll=90.0 - _screen_angle(ax, look, frame[1]))
    ax.set_axis_off()

    t = w.stats["steps"] * w.c.dt
    if session.human is not None:
        bots = session.scores[1:]
        title = f"You {session.scores[0]}    Best bot {bots.max() if len(bots) else 0}"
    else:
        title = "Bots " + "  ".join(str(s) for s in session.scores)
    hits = int(w.stats["surface_hits"])
    ax.set_title(f"{title}    coins left {int(w.alive.sum())}    surface hits {hits}    {t:4.1f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["demo", "compare"])
    p.add_argument("--policy", type=Path, default=Path("models/flight_policy.pt"))
    p.add_argument("--bots", type=int, default=1, help="trained bots")
    p.add_argument("--coins", type=int, default=60)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--save", type=Path, default=None, help="write a video (.mp4 or .gif) instead of a window")
    a = p.parse_args()

    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
        plt.rcParams[key] = []  # free every key for driving

    session = Session(a.mode, a.policy, a.bots, a.coins, a.seed)
    strip = session.world.strip
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)  # draw in zorder: strip, trail, coins, bots

    def on_press(e):
        if e.key == " ":
            session.paused = not session.paused
        elif e.key:
            session.keys.add(e.key.lower())

    def on_release(e):
        if e.key:
            session.keys.discard(e.key.lower())

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    frames = int(a.seconds / session.config.dt)

    def update(_):
        session.step()
        draw(ax, session, strip)

    anim = FuncAnimation(fig, update, frames=frames, interval=session.config.dt * 1000, repeat=False)
    if a.save:
        anim.save(a.save, fps=int(1 / session.config.dt), dpi=90)
        print(f"wrote {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
