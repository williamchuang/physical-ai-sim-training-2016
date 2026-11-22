"""Train the flight policy by backpropagation through the simulator.

The loss rewards progress along the strip and closing in on coins, and penalises
coming near the surface, near other bots, and leaving the corridor above the strip.
Because the world is differentiable, its gradient flows from the loss through the
physics and the sensors back into the network (truncated every `window` steps).

    python train.py --episodes 300 --out models/flight_policy.pt
"""

import argparse
import json
import time
from pathlib import Path

import torch

from policy import make_policy
from world import World, WorldConfig

WEIGHTS = {
    "progress": -0.5,
    "coin_distance": 1.0,
    "surface_clearance": 5.0,
    "bot_clearance": 3.0,
    "outside": 5.0,
    "effort": 0.01,
}


def step_loss(terms, config: WorldConfig):
    per_step_distance = config.max_thrust / config.drag * config.dt
    loss = WEIGHTS["progress"] * terms["progress"] / per_step_distance
    for key in ("coin_distance", "surface_clearance", "bot_clearance", "outside", "effort"):
        loss = loss + WEIGHTS[key] * terms[key]
    return loss.mean()


def train(episodes: int, steps: int, window: int, lr: float, seed: int, out: Path, log_every: int = 10,
          time_limit: float = float("inf"), kind: str = "mlp", memory: int = 8):
    """Trains, saving a checkpoint every few episodes; a second call with the same
    arguments resumes where the first one stopped. Returns True when finished."""
    torch.manual_seed(seed)
    config = WorldConfig()
    policy = make_policy(kind, memory)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, episodes)
    history, first = [], 0
    checkpoint = out.with_suffix(".checkpoint.pt")
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        history, first = state["history"], state["episode"]
        print(f"resuming at episode {first}")
    start = time.time()

    def save_checkpoint(next_episode):
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(), "history": history, "episode": next_episode}, checkpoint)

    for episode in range(first, episodes):
        if time.time() - start > time_limit:
            save_checkpoint(episode)
            print(f"time limit: stopped before episode {episode}; run again to resume")
            return False
        world = World(config, seed=seed * 100003 + episode)
        policy.reset()
        total = 0.0
        for t0 in range(0, steps, window):
            world.detach()
            policy.detach()
            loss = 0.0
            for _ in range(window):
                loss = loss + step_loss(world.step(policy(world.observe())), config)
            loss = loss / window
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        schedule.step()

        s = world.stats
        bot_minutes = config.worlds * config.bots * s["steps"] * config.dt / 60
        row = {
            "episode": episode,
            "loss": total / (steps // window),
            "coins_per_bot_min": s["coins"] / bot_minutes,
            "surface_hits_per_bot_min": s["surface_hits"] / bot_minutes,
            "bot_hits_per_bot_min": s["bot_hits"] / bot_minutes,
            "speed_along_track": s["progress"] / (config.worlds * config.bots * s["steps"] * config.dt),
            "outside_fraction": s["outside"] / (config.worlds * config.bots * s["steps"]),
        }
        history.append(row)
        if episode % log_every == 0 or episode == episodes - 1:
            print(
                f"ep {episode:4d}  loss {row['loss']:7.3f}  coins/bot-min {row['coins_per_bot_min']:6.2f}  "
                f"surface hits {row['surface_hits_per_bot_min']:5.2f}  bot hits {row['bot_hits_per_bot_min']:5.2f}  "
                f"speed {row['speed_along_track']:6.1f}  outside {row['outside_fraction']:.3f}  ({time.time() - start:.0f} s)",
                flush=True,
            )
        if episode % 5 == 4:
            save_checkpoint(episode + 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": kind, "memory": memory, "state_dict": policy.state_dict(), "weights": WEIGHTS, "config": config.__dict__}, out)
    out.with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    checkpoint.unlink(missing_ok=True)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=400, help="steps per episode (dt = 0.05 s)")
    p.add_argument("--window", type=int, default=25, help="steps per truncated backpropagation window")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=2016)
    p.add_argument("--out", type=Path, default=Path("models/flight_policy.pt"))
    p.add_argument("--kind", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--memory", type=int, default=8, help="observations the transformer sees")
    p.add_argument("--minutes", type=float, default=float("inf"), help="stop (resumably) after this long")
    a = p.parse_args()
    train(a.episodes, a.steps, a.window, a.lr, a.seed, a.out, time_limit=a.minutes * 60, kind=a.kind, memory=a.memory)
