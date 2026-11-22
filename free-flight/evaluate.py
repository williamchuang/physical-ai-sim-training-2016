"""Measure a flight policy against an untrained one on the same worlds.

    python evaluate.py --policy models/flight_policy.pt

Each policy flies 60 s in 32 worlds with the same seeds; coins do not come back
once collected. Reports per bot and minute: coins, hits on the surface, hits
between bots, speed along the track, and how often a bot left the corridor.
"""

import argparse
import json
from pathlib import Path

import torch

from policy import FlightPolicy, make_policy
from world import World, WorldConfig


def load_policy(path: Path) -> FlightPolicy:
    state = torch.load(path, map_location="cpu", weights_only=True)
    policy = make_policy(state.get("kind", "mlp"), state.get("memory", 8))
    policy.load_state_dict(state["state_dict"])
    policy.eval()
    return policy


@torch.no_grad()
def measure(policy, seconds: float = 60.0, worlds: int = 32, bots: int = 4, seed: int = 7) -> dict:
    config = WorldConfig(worlds=worlds, bots=bots)
    world = World(config, seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    for _ in range(int(seconds / config.dt)):
        world.step(policy(world.observe()))
    s = world.stats
    bot_minutes = worlds * bots * seconds / 60
    return {
        "coins_per_bot_min": s["coins"] / bot_minutes,
        "surface_hits_per_bot_min": s["surface_hits"] / bot_minutes,
        "bot_hits_per_bot_min": s["bot_hits"] / bot_minutes,
        "speed_along_track": s["progress"] / (worlds * bots * seconds),
        "outside_fraction": s["outside"] / (worlds * bots * s["steps"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", type=Path, default=Path("models/flight_policy.pt"))
    p.add_argument("--out", type=Path, default=None, help="write the results as JSON")
    a = p.parse_args()

    torch.manual_seed(0)
    results = {"untrained": measure(FlightPolicy()), "trained": measure(load_policy(a.policy))}
    keys = list(results["trained"])
    print(f"{'':12s}" + "".join(f"{k:>26s}" for k in keys))
    for name, r in results.items():
        print(f"{name:12s}" + "".join(f"{r[k]:26.3f}" for k in keys))
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
