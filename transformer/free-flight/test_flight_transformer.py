"""Checks for the free-flight transformer. Run from the repository root:
    python -m pytest transformer/free-flight
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "free-flight")]

from flight_transformer import FlightTransformer  # noqa: E402
from train import step_loss  # noqa: E402
from world import OBSERVATION_SIZE, World, WorldConfig  # noqa: E402


def test_transformer_remembers_and_learns_through_the_world():
    torch.manual_seed(3)
    config = WorldConfig(worlds=2, bots=2, coins=8)
    world = World(config, seed=5)
    policy = FlightTransformer(OBSERVATION_SIZE, history=4)
    first = policy(world.observe())
    assert first.shape == (2, 2, 3) and first.abs().max() <= 1
    assert policy._memory.shape == (4, 4, OBSERVATION_SIZE)  # first view repeated
    # The answer depends on what came before: same view, different memory -> different thrust.
    with torch.no_grad():
        for _ in range(5):  # fly a little so the views differ
            world.step(policy(world.observe()))
    obs = world.observe()
    with_memory = policy(obs)
    policy.reset()
    without = policy(obs)
    assert not torch.allclose(with_memory, without)
    # Backpropagation reaches every weight through the simulator.
    policy.reset()
    loss = sum(step_loss(world.step(policy(world.observe())), config) for _ in range(3))
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())



def test_shipped_transformer_loads_and_flies():
    from evaluate import load_policy, measure

    policy = load_policy(HERE / "models" / "flight_transformer.pt")
    result = measure(policy, seconds=5.0, worlds=2, bots=2)
    assert result["surface_hits_per_bot_min"] == 0
