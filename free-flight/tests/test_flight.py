"""Checks for the free-flight variant.  Run:  python -m pytest tests   (or  python tests/test_flight.py)"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobius import PERIOD, Mobius  # noqa: E402
from policy import FlightPolicy  # noqa: E402
from train import step_loss  # noqa: E402
from world import OBSERVATION_SIZE, World, WorldConfig  # noqa: E402

D = torch.float64


def test_strip_matches_parametrisation_and_has_one_side():
    s = Mobius()
    u, v = torch.tensor(1.1, dtype=D), torch.tensor(0.3, dtype=D)
    ring = 1 + 0.3 * math.cos(0.55)
    expected = torch.tensor([1000 * ring * math.cos(1.1), 1000 * ring * math.sin(1.1), 300 * math.sin(0.55)], dtype=D)
    assert torch.allclose(s.position(u, v), expected)
    assert torch.allclose(s.position(u + PERIOD / 2, -v), s.position(u, v), atol=1e-9)
    assert torch.allclose(s.frame(u + PERIOD / 2, -v)[1], -s.frame(u, v)[1])


def test_frame_is_orthonormal():
    s = Mobius()
    f = s.frame(torch.linspace(0, PERIOD, 50, dtype=D), torch.full((50,), 0.2, dtype=D))
    eye = torch.eye(3, dtype=D).expand(50, 3, 3)
    assert torch.allclose(f @ f.transpose(-1, -2), eye, atol=1e-12)


def test_projection_finds_the_closest_point():
    s = Mobius()
    u = torch.tensor([0.3, 2.0, 5.0, 9.0], dtype=D)
    v = torch.tensor([0.1, -0.3, 0.4, 0.0], dtype=D)
    x = s.position(u, v) + 25.0 * s.frame(u, v)[:, 1]
    pu, pv = s.project(x, u + 0.01, v - 0.02, iterations=6)
    assert torch.allclose(pu, u, atol=1e-6) and torch.allclose(pv, v, atol=1e-6)


def test_world_step_gradients_match_finite_differences():
    """Backpropagation through the physics, sensors and loss, checked by gradcheck."""
    torch.manual_seed(0)
    config = WorldConfig(worlds=2, bots=3, coins=8)
    policy = FlightPolicy(hidden=8).double()

    names = [n for n, _ in policy.named_parameters()]

    def rollout(*params):
        world = World(config, seed=3, dtype=D)
        loss = 0
        for _ in range(4):
            action = torch.func.functional_call(policy, dict(zip(names, params)), (world.observe(),))
            loss = loss + step_loss(world.step(action), config)
        return loss

    params = tuple(p.detach().clone().requires_grad_(True) for p in policy.parameters())
    assert torch.autograd.gradcheck(rollout, params, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_surface_is_solid():
    config = WorldConfig(worlds=1, bots=2, coins=1)
    world = World(config, seed=1)
    down = torch.tensor([0.0, -1.0, 0.0]).expand(1, 2, 3)
    for _ in range(200):
        world.step(down)
    height = ((world.pos - world.strip.position(world.u, world.v)) * world.strip.frame(world.u, world.v)[..., 1, :]).sum(-1)
    assert (height >= config.bot_radius - 1e-3).all()
    assert world.stats["surface_hits"] == 2  # each bot landed once and stayed in contact


def test_collected_coins_vanish():
    config = WorldConfig(worlds=1, bots=1, coins=5)
    world = World(config, seed=2)
    world.coins[0, 1] = world.pos[0, 0]
    terms = world.step(torch.zeros(1, 1, 3))
    assert terms["collected"].item() == 1
    assert world.alive.sum().item() == 4
    assert world.observe().shape == (1, 1, OBSERVATION_SIZE)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
