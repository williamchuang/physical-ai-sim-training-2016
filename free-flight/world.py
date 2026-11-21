"""A differentiable world: bots that move freely in 3D above a Möbius strip.

Each bot is a sphere driven by a thrust vector the policy chooses in the strip's
local frame (across, up, forward). Nothing holds a bot on the surface: the surface
is a solid it can hit. A hit is resolved like a wall (the bot is pushed back out
and loses its velocity into the surface) and is counted. Bots also collide with
each other. Coins float above the strip; touching one collects it and a new coin
appears elsewhere.

Every step is written in PyTorch, so the training loss can be differentiated
through the physics (backpropagation through time).
"""

from dataclasses import dataclass

import torch

from mobius import PERIOD, Mobius


def _safe_norm(x: torch.Tensor) -> torch.Tensor:
    """Length with a finite gradient at zero (a bot's distance to itself)."""
    return torch.sqrt((x * x).sum(-1) + 1e-9)


@dataclass
class WorldConfig:
    worlds: int = 16  # independent worlds simulated side by side
    bots: int = 4  # bots per world, sharing it
    coins: int = 60  # coins per world
    dt: float = 0.05
    bot_radius: float = 10.0
    max_thrust: float = 400.0  # units / s^2
    drag: float = 1.5  # 1 / s; top speed = max_thrust / drag
    coin_radius: float = 12.0
    coin_height: tuple = (20.0, 60.0)  # coins float this high above the surface
    start_height: float = 30.0
    ceiling: float = 120.0  # "along the strip" means below this height ...
    lane_limit: float = 0.5  # ... and within |v| <= lane_limit
    observed_coins: int = 3
    respawn_coins: bool = False  # False: a collected coin is gone, as in the 2016 demo


OBSERVATION_SIZE = 2 + 3 + 3 * 4 + 3 + 3  # height, lane | velocity | nearest coins (+ present flag) | nearest bot


class World:
    def __init__(self, config: WorldConfig, strip: Mobius = None, seed: int = 0, device="cpu", dtype=torch.float32):
        self.c = config
        self.strip = strip or Mobius()
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.device, self.dtype = device, dtype
        self.reset()

    # ------------------------------------------------------------------ setup

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen).to(self.device, self.dtype)

    def _coin_positions(self, n_shape):
        u = self._rand(*n_shape) * PERIOD
        v = (self._rand(*n_shape) * 2 - 1) * 0.8 * self.c.lane_limit
        lo, hi = self.c.coin_height
        h = lo + (hi - lo) * self._rand(*n_shape)
        up = self.strip.frame(u, v)[..., 1, :]
        return self.strip.position(u, v) + h.unsqueeze(-1) * up

    def reset(self):
        c, B, K = self.c, self.c.worlds, self.c.bots
        # Bots spread around the track, a little apart, flying at start_height.
        self.u = (torch.arange(K, dtype=self.dtype) * PERIOD / K).expand(B, K) + self._rand(B, K) * 0.3
        self.v = (self._rand(B, K) * 2 - 1) * 0.3
        f = self.strip.frame(self.u, self.v)
        self.pos = self.strip.position(self.u, self.v) + c.start_height * f[..., 1, :]
        self.vel = 100.0 * f[..., 2, :]
        self.coins = self._coin_positions((B, c.coins))
        self.alive = torch.ones(B, c.coins, dtype=torch.bool)
        self.in_contact = torch.zeros(B, K, dtype=torch.bool)
        self.pair_contact = torch.zeros(B, K, K, dtype=torch.bool)
        self.stats = {"coins": 0.0, "surface_hits": 0.0, "bot_hits": 0.0, "progress": 0.0, "outside": 0.0, "steps": 0}

    def detach(self):
        """Cut the gradient history (truncated backpropagation through time)."""
        self.u, self.v = self.u.detach(), self.v.detach()
        self.pos, self.vel = self.pos.detach(), self.vel.detach()

    # ------------------------------------------------------------------ sensing

    def observe(self):
        c, B, K = self.c, self.c.worlds, self.c.bots
        f = self.strip.frame(self.u, self.v)  # (B, K, 3, 3), rows right/up/forward
        to_local = lambda d: torch.einsum("bkij,bk...j->bk...i", f, d)
        height = ((self.pos - self.strip.position(self.u, self.v)) * f[..., 1, :]).sum(-1)

        # nearest coins, in the bot's frame
        rel = self.coins.unsqueeze(1) - self.pos.unsqueeze(2)  # (B, K, M, 3)
        dist = rel.norm(dim=-1) + (~self.alive).unsqueeze(1) * 1e9
        idx = dist.topk(c.observed_coins, dim=-1, largest=False).indices
        near = torch.gather(rel, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        present = torch.gather(self.alive.unsqueeze(1).expand(-1, K, -1), 2, idx).to(self.dtype).unsqueeze(-1)
        near_local = torch.cat(((to_local(near) / 300.0).clamp(-3, 3) * present, present), -1)

        # nearest other bot
        drel = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)  # [b, i, j] = pos_j - pos_i
        dd = _safe_norm(drel) + torch.eye(K, dtype=self.dtype) * 1e9
        j = dd.argmin(-1)
        other_pos = torch.gather(drel, 2, j[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        other_vel = torch.gather(self.vel.unsqueeze(1).expand(-1, K, -1, -1), 2, j[..., None, None].expand(-1, -1, 1, 3)).squeeze(2) - self.vel
        vmax = c.max_thrust / c.drag

        obs = torch.cat(
            (
                (height / 100.0).unsqueeze(-1),
                (self.v / c.lane_limit).unsqueeze(-1),
                to_local(self.vel) / vmax,
                near_local.flatten(-2),
                (to_local(other_pos) / 300.0).clamp(-3, 3),
                to_local(other_vel) / vmax,
            ),
            -1,
        )
        return obs

    # ------------------------------------------------------------------ physics

    def step(self, action):
        """action: (B, K, 3) in [-1, 1], thrust along (across, up, forward).
        Returns a dict of per-bot loss terms for this step."""
        c, K = self.c, self.c.bots
        f = self.strip.frame(self.u, self.v)
        thrust = c.max_thrust * torch.einsum("bki,bkij->bkj", action, f)
        u_before = self.u

        self.vel = self.vel + (thrust - c.drag * self.vel) * c.dt
        self.pos = self.pos + self.vel * c.dt
        self.u, self.v = self.strip.project(self.pos, self.u, self.v)

        # --- the surface is solid
        f = self.strip.frame(self.u, self.v)
        normal = f[..., 1, :]
        height = ((self.pos - self.strip.position(self.u, self.v)) * normal).sum(-1)
        depth = torch.relu(c.bot_radius - height)
        touching = depth > 0
        into = torch.relu(-(self.vel * normal).sum(-1)) * touching
        self.pos = self.pos + depth.unsqueeze(-1) * normal
        self.vel = self.vel + into.unsqueeze(-1) * normal
        surface_hits = touching & ~self.in_contact
        self.in_contact = touching

        # --- bots are solid to each other: push overlapping pairs apart, half each
        drel = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)  # pos_j - pos_i
        dist = _safe_norm(drel) + torch.eye(K, dtype=self.dtype) * 1e9
        overlap = torch.relu(2 * c.bot_radius - dist)
        push = (overlap / dist).unsqueeze(-1) * drel * 0.5
        self.pos = self.pos - push.sum(2)
        pair_contact = overlap > 0
        bot_hits = (pair_contact & ~self.pair_contact).sum(-1)  # new contacts, per bot
        self.pair_contact = pair_contact

        # --- coins
        cdist = (self.coins.unsqueeze(1) - self.pos.unsqueeze(2)).norm(dim=-1)  # (B, K, M)
        cdist = cdist + (~self.alive).unsqueeze(1) * 1e9
        taken = (cdist < c.bot_radius + c.coin_radius).detach()
        # a coin touched by two bots in the same step goes to the first
        first = taken & (taken.cumsum(1) == 1)
        collected = first.sum(-1)
        any_taken = taken.any(1)  # (B, M)
        if any_taken.any():
            if c.respawn_coins:
                fresh = self._coin_positions(self.coins.shape[:2])
                self.coins = torch.where(any_taken.unsqueeze(-1), fresh, self.coins)
            else:
                self.alive = self.alive & ~any_taken

        # --- loss terms (per bot)
        progress = (self.u - u_before) * self.strip.radius  # distance gained along the track
        nearest_coin = cdist.min(-1).values.clamp(max=1000.0)  # no pull once every coin is gone
        outside = torch.relu(self.v.abs() - c.lane_limit) * 10 + torch.relu(height - c.ceiling) / 50
        terms = {
            "progress": progress,
            "coin_distance": nearest_coin / 100.0,
            "surface_clearance": torch.relu(2.5 * c.bot_radius - height) ** 2 / c.bot_radius**2,
            "bot_clearance": (torch.relu(4 * c.bot_radius - dist) ** 2).sum(-1) / c.bot_radius**2,
            "outside": outside**2,
            "effort": (action**2).sum(-1),
            "collected": collected.to(self.dtype),
        }

        s = self.stats
        s["coins"] += collected.sum().item()
        s["surface_hits"] += surface_hits.sum().item()
        s["bot_hits"] += bot_hits.sum().item() / 2
        s["progress"] += progress.detach().sum().item()
        s["outside"] += (outside > 0).sum().item()
        s["steps"] += 1
        return terms
