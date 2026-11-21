"""Möbius strip geometry in PyTorch (batched and differentiable).

Same track as the C++ simulation:
    X(u, v) = R ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2)),
u along the track (one lap on the same face is 4*pi), v across it, |v| <= half_width.
"""

import math

import torch

PERIOD = 4.0 * math.pi


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Mobius:
    def __init__(self, radius: float = 1000.0, half_width: float = 0.5):
        self.radius = radius
        self.half_width = half_width

    def position(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ring = 1 + v * torch.cos(u / 2)
        return self.radius * torch.stack((ring * torch.cos(u), ring * torch.sin(u), v * torch.sin(u / 2)), -1)

    def tangent_u(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ring = 1 + v * torch.cos(u / 2)
        dring = -0.5 * v * torch.sin(u / 2)
        return self.radius * torch.stack(
            (dring * torch.cos(u) - ring * torch.sin(u), dring * torch.sin(u) + ring * torch.cos(u), 0.5 * v * torch.cos(u / 2)),
            -1,
        )

    def tangent_v(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        c = torch.cos(u / 2)
        return self.radius * torch.stack((c * torch.cos(u), c * torch.sin(u), torch.sin(u / 2)), -1)

    def frame(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rows (right, up, forward): right across the track, up along the normal,
        forward = right x up along the direction of travel. Shape (..., 3, 3)."""
        tu, tv = self.tangent_u(u, v), self.tangent_v(u, v)
        right = _normalize(tv)
        up = _normalize(torch.cross(tu, tv, dim=-1))
        forward = torch.cross(right, up, dim=-1)
        return torch.stack((right, up, forward), -2)

    def project(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, iterations: int = 3):
        """Closest point on the surface to x, by Gauss-Newton from the guess (u, v).

        u is kept continuous (not wrapped), so a body that stays near the strip keeps
        the face it started on; the steps are differentiable."""
        for _ in range(iterations):
            r = self.position(u, v) - x
            tu, tv = self.tangent_u(u, v), self.tangent_v(u, v)
            a, b, c = (tu * tu).sum(-1), (tu * tv).sum(-1), (tv * tv).sum(-1)
            gu, gv = (tu * r).sum(-1), (tv * r).sum(-1)
            det = (a * c - b * b).clamp_min(1e-9)
            u = u - (c * gu - b * gv) / det
            v = v - (a * gv - b * gu) / det
        return u, v
