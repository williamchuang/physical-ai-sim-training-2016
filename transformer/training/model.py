"""The bot cube's transformer policy in PyTorch, and its export to the game's text format.

The C++ game (src/ai/Transformer.cpp) runs the same network forward-only from
the exported file; transformer/tests/transformer_fixture.txt checks that both agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Shape:
    frames: int = 4
    height: int = 20
    width: int = 20
    patch: int = 5
    model: int = 32
    heads: int = 2
    layers: int = 2
    feed_forward: int = 64
    outputs: int = 4

    @property
    def patches_per_frame(self) -> int:
        return (self.height // self.patch) * (self.width // self.patch)

    @property
    def tokens(self) -> int:
        return self.frames * self.patches_per_frame


def patchify(images: torch.Tensor, shape: Shape) -> torch.Tensor:
    """(batch, frames, H, W) -> (batch, tokens, patch*patch), in the game's token order."""
    b, f, h, w = images.shape
    p = shape.patch
    x = images.reshape(b, f, h // p, p, w // p, p)          # b f rows dy cols dx
    x = x.permute(0, 1, 2, 4, 3, 5)                          # b f rows cols dy dx
    return x.reshape(b, f * (h // p) * (w // p), p * p)


class Block(nn.Module):
    def __init__(self, s: Shape):
        super().__init__()
        d = s.model
        self.heads = s.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, s.feed_forward)
        self.ff2 = nn.Linear(s.feed_forward, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=-1)
        split = lambda z: z.reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(split(q), split(k), split(v))
        x = x + self.out(a.transpose(1, 2).reshape(b, t, d))
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))


class TransformerPolicy(nn.Module):
    def __init__(self, s: Shape):
        super().__init__()
        self.shape = s
        self.patch = nn.Linear(s.patch * s.patch, s.model)
        self.position = nn.Parameter(0.02 * torch.randn(s.tokens, s.model))
        self.layers = nn.ModuleList(Block(s) for _ in range(s.layers))
        self.norm = nn.LayerNorm(s.model)
        self.head = nn.Linear(s.model, s.outputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (batch, frames, H, W) grey values in [0, 1], oldest frame first."""
        x = self.patch(patchify(images, self.shape)) + self.position
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x).mean(dim=1))


class MlpPolicy(nn.Module):
    """The 2016 network (one frame -> sigmoid hidden -> 4), trained here on the same data."""

    def __init__(self, inputs: int = 400, hidden: int = 32, outputs: int = 4):
        super().__init__()
        self.l1 = nn.Linear(inputs, hidden)
        self.l2 = nn.Linear(hidden, outputs)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.sigmoid(self.l1(images[:, -1].flatten(1))))


def _write(out, name: str, t: torch.Tensor) -> None:
    flat = t.detach().reshape(-1).tolist()
    out.write(f"{name} {len(flat)}\n")
    out.write(" ".join(f"{v:.9g}" for v in flat) + "\n")


def export_transformer(model: TransformerPolicy, path: str) -> None:
    s = model.shape
    with open(path, "w") as out:
        out.write(f"transformer 1 {s.frames} {s.height} {s.width} {s.patch} {s.model} {s.heads} "
                  f"{s.layers} {s.feed_forward} {s.outputs}\n")
        _write(out, "patch.weight", model.patch.weight)
        _write(out, "patch.bias", model.patch.bias)
        _write(out, "position", model.position)
        for i, L in enumerate(model.layers):
            for part in ("ln1", "qkv", "out", "ln2", "ff1", "ff2"):
                m = getattr(L, part)
                _write(out, f"layers.{i}.{part}.weight", m.weight)
                _write(out, f"layers.{i}.{part}.bias", m.bias)
        _write(out, "norm.weight", model.norm.weight)
        _write(out, "norm.bias", model.norm.bias)
        _write(out, "head.weight", model.head.weight)
        _write(out, "head.bias", model.head.bias)


def export_mlp(model: MlpPolicy, path: str) -> None:
    """Writes the game's existing "mlp 1" policy format (src/ai/Mlp.cpp)."""
    w1, b1, w2, b2 = (t.detach().reshape(-1).tolist() for t in
                      (model.l1.weight, model.l1.bias, model.l2.weight, model.l2.bias))
    with open(path, "w") as out:
        out.write(f"mlp 1 {model.l1.in_features} {model.l1.out_features} {model.l2.out_features}\n")
        for a in (w1, b1, w2, b2):
            out.write(" ".join(f"{v:.9g}" for v in a) + "\n")


def shape_dict(s: Shape) -> dict:
    return asdict(s)
