"""Writes tests/data/transformer_fixture.txt: a random transformer, some inputs and
PyTorch's outputs for them. The C++ test loads the same weights and must match.

    python transformer/training/make_fixture.py
"""
from pathlib import Path

import torch

from model import Shape, TransformerPolicy, export_transformer

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    torch.manual_seed(2016)
    # Smaller than the game's network, same code paths (several frames, heads and layers).
    shape = Shape(frames=3, height=10, width=10, patch=5, model=16, heads=2, layers=2, feed_forward=24)
    model = TransformerPolicy(shape).double()
    with torch.no_grad():  # move every parameter away from its default so each one is tested
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    path = ROOT / "transformer" / "tests" / "transformer_fixture.txt"
    export_transformer(model, str(path))
    inputs = torch.rand(5, shape.frames, shape.height, shape.width, dtype=torch.float64)
    with torch.no_grad():
        logits = model(inputs)
    with open(path, "a") as out:
        out.write(f"cases {len(inputs)}\n")
        for x, z in zip(inputs, logits):
            out.write(" ".join(f"{v:.9g}" for v in x.reshape(-1).tolist()) + "\n")
            out.write(" ".join(f"{v:.9g}" for v in z.tolist()) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
