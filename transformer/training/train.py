"""Trains the bot cube's policy in PyTorch from the game's recordings, then exports it.

    python transformer/training/train.py --kind transformer --out transformer/models/transformer.txt transformer/recordings/*.bin
    python transformer/training/train.py --kind mlp --out transformer/models/policy-same-data.txt transformer/recordings/*.bin

--kind mlp trains the 2016 network (one frame -> 32 sigmoid units -> 4) on the
same data, so the comparison isolates the architecture from the data.
--init continues from an earlier checkpoint (used between DAgger rounds).
The last tenth of every recording is held out for validation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import Recordings
from model import MlpPolicy, Shape, TransformerPolicy, export_mlp, export_transformer, shape_dict


def split(data: Recordings, files_rows: list[int]):
    train, val, start = [], [], 0
    for n in files_rows:
        cut = start + int(0.9 * n)
        train.append(torch.arange(start, cut))
        val.append(torch.arange(cut, start + n))
        start += n
    return torch.cat(train), torch.cat(val)


@torch.no_grad()
def accuracy(model, data: Recordings, rows: torch.Tensor) -> tuple[float, float]:
    model.eval()
    right, loss = 0, 0.0
    for chunk in rows.split(2048):
        x, y = data.batch(chunk)
        z = model(x)
        loss += F.cross_entropy(z, y, reduction="sum").item()
        right += (z.argmax(1) == y).sum().item()
    model.train()
    return right / len(rows), loss / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--kind", choices=["transformer", "mlp"], default="transformer")
    ap.add_argument("--out", required=True, help="weight file for the game")
    ap.add_argument("--init", help="checkpoint (.pt) to continue from")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=2016)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    shape = Shape()
    frames = shape.frames if args.kind == "transformer" else 1
    rows_per_file, parts = [], []
    for p in args.recordings:
        parts.append(Recordings([p], frames))
        rows_per_file.append(len(parts[-1]))
    data = Recordings(args.recordings, frames)
    shape.height, shape.width = data.height, data.width
    train_rows, val_rows = split(data, rows_per_file)
    counts = torch.bincount(data.labels, minlength=4).tolist()
    print(f"{len(train_rows)} training and {len(val_rows)} validation samples; actions {counts}")

    model = TransformerPolicy(shape) if args.kind == "transformer" else MlpPolicy(data.height * data.width)
    ckpt = Path(args.out).with_suffix(".pt")
    if args.init:
        model.load_state_dict(torch.load(args.init)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    params = sum(p.numel() for p in model.parameters())
    print(f"{args.kind}: {params} parameters")

    deadline = time.time() + 60 * args.minutes
    start, step, best = time.time(), 0, (-1.0, None)
    history = []
    while time.time() < deadline:
        perm = train_rows[torch.randperm(len(train_rows))]
        for chunk in perm.split(args.batch):
            frac = min(1.0, (time.time() - start) / (deadline - start))
            for g in opt.param_groups:  # cosine decay over the time budget
                g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * frac))
            x, y = data.batch(chunk)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if time.time() >= deadline:
                break
        acc, vloss = accuracy(model, data, val_rows)
        history.append({"step": step, "seconds": round(time.time() - start), "val_accuracy": acc, "val_loss": vloss})
        print(f"step {step:6d}  {time.time() - start:5.0f} s  validation accuracy {acc:.3f}  loss {vloss:.3f}",
              flush=True)
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()})

    model.load_state_dict(best[1])
    torch.save({"model": best[1], "kind": args.kind, "shape": shape_dict(shape)}, ckpt)
    (export_transformer if args.kind == "transformer" else export_mlp)(model, args.out)
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(
        {"kind": args.kind, "parameters": params, "recordings": [Path(p).name for p in args.recordings],
         "samples": len(data), "best_val_accuracy": best[0], "history": history}, indent=1))
    print(f"best validation accuracy {best[0]:.3f}; wrote {args.out} and {ckpt}")


if __name__ == "__main__":
    main()
