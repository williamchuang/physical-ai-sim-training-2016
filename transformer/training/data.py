"""Reads the game's recordings (config key record_file).

File layout, little-endian:
    header : int32 magic "REC1", int32 width, int32 height
    record : int32 episode, int32 slot, int32 label (-1: no key), uint8 policy_drove,
             uint8 is_bot, then width*height uint8 grey pixels (0..255)

Records of one (episode, slot) pair are one cube's camera, in time order.
The game pads the start of a round by repeating the first frame; so does this reader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MAGIC = 0x31434552


def read(path: str | Path):
    raw = Path(path).read_bytes()
    magic, width, height = np.frombuffer(raw[:12], dtype="<i4")
    if magic != MAGIC:
        raise ValueError(f"{path}: not a recording")
    pixels = int(width * height)
    dtype = np.dtype([("episode", "<i4"), ("slot", "<i4"), ("label", "<i4"),
                      ("policy", "u1"), ("bot", "u1"), ("image", "u1", (pixels,))])
    count = (len(raw) - 12) // dtype.itemsize  # a run stopped mid-write leaves a partial record
    rec = np.frombuffer(raw[12:12 + count * dtype.itemsize], dtype=dtype)
    return rec, int(width), int(height)


def histories(rec, width: int, height: int, frames: int):
    """Indices (n, frames) into rec giving each record's last `frames` frames, oldest first."""
    n = len(rec)
    stream = rec["episode"].astype(np.int64) * 1000 + rec["slot"]
    idx = np.zeros((n, frames), dtype=np.int64)
    last: dict[int, list[int]] = {}
    for i in range(n):
        h = last.setdefault(int(stream[i]), [])
        h.append(i)
        if len(h) > frames:
            h.pop(0)
        row = [h[0]] * (frames - len(h)) + h
        idx[i] = row
    return idx


class Recordings:
    """All labelled samples from several recording files, as tensors."""

    def __init__(self, paths, frames: int):
        images, idx_all, labels, offset = [], [], [], 0
        for p in paths:
            rec, w, h = read(p)
            idx = histories(rec, w, h, frames) + offset
            keep = rec["label"] >= 0
            images.append(rec["image"])
            idx_all.append(idx[keep])
            labels.append(rec["label"][keep])
            offset += len(rec)
            self.width, self.height = w, h
        self.images = torch.from_numpy(np.concatenate(images)).reshape(-1, self.height, self.width)
        self.index = torch.from_numpy(np.concatenate(idx_all))
        self.labels = torch.from_numpy(np.concatenate(labels).astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, rows: torch.Tensor):
        """(images (b, frames, H, W) float in [0, 1], labels (b,))."""
        x = self.images[self.index[rows]].float() / 255.0
        return x, self.labels[rows]
