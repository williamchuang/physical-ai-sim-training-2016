"""Flight policies: observation -> thrust in the strip's local frame.

FlightPolicy is the MLP. make_policy("transformer") returns the transformer from
transformer/free-flight/flight_transformer.py, which attends over the last 8 observations."""

import sys
from pathlib import Path

import torch
from torch import nn

from world import OBSERVATION_SIZE


class FlightPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3),
            nn.Tanh(),  # thrust components in [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def reset(self) -> None:  # no memory
        pass

    def detach(self) -> None:
        pass


def make_policy(kind: str = "mlp", memory: int = 8) -> nn.Module:
    if kind == "transformer":
        # The transformer lives with the other transformer work, in transformer/free-flight.
        folder = str(Path(__file__).resolve().parents[1] / "transformer" / "free-flight")
        if folder not in sys.path:
            sys.path.insert(0, folder)
        from flight_transformer import FlightTransformer

        return FlightTransformer(OBSERVATION_SIZE, history=memory)
    return FlightPolicy()
