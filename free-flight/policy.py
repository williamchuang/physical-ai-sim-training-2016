"""The flight policy: observation -> thrust in the strip's local frame."""

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
