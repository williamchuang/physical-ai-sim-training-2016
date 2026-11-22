"""The free-flight transformer policy: attention over a bot's last observations -> thrust.

Used by free-flight/ through make_policy("transformer") in free-flight/policy.py;
trained there by backpropagation through the differentiable simulator:
    cd free-flight && python train.py --kind transformer --episodes 100 \
        --out ../transformer/free-flight/models/flight_transformer.pt
"""

import torch
from torch import nn


class FlightTransformer(nn.Module):
    """Attention over the bot's last `history` observations -> thrust.

    Each observation is one token (linear map to `model` numbers plus a learned
    position). Two pre-norm layers of multi-head self-attention and a GELU MLP
    follow; the newest token's output gives the thrust. The policy keeps its own
    short memory: call reset() when a new world starts and detach() wherever the
    simulator is detached for truncated backpropagation.
    """

    def __init__(self, observation_size: int, history: int = 8, model: int = 32, heads: int = 2, layers: int = 2, feed_forward: int = 64):
        super().__init__()
        self.history = history
        self.inp = nn.Linear(observation_size, model)
        self.position = nn.Parameter(0.02 * torch.randn(history, model))
        layer = nn.TransformerEncoderLayer(model, heads, feed_forward, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(model)
        self.out = nn.Linear(model, 3)
        self._memory = None

    def reset(self) -> None:
        self._memory = None

    def detach(self) -> None:
        if self._memory is not None:
            self._memory = self._memory.detach()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lead = obs.shape[:-1]
        x = obs.reshape(-1, 1, obs.shape[-1])
        if self._memory is None or self._memory.shape[0] != x.shape[0]:
            memory = x.expand(-1, self.history, -1)  # a new world starts with its first view repeated
        else:
            memory = torch.cat((self._memory[:, 1:], x), dim=1)
        self._memory = memory
        h = self.layers(self.inp(memory) + self.position)
        return torch.tanh(self.out(self.norm(h[:, -1]))).reshape(*lead, 3)
