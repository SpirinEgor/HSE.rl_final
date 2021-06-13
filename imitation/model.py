import torch
from torch import nn


class ImitationModel(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 1024, n_layers: int = 5, dropout: float = 0.0
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dropout = dropout
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
        for _ in range(n_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, action_dim), nn.Tanh()]
        self._layers = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self._layers(state)

    def save(self, name: str):
        torch.save(
            {
                "parameters": {
                    "state_dim": self._state_dim,
                    "action_dim": self._action_dim,
                    "hidden_dim": self._hidden_dim,
                    "n_layers": self._n_layers,
                    "dropout": self._dropout,
                },
                "state_dict": self.state_dict(),
            },
            name,
        )
