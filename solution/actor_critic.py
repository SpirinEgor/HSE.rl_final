import torch
from torch import nn

from config import Config


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        super().__init__()
        modules = [nn.Linear(state_dim, config.actor_hidden_dim), nn.LayerNorm(config.actor_hidden_dim), nn.ELU()]
        for _ in range(config.actor_hidden_layers):
            modules += [
                nn.Linear(config.actor_hidden_dim, config.actor_hidden_dim),
                nn.LayerNorm(config.actor_hidden_dim),
                nn.ELU()
            ]
        modules += [nn.Linear(config.actor_hidden_dim, action_dim)]
        self.model = nn.Sequential(*modules)

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        super().__init__()
        modules = [
            nn.Linear(state_dim + action_dim, config.critic_hidden_dim),
            nn.LayerNorm(config.critic_hidden_dim),
            nn.ELU()
        ]
        for _ in range(config.critic_hidden_layers):
            modules += [
                nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
                nn.LayerNorm(config.critic_hidden_dim),
                nn.ELU()
            ]
        modules += [nn.Linear(config.critic_hidden_dim, 1)]
        self.model = nn.Sequential(*modules)

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))
