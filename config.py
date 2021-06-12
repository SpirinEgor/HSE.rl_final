from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 7

    actor_lr: float = 2e-3
    actor_hidden_dim: int = 256
    actor_hidden_layers: int = 2

    critic_lr: float = 1e-4
    critic_hidden_dim: int = 256
    critic_hidden_layers: int = 2

    gamma: float = 0.99
    tau: float = 0.002
    eps: float = 0.2
    sigma: float = 2
    batch_size: int = 128
    transitions: int = 5_000_000
    buffer_size: int = 2_000_000
    initial_steps: int = batch_size * 256
    clip_value: float = 0.5

    evaluate_step: int = 50_000
    evaluate_episodes: int = 5
