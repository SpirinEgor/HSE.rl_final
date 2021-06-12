from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam

from solution.actor_critic import Actor, Critic
from config import Config


class TD3:
    def __init__(self, state_dim: int, action_dim: int, config: Config):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._config = config
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self._device}")

        self.actor = Actor(state_dim, action_dim, config).to(self._device)
        self.critics = [
            Critic(state_dim, action_dim, config).to(self._device),
            Critic(state_dim, action_dim, config).to(self._device),
        ]

        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizers = [Adam(critic.parameters(), lr=config.critic_lr) for critic in self.critics]

        self.target_actor = deepcopy(self.actor)
        self.target_critics = [deepcopy(critic) for critic in self.critics]

        self._consumed_transitions = 0
        self._buffer_pos = 0
        self._state_buffer = torch.empty((self._config.buffer_size, state_dim), dtype=torch.float, device=self._device)
        self._action_buffer = torch.empty(
            (self._config.buffer_size, action_dim), dtype=torch.float, device=self._device
        )
        self._next_state_buffer = torch.empty(
            (self._config.buffer_size, state_dim), dtype=torch.float, device=self._device
        )
        self._reward_buffer = torch.empty((self._config.buffer_size, 1), dtype=torch.float, device=self._device)
        self._done_buffer = torch.empty((self._config.buffer_size, 1), dtype=torch.float, device=self._device)

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self._config.tau) * tp.data + self._config.tau * sp.data)

    def consume_transition(self, state, action, next_state, reward, done):
        self._state_buffer[self._buffer_pos] = torch.tensor(state, device=self._device)
        self._action_buffer[self._buffer_pos] = torch.tensor(action, device=self._device)
        self._next_state_buffer[self._buffer_pos] = torch.tensor(next_state, device=self._device)
        self._reward_buffer[self._buffer_pos] = reward
        self._done_buffer[self._buffer_pos] = done

        self._buffer_pos = (self._buffer_pos + 1) % self._config.buffer_size
        self._consumed_transitions += 1

    def update(self, state, action, next_state, reward, done):
        self.consume_transition(state, action, next_state, reward, done)

        if self._consumed_transitions < 16 * self._config.batch_size:
            return

        # Sample batch
        max_idx = min(self._consumed_transitions, self._config.buffer_size)
        batch_idx = torch.randperm(max_idx)[: self._config.batch_size]
        b_state = self._state_buffer[batch_idx]
        b_action = self._action_buffer[batch_idx]
        b_next_state = self._next_state_buffer[batch_idx]
        b_reward = self._reward_buffer[batch_idx]
        b_done = self._done_buffer[batch_idx]

        with torch.no_grad():
            target_action = self.target_actor(b_next_state)
            noise = torch.clip(
                self._config.sigma * torch.randn_like(target_action), -self._config.clip_value, self._config.clip_value
            )
            target_action = torch.clip(target_action + noise, -1, 1)

            q_target = b_reward + self._config.gamma * (1 - b_done) * torch.minimum(
                self.target_critics[0](b_next_state, target_action), self.target_critics[1](b_next_state, target_action)
            )

        # Update critic
        losses = [F.mse_loss(critic(b_state, b_action), q_target) for critic in self.critics]

        for i in [0, 1]:
            self.critic_optimizers[i].zero_grad()
            losses[i].backward()
            self.critic_optimizers[i].step()

        # Update actor
        actor_loss = -self.critics[0](b_state, self.actor(b_state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update targets
        for critic, target_critic in zip(self.critics, self.target_critics):
            self.soft_update(target_critic, critic)
        self.soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], dtype=torch.float32, device=self._device)
            x, y = self.actor(state).squeeze(0).cpu().numpy()
            return np.arctan2(x, y) / np.pi

    def save(self, filename: str = "agent.pkl"):
        state = {
            "init_params": {"state_dim": self._state_dim, "action_dim": self._action_dim, "config": self._config},
            "state_dict": self.actor.state_dict(),
        }
        torch.save(state, filename)
