import random
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from solution.td3 import TD3
from config import Config
from obstacle_avoidance.agents import ObstacleAvoidancePredatorAgent, ObstacleAvoidancePreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv
from utils import calculate_dead, state_dict_to_array


class Trainer:
    def __init__(self, config: Config):
        self._config = config

    def _seed_everything(self, env: PredatorsAndPreysEnv):
        env.game.seed(self._config.seed)
        random.seed(self._config.seed)
        torch.manual_seed(self._config.seed)
        np.random.seed(self._config.seed)

    def evaluate_policy(self, prey_agent: TD3):
        env = PredatorsAndPreysEnv(render=True)
        env.game.seed(self._config.seed)
        predator_agent = ObstacleAvoidancePredatorAgent(env.config["game"]["num_preds"])
        returns = []
        for _ in range(self._config.evaluate_episodes):
            done = False
            state_dict = env.reset()

            while not done:
                state = state_dict_to_array(state_dict)
                state_dict, done = env.step(predator_agent.act(state_dict), [prey_agent.act(state)])
            returns.append(state_dict["preys"][0]["is_alive"])
        return returns

    def train(self):
        str_config = "\n".join(f"{key}: {value}" for key, value in asdict(self._config).items())
        print(f"Train config:\n{str_config}")

        env = PredatorsAndPreysEnv()
        self._seed_everything(env)
        n_obstacles = env.config["game"]["num_obsts"]
        n_predators = env.config["game"]["num_preds"]
        n_preys = env.config["game"]["num_preys"]
        state_dim = (n_obstacles + n_predators + n_preys) * 2 + n_preys

        predator_pfm = ObstacleAvoidancePredatorAgent(n_predators)
        prey_pfm = ObstacleAvoidancePreyAgent(n_preys)
        prey_td3 = TD3(state_dim=state_dim, action_dim=2, config=self._config)

        state_dict = env.reset()
        print("Do initial steps")
        for _ in tqdm(range(self._config.initial_steps), total=self._config.initial_steps):
            action = prey_pfm.act(state_dict)
            next_state_dict, done = env.step(predator_pfm.act(state_dict), action)
            reward = calculate_dead(state_dict) * -10

            state = state_dict_to_array(state_dict)
            next_state = state_dict_to_array(next_state_dict)
            prey_td3.consume_transition(state, action, next_state, reward, done)

            state_dict = next_state_dict if not done else env.reset()

        print(f"Start training")
        best_score = None
        state_dict = env.reset()
        for i in tqdm(range(self._config.transitions), total=self._config.transitions):
            # Epsilon-greedy policy
            state = state_dict_to_array(state_dict)
            action = prey_td3.act(state)
            action = np.clip(action + self._config.eps * np.random.randn(*action.shape), -1, 1)

            next_state_dict, done = env.step(predator_pfm.act(state_dict), [action])
            next_state = state_dict_to_array(next_state_dict)
            reward = calculate_dead(state_dict) * -10

            prey_td3.update(state, action, next_state, reward, done)

            state_dict = next_state_dict if not done else env.reset()

            if (i + 1) % self._config.evaluate_step == 0:
                rewards = self.evaluate_policy(prey_td3)
                mean_reward = np.mean(rewards)
                print(f"Step: {i + 1}, Reward mean: {mean_reward}, Reward std: {np.std(rewards)}")
                if best_score is None or mean_reward > best_score:
                    prey_td3.save("agent_best.pkl")
                    best_score = mean_reward
                prey_td3.save("agent_last.pkl")


if __name__ == "__main__":
    _trainer = Trainer(Config())
    _trainer.train()
