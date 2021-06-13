from os.path import dirname, join
from typing import Dict, List

import torch

from predators_and_preys_env.agent import PredatorAgent, PreyAgent
from solution.actor_critic import Actor
from utils import state_dict_to_array


class SolutionPredatorAgent(PredatorAgent):
    def act(self, state_dict: Dict) -> List[float]:
        pass


class SolutionPreyAgent(PreyAgent):
    _device = torch.device("cpu")

    def __init__(self, path="agent_last.pkl"):
        base_dir = dirname(__file__)
        state_dict = torch.load(join(base_dir, path), map_location=self._device)
        self._actor = Actor(**state_dict["init_params"])
        self._actor.load_state_dict(state_dict["state_dict"])

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for prey in state_dict["preys"]:
            with torch.no_grad():
                state = torch.tensor(state_dict_to_array(state_dict["predators"], [prey], state_dict["obstacles"]))
                action.append(self._actor(state))
        return action
