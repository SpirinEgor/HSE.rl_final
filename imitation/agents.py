from os.path import dirname, join
from typing import Dict, List

import torch

from imitation.model import ImitationModel
from predators_and_preys_env.agent import PredatorAgent, PreyAgent
from utils import state_dict_to_array


class ImitationPredatorAgent(PredatorAgent):
    _device = torch.device("cpu")

    def __init__(self, path="../predators.pkl"):
        super().__init__()
        base_dir = dirname(__file__)
        state_dict = torch.load(join(base_dir, path), map_location=self._device)
        self._model = ImitationModel(**state_dict["parameters"])
        self._model.load_state_dict(state_dict["state_dict"])

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for predator in state_dict["predators"]:
            with torch.no_grad():
                state = torch.tensor(
                    state_dict_to_array([predator], state_dict["preys"], state_dict["obstacles"]),
                    dtype=torch.float32,
                    device=self._device,
                )
                action.append(self._model(state).item())

        return action


class ImitationPreyAgent(PreyAgent):
    _device = torch.device("cpu")

    def __init__(self, path="../preys.pkl"):
        base_dir = dirname(__file__)
        state_dict = torch.load(join(base_dir, path), map_location=self._device)
        self._model = ImitationModel(**state_dict["parameters"])
        self._model.load_state_dict(state_dict["state_dict"])

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for prey in state_dict["preys"]:
            with torch.no_grad():
                state = torch.tensor(
                    state_dict_to_array(state_dict["predators"], [prey], state_dict["obstacles"]),
                    dtype=torch.float32,
                    device=self._device,
                )
                action.append(self._model(state).item())

        return action
