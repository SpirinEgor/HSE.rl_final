from typing import Dict, List

from obstacle_avoidance.pfm import PFM
from predators_and_preys_env.agent import PredatorAgent, PreyAgent
from utils import get_closest


class ObstacleAvoidancePredatorAgent(PredatorAgent):
    def __init__(self, n_predators):
        self._pfms = [PFM() for _ in range(n_predators)]

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for predator, pfm in zip(state_dict["predators"], self._pfms):
            target = get_closest(predator, state_dict["preys"])
            if target is None:
                action.append(0)
                continue
            action.append(pfm.pfm_obstacle_avoidance(predator, target, state_dict["obstacles"]))
        return action


class ObstacleAvoidancePreyAgent(PreyAgent):
    def __init__(self, n_preys):
        self._pfms = [PFM() for _ in range(n_preys)]

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for prey, pfm in zip(state_dict["preys"], self._pfms):
            target = get_closest(prey, state_dict["predators"])
            if target is None:
                action.append(0)
                continue
            action.append(1 + pfm.pfm_obstacle_avoidance(prey, target, state_dict["obstacles"]))

        return action
