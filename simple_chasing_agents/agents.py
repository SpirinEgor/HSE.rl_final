import numpy as np

from predators_and_preys_env.agent import PredatorAgent, PreyAgent
from utils import get_closest


class ChasingPredatorAgent(PredatorAgent):
    def act(self, state_dict):
        action = []
        for predator in state_dict["predators"]:
            closest_prey = get_closest(predator, state_dict["preys"])
            if closest_prey is None:
                action.append(0.0)
            else:
                action.append(
                    np.arctan2(closest_prey["y_pos"] - predator["y_pos"], closest_prey["x_pos"] - predator["x_pos"])
                    / np.pi
                )
        return action


class FleeingPreyAgent(PreyAgent):
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = get_closest(prey, state_dict["predators"])
            if closest_predator is None:
                action.append(0.0)
            else:
                action.append(
                    1
                    + np.arctan2(closest_predator["y_pos"] - prey["y_pos"], closest_predator["x_pos"] - prey["x_pos"])
                    / np.pi
                )
        return action
