import numpy as np

from predators_and_preys_env.agent import PredatorAgent, PreyAgent
from utils import distance


class ChasingPredatorAgent(PredatorAgent):
    def act(self, state_dict):
        target = None
        for prey in state_dict["preys"]:
            if prey["is_alive"]:
                target = prey
                break
        if target is None:
            return [0 for _ in state_dict["predators"]]
        return [
            np.arctan2(target["y_pos"] - predator["y_pos"], target["x_pos"] - predator["x_pos"]) / np.pi
            for predator in state_dict["predators"]
        ]


class FleeingPreyAgent(PreyAgent):
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.0)
            else:
                action.append(
                    1
                    + np.arctan2(closest_predator["y_pos"] - prey["y_pos"], closest_predator["x_pos"] - prey["x_pos"])
                    / np.pi
                )
        return action
