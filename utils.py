from typing import Dict, List, Optional

import numpy as np


def calculate_dead(state_dict: Dict) -> int:
    cnt = 0
    for p in state_dict["preys"]:
        if not p["is_alive"]:
            cnt += 1
    return cnt


def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5


def get_closest(source: Dict, targets: Dict) -> Optional[Dict]:
    closest = None
    for t in targets:
        if "is_alive" in t and not t["is_alive"]:
            continue
        if closest is None:
            closest = t
        elif distance(source, t) < distance(source, closest):
            closest = t
    return closest


def action_to_closest_prey(state_dict: Dict) -> List[int]:
    action = []
    for predator in state_dict["predators"]:
        closest_prey = get_closest(predator, state_dict["preys"])
        if closest_prey is None:
            action.append(0.0)
        else:
            action.append(
                np.arctan2(closest_prey["y_pos"] - predator["y_pos"], closest_prey["x_pos"] - predator["x_pos"]) / np.pi
            )
    return action


def state_dict_to_array(state_dict: Dict) -> np.ndarray:
    state_dim = len(state_dict["preys"]) + 2 * sum(len(state_dict[k]) for k in ["predators", "preys", "obstacles"])
    result = np.zeros(state_dim, dtype=np.float32)
    pos = 0
    for it in ["predators", "preys", "obstacles"]:
        for desc in state_dict[it]:
            result[pos] = desc["x_pos"]
            result[pos + 1] = desc["y_pos"]
            pos += 2
    for desc in state_dict["preys"]:
        result[pos] = desc["is_alive"]
        pos += 1
    return result
