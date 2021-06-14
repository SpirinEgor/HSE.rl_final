import random
from typing import Dict, List, Optional

import numpy as np
import torch


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


def state_dict_to_array(predators: List[Dict], preys: List[Dict], obstacles: List[Dict]) -> List:
    result = []
    for pred in predators:
        result += [pred["x_pos"], pred["y_pos"]]
    for prey in preys:
        result += [prey["x_pos"], prey["y_pos"], float(prey["is_alive"])]
    for obstacle in obstacles:
        result += [obstacle["x_pos"], obstacle["y_pos"], obstacle["radius"]]
    return result


def run_until_done(env, predator_agent, prey_agent) -> Dict:
    state_dict = env.reset()
    while True:
        state_dict, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
        if done:
            return state_dict


def seed_everything(env, seed: int):
    if env is not None:
        env.game.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
