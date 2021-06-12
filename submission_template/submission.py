# Эти классы (неявно) имплементируют необходимые классы агентов
# Если Вам здесь нужны какие-то локальные импорты, то их необходимо относительно текущего пакета
# Пример: `import .utils`, где файл `utils.py` лежит рядом с `submission.py`
from typing import Dict, List, Optional

from .pfm import PFM


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


class PredatorAgent:
    _pfms = [PFM() for _ in range(2)]

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for predator, pfm in zip(state_dict["predators"], self._pfms):
            target = get_closest(predator, state_dict["preys"])
            if target is None:
                action.append(0)
                continue
            action.append(pfm.pfm_obstacle_avoidance(predator, target, state_dict["obstacles"]))
        return action


class PreyAgent:
    _pfms = [PFM() for _ in range(5)]

    def act(self, state_dict: Dict) -> List[float]:
        action = []
        for prey, pfm in zip(state_dict["preys"], self._pfms):
            target = get_closest(prey, state_dict["predators"])
            if target is None:
                action.append(0)
                continue
            action.append(1 + pfm.pfm_obstacle_avoidance(prey, target, state_dict["obstacles"]))

        return action
