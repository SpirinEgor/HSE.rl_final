# Эти классы (неявно) имплементируют необходимые классы агентов
# Если Вам здесь нужны какие-то локальные импорты, то их необходимо относительно текущего пакета
# Пример: `import .utils`, где файл `utils.py` лежит рядом с `submission.py`
from typing import Dict, List

from .obstacle_avoidance.agents import ObstacleAvoidancePredatorAgent, ObstacleAvoidancePreyAgent


class PredatorAgent:
    def __init__(self, n_predators: int = 2):
        self._predator_agent = ObstacleAvoidancePredatorAgent(n_predators)

    def act(self, state_dict: Dict) -> List[float]:
        return self._predator_agent.act(state_dict)


class PreyAgent:
    def __init__(self, n_preys: int = 5):
        self._prey_agent = ObstacleAvoidancePreyAgent(n_preys)

    def act(self, state_dict: Dict) -> List[float]:
        return self._prey_agent.act(state_dict)
