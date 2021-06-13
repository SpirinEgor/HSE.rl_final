from obstacle_avoidance.agents import ObstacleAvoidancePredatorAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv
from solution.agents import SolutionPreyAgent
from utils import run_until_done, calculate_dead


def evaluate():
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ObstacleAvoidancePredatorAgent(env.config["game"]["num_preds"])
    prey_agent = SolutionPreyAgent()

    while True:
        state_dict = run_until_done(env, predator_agent, prey_agent)
        print(f"Done simulation, killed: {calculate_dead(state_dict)}")


if __name__ == "__main__":
    evaluate()
