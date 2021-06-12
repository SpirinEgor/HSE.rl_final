from obstacle_avoidance.agents import ObstacleAvoidancePredatorAgent, ObstacleAvoidancePreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv
from solution.agents import SolutionPreyAgent


def main():
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ObstacleAvoidancePredatorAgent(env.config["game"]["num_preds"])
    prey_agent = SolutionPreyAgent()

    step_count = 0
    state_dict = env.reset()
    while True:
        state_dict, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
        step_count += 1

        if done:
            state_dict = env.reset()


if __name__ == "__main__":
    main()
