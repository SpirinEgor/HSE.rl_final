from examples.simple_chasing_agents.agents import ChasingPredatorAgent
from examples.simple_chasing_agents.agents import FleeingPreyAgent
from obstacle_avoidance.agents import ObstacleAvoidancePredatorAgent, ObstacleAvoidancePreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv

if __name__ == "__main__":
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ObstacleAvoidancePredatorAgent(2)
    prey_agent = ObstacleAvoidancePreyAgent(5)

    done = None
    step_count = 0
    state_dict = env.reset()
    while True:
        if done:
            state_dict = env.reset()
            done = False

        state_dict, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
        step_count += 1
