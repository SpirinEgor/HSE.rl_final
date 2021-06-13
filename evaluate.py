from imitation.agents import ImitationPredatorAgent, ImitationPreyAgent
from predators_and_preys_env.env import PredatorsAndPreysEnv
from utils import run_until_done, calculate_dead


def evaluate():
    env = PredatorsAndPreysEnv(render=True)
    predator_agent = ImitationPredatorAgent()
    prey_agent = ImitationPreyAgent()

    while True:
        state_dict = run_until_done(env, predator_agent, prey_agent)
        print(f"Done simulation, killed: {calculate_dead(state_dict)}")


if __name__ == "__main__":
    evaluate()
