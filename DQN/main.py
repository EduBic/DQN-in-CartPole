
import matplotlib.pyplot as plt
import gym
import numpy as np


import csv
import datetime as dt
import random

from environment import Environment
from agent import Agent
from randomAgent import RandomAgent

from timeit import default_timer as timer

start = 0
end = 0

def get_rand_agent_memory(env, actionsCount):
    randAgent = RandomAgent(actionsCount)
    while randAgent.memory.is_full():
        env.run(randAgent)

    return randAgent.memory


def init_CartPole():
    CartPoleProb = "CartPole-v0"
    env = Environment(CartPoleProb, normalize=False, render=False)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.n

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, double_q_learning=False, min_eps=0.01)
    agent.memory = get_rand_agent_memory(env, actionsCount)

    return agent, env


#def run_alg(stateDataCount, actionsCount):


def init_MountainCar():
    MountainProb ="MountainCarContinuous-v0"
    env = Environment(MountainProb, normalize=True, render=False)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.shape[0]

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, min_eps=0.1)
    agent.memory = get_rand_agent_memory(env, actionsCount)

    return agent, env


def main():
    prefix = "test-seed"

    random.seed(42)
    np.random.seed(42)
    agent, env = init_CartPole()
    env.set_seed(42)

    print("\nStart")

    # initialize the csv 
    folder = 'results/'
    nameResult = prefix + '-' + env.name + '-' + dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fileNetPath = folder + nameResult + '.h5'
    fileCsvPath = folder + nameResult + '.csv'

    with open(fileCsvPath, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'q-online-value', 'q-target-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            start = timer()
            for episode in range(150):

                reward_result = env.run(agent)
                q_online_results = agent.get_and_reinit_q_online_results()
                q_target_results = agent.get_and_reinit_q_target_results()

                writer.writerow({
                    fieldnames[0]: episode + 1,
                    fieldnames[1]: reward_result,
                    fieldnames[2]: np.mean(q_online_results),
                    fieldnames[3]: np.mean(q_target_results)
                })

        finally:
            end = timer()
            pass
            #agent.brain.model.save(fileNetPath)
        elapsed_seconds = end - start
        csvfile.write(str(elapsed_seconds))
    print("End\n")

if __name__ == "__main__":
    main()