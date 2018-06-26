
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

def get_rand_agent_memory(env, actionsCount, memory_capacity):
    randAgent = RandomAgent(actionsCount, memory_capacity)
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

    agent = Agent(stateDataCount, actionsCount, double_q_learning=True, min_eps=0.01, mLambda=0.001)
    agent.memory = get_rand_agent_memory(env, actionsCount, agent.memory_capacity)

    return agent, env

def main():

    seed = 42
    prefix = "test-mse-DQN-seed-" + str(seed)

    random.seed(seed)
    np.random.seed(seed)
    agent, env = init_CartPole()
    env.set_seed(seed)

    print("\nStart")

    # initialize the csv 
    folder = 'results/'
    nameResult = prefix + '-' + env.name + '-' + dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fileNetPath = folder + nameResult + '.h5'
    fileCsvPath = folder + nameResult + '.csv'
    fileCsvPath_epoch = folder + nameResult + '-epoch.csv'

    with open(fileCsvPath, 'w', newline='') as csvfile, open(fileCsvPath_epoch, 'w', newline='') as csvFile_epoch:
        fieldnames = ['episode', 'reward', 'q-online-value', 'q-target-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        fieldnames_ep = ['epoch', 'q-online-value', 'q-target-value', 'epsilon']
        writer_ep = csv.DictWriter(csvFile_epoch, fieldnames=fieldnames_ep)
        writer_ep.writeheader()

        agent.set_writer_epochs(writer_ep)

        try:
            start = timer()

            for episode in range(3500):

                reward_result = env.run(agent)
                #print("Tot. reward", reward_result)

                q_online_results = agent.get_and_reinit_q_online_results()
                q_target_results = agent.get_and_reinit_q_target_results()

                writer.writerow({
                    fieldnames[0]: episode + 1,
                    fieldnames[1]: reward_result,
                    fieldnames[2]: np.mean(q_online_results),
                    fieldnames[3]: np.mean(q_target_results)
                })

        finally:
            agent.brain.model.save(fileNetPath)
            
            end = timer()
            elapsed_seconds = end - start
            
            csvfile.write(str(elapsed_seconds))
            agent.set_writer_epochs(None)
    
    print("End\n")

if __name__ == "__main__":
    main()