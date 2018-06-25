
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

def write_q_values_epoch(fileCsvPath, mean_q_online_values, mean_q_target_values):

    with open(fileCsvPath, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'q-online-value', 'q-target-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(mean_q_online_values.size):
            writer.writerow({
                fieldnames[0]: epoch + 1,
                fieldnames[1]: mean_q_online_values.item(epoch),
                fieldnames[2]: mean_q_target_values.item(epoch)
            })



def main():

    seed = 42
    prefix = "mse-DQN-seed-" + str(seed)

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

    with open(fileCsvPath, 'w', newline='') as csvfile:
        fieldnames = ['episode', 'reward', 'q-online-value', 'q-target-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        try:
            start = timer()

            for episode in range(1000):

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

            mean_q_online, mean_q_target = agent.get_q_value_means_epoch()
            write_q_values_epoch(folder + nameResult + "-epochs.csv", mean_q_online, mean_q_target)
    
    print("End\n")

if __name__ == "__main__":
    main()