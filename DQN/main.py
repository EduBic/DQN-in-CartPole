
import matplotlib.pyplot as plt
import gym
import numpy as np

import os
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


def init_CartPole(double_enable):
    CartPoleProb = "CartPole-v0"
    env = Environment(CartPoleProb, normalize=False, render=False)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.n

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, 
                double_q_learning=double_enable, 
                max_eps=1,
                min_eps=0.01, 
                update_target_frequency=800,
                memory_capacity=10000,
                mem_batch_size=64,
                mLambda=0.001,
                gamma=0.99)
    agent.memory = get_rand_agent_memory(env, actionsCount, agent.memory_capacity)

    return agent, env

def main():

    # Settings
    seed = 42
    double_enable = True

    method = "No-method"
    if double_enable: 
        method = "DDQN"
    else:
        method = "DQN"

    prefix = method + "-" + str(seed)

    random.seed(seed)
    np.random.seed(seed)
    agent, env = init_CartPole(double_enable)
    env.set_seed(seed)

    print("Seed set:", seed)
    print("\nStart")

    # initialize the csv 
    folder = 'results/'

    if not os.path.exists(folder): 
        os.makedirs(folder)

    nameResult = prefix + '-' + dt.datetime.now().strftime("%m-%dT%H-%M")
    fileNetPath = folder + nameResult + '.h5'
    fileCsvPath = folder + nameResult + '.csv'
    fileCsvPath_epoch = folder + nameResult + '-epoch.csv'

    with open(fileCsvPath, 'w', newline='') as csvfile, open(fileCsvPath_epoch, 'w', newline='') as csvFile_epoch:
        fieldnames = ['episode', 'reward', 'q-online-value', 'q-target-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        fieldnames_ep = ['epoch', 'q-online-value', 'q-target-value', 'epsilon', 'loss_mean']
        writer_ep = csv.DictWriter(csvFile_epoch, fieldnames=fieldnames_ep)
        writer_ep.writeheader()

        agent.set_writer_epochs(writer_ep)

        try:
            start = timer()

            for episode in range(5000):

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
            #agent.brain.model.save(fileNetPath)
            
            end = timer()
            elapsed_seconds = end - start
            
            #csvfile.write(str(elapsed_seconds))
            print("Total time (s)", str(elapsed_seconds))
            agent.set_writer_epochs(None)
    
    print("End\n")

if __name__ == "__main__":
    main()