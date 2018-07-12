
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
import winsound

duration = 720  # millisecond
freq = 505  # Hz
BEEP = True


EPISODES = 3600
CHECKPOINT_STEP = 100
SAVE_CHECK = True
MAX_STEPS = 1024000

EXPERIMENT = '11'
EXPERIMENT_FOLDER = "res_experiment_" + EXPERIMENT + "/"

# DQN settings
SEED = 52
TAU = 1000
DOUBLE_SET = True

# Architecture DQN settings
DEEP_SET = True


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

    agent = Agent(stateDataCount, actionsCount, 
                diff_target_network=True, 
                double_DQN=DOUBLE_SET,
                max_eps=1,
                min_eps=0.01, 
                update_target_frequency=TAU,
                memory_capacity=10000,
                mem_batch_size=32,
                mLambda=0.001,
                gamma=0.99,
                deep_set=DEEP_SET)
    agent.memory = get_rand_agent_memory(env, actionsCount, agent.memory_capacity)

    return agent, env

def main():

    if DOUBLE_SET: 
        method = "DoubleDQN"    # True DDQN
    else:
        method = "DQN"

    if DEEP_SET:
        architecture = "-deep-"
    else:
        architecture = "-"

    prefix = method + architecture + str(SEED) + "-tau" + str(TAU)

    random.seed(SEED)
    np.random.seed(SEED)
    agent, env = init_CartPole()
    env.set_seed(SEED)

    print("Seed set:", SEED)
    print("\nStart")

    # initialize the csv 
    folder = 'results/' + EXPERIMENT_FOLDER
    models_folder = 'models/' + 'mod_' + EXPERIMENT + '/'

    nameResult = prefix + '-' + dt.datetime.now().strftime("%m-%dT%H-%M")
    fileNetPath = folder + nameResult + '.h5'
    fileCsvPath = folder + nameResult + '.csv'
    fileCheckpointPath = models_folder + nameResult
    fileCsvPath_epoch = folder + nameResult + '-epoch.csv'

    if not os.path.exists(folder):
        os.makedirs(folder)

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

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

            step_counter = 0
            episode = 0

            while step_counter <= MAX_STEPS:
            #for episode in range(EPISODES+1):

                reward_result = env.run(agent)
                step_counter += reward_result
                # print("steps: ", step_counter)

                # print("Tot. reward", reward_result)

                q_online_results = agent.get_and_reinit_q_online_results()
                q_target_results = agent.get_and_reinit_q_target_results()

                writer.writerow({
                    fieldnames[0]: episode + 1,
                    fieldnames[1]: reward_result,
                    fieldnames[2]: np.mean(q_online_results),
                    fieldnames[3]: np.mean(q_target_results)
                })

                if SAVE_CHECK and episode % CHECKPOINT_STEP == 0:
                    agent.brain.model.save(fileCheckpointPath + '_' + str(episode) + '.h5')

                episode += 1

        finally:
            #agent.brain.model.save(fileNetPath)
            
            end = timer()
            elapsed_seconds = end - start
            
            #csvfile.write(str(elapsed_seconds))
            print("Total time (s)", str(elapsed_seconds))
            agent.set_writer_epochs(None)
    
    print("End\n")
    if BEEP:
        winsound.Beep(freq, duration)

if __name__ == "__main__":
    main()