
import matplotlib.pyplot as plt
import gym
import numpy as np

import os
from os import listdir
from os.path import isfile, join
import csv
import datetime as dt
import random
import re

from environment import Environment
from agent import Agent
from randomAgent import RandomAgent
from keras.models import load_model

from timeit import default_timer as timer
import winsound
duration = 420  # millisecond
freq = 505  # Hz

start = 0
end = 0

EPISODES = 100

# exp = "mod_31b"

models_dir = '../DQN/models/'
save_sessions_folder = 'sessions/'

RENDER = False


def get_rand_agent_memory(env, actionsCount, memory_capacity):
    randAgent = RandomAgent(actionsCount, memory_capacity)
    while randAgent.memory.is_full():
        env.run(randAgent)

    return randAgent.memory


def init_CartPole(model):
    CartPoleProb = "CartPole-v0"
    env = Environment(CartPoleProb, normalize=False, render=RENDER)

    stateDataCount = env.env.observation_space.shape[0]
    actionsCount = env.env.action_space.n

    print("\nState Data Count:", stateDataCount)
    print("Action Count:", actionsCount)

    agent = Agent(stateDataCount, actionsCount, model)

    return agent, env


def main():
    # Settings
    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(save_sessions_folder):
        os.makedirs(save_sessions_folder)

    folders_name = [f for f in listdir(models_dir) if os.path.isdir(models_dir + f)]
    folders_path = [models_dir + f for f in listdir(models_dir) if os.path.isdir(models_dir + f)]

    for i in range(1):

        model_names = [f for f in listdir(folders_path[i]) if isfile(join(folders_path[i], f))]

        dir_run = save_sessions_folder + folders_name[i] + '/'

        os.makedirs(dir_run)

        for m in model_names:

            steps = re.findall('_(.*).h5', m)

            print(steps)

            step_trained = steps[0]

            model = load_model(models_dir + folders_name[i] + '/' + m)
            agent, env = init_CartPole(model)
            env.set_seed(seed)

            print("Seed set:", seed)
            print("\nStart")

            fileCsvPath_gamer = dir_run + '/' + '_e' + step_trained + '.csv'

            with open(fileCsvPath_gamer, 'w', newline='') as csvfile:
                fieldnames = ['episode', 'reward']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                try:
                    start = timer()

                    for episode in range(EPISODES):

                        reward_result = env.run(agent)

                        # print("reward: ", reward_result)

                        writer.writerow({
                            fieldnames[0]: episode + 1,
                            fieldnames[1]: reward_result,
                        })

                finally:
                    end = timer()
                    elapsed_seconds = end - start
                    # csvfile.write(str(elapsed_seconds))
                    print("Total time (s)", str(elapsed_seconds))

        print("End\n")
        winsound.Beep(freq, duration)

if __name__ == "__main__":
    main()