import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import os

FOLDER = 'DQN/results/'
SESSIONS_FOLDER = 'Gamer/sessions/' + '07-02T16-26-44' + '/'
plot_dir = 'Plot/'

SAVE_PLOT = True
SHOW = False


def save_fig(plot, name):

    if SAVE_PLOT:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot.savefig(plot_dir + name + ".png")


def plot_q_values(files, indeces, xlabel):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(FOLDER + nameFileCsv + '.csv', delimiter=',')
        q_value_online = csv_file[:, indeces[0]]
        q_value_target = csv_file[:, indeces[1]]

        steps = range(0, len(q_value_online))

        plt.plot(steps, q_value_online, label=nameFileCsv[:14], linewidth=0.4)

    plt.title("Q-values")
    plt.xlabel(xlabel)
    plt.ylabel('Q-value')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt)


def plot_rewards(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(FOLDER + nameFileCsv + '.csv', delimiter=',')
        rewards = csv_file[:, 1]

        steps = range(0, len(rewards))

        plt.plot(steps, rewards, label=nameFileCsv[:14], linewidth=0.4)

    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt)


def plot_sessions(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(SESSIONS_FOLDER + nameFileCsv, delimiter=',')
        rewards = csv_file[:, 1]

        steps = range(0, len(rewards))

        episodes = re.findall('_e(.*).csv', nameFileCsv)

        plt.plot(steps, rewards, label=episodes[0], linewidth=0.4)

    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt)


def plot_mean_sessions(files):

    r_means = []
    x = []
    std_array = []

    for nameFileCsv in files:
        csv_file = genfromtxt(SESSIONS_FOLDER + nameFileCsv, delimiter=',')
        rewards = csv_file[:, 1]

        episodes = re.findall('_e(.*).csv', nameFileCsv)

        r_means.append(rewards[1:].mean())

        x.append(int(re.search(r'\d+', episodes[0]).group()))

        std_array.append(rewards[1:].std())

    plt.clf()

    plt.xticks(x)
    plt.errorbar(x, r_means, std_array, linestyle='None', marker='o')

    plt.title('Reward Mean')
    plt.xlabel('Episodes of training')
    plt.ylabel('Reward Mean')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt)


def plot_loss(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(FOLDER + nameFileCsv + '.csv', delimiter=',')
        loss = csv_file[:, 4]

        steps = range(0, len(loss))

        plt.plot(steps, loss, label=nameFileCsv[:14], linewidth=0.4)

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt)


def main():

    files = [
        # DQN
        #"DQN-seed-52-2018-06-26T15-41-06",
        #"DQN-seed-42-06-26T16-56",

        # Deep DDQN
        #"DDQN-deep-52-06-28T09-55",
        #"DDQN-deep-32-06-28T15-13",

        # More Deep DDQN
        #"DDQN-dd-52-06-29T10-48",
        #"DDQN-dd-32-06-29T14-11",


        # Double DQN
        # "DDQN-seed-52-2018-06-26T13-56-28",
        # "DDQN-seed-42-2018-06-26T10-48-11",

        # Lambda = 0.00001 -> Epsilon need more steps to decay to 0.01
        # "DQN-lambda-42-06-26T21-03",
        # "DDQN-32-lambda-06-27T17-06",
        # "DQN-lambda-52-06-27T09-40"

        'DDQN-42-07-02T15-47-epoch'
    ]

    game_sessions = [f for f in listdir(SESSIONS_FOLDER) if isfile(join(SESSIONS_FOLDER, f))]
    game_sessions.sort(key=len)

    # plot_mean_sessions(game_sessions)

    plot_loss(files)

    # step , reward, q-online, q-target
    # plot_rewards(files)
    # plot_q_values(files, indeces=[2, 3], xlabel='episode')

    # files_epochs = [csv_file + '-epoch' for csv_file in files]

    # epoch, q_online, q_target
    # plot_q_values(files_epochs, indeces=[1, 2], xlabel='epoch')
    

if __name__ == "__main__":
    main()
