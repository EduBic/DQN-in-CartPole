import matplotlib.pyplot as plt
from numpy import genfromtxt
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import os

RESULTS_DIR = 'DQN/results/'

experiment = '07-02T16-26-44'
SESSIONS_DIR = 'Gamer/sessions/' + experiment + '/'

version = 0
PLOT_DIR = 'Plot/' + str(version) + "/"

SAVE_PLOT = True
SHOW = False

'''
*     0         1                   2               3               4
* episode,  reward,           q-online-value,   q-target-value
*
* epoch,    q-online-value,   q-target-value,   epsilon,        loss_mean
'''

xlabel = ""
indeces = []
indeces_episode = [2, 3]
indeces_epoch   = [1, 2]


def save_fig(plot, type):

    if SAVE_PLOT:
        if not os.path.exists(PLOT_DIR):
            os.makedirs(PLOT_DIR)

        plot.savefig(PLOT_DIR + type + ".png")

def plot_q_values(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(RESULTS_DIR + nameFileCsv + '.csv', delimiter=',')

        if "epoch" in nameFileCsv:
            indeces = indeces_epoch
            xlabel = "epoch"
        else:
            indeces = indeces_episode
            xlabel  = "episode"

        q_value_online = csv_file[:, indeces[0]]
        q_value_target = csv_file[:, indeces[1]]

        steps = range(0, len(q_value_online))

        plt.plot(steps, q_value_online, label=nameFileCsv[:14], linewidth=0.4)

    plt.title("Q-values")
    plt.xlabel(xlabel)
    plt.ylabel('Q-value')
    plt.legend(loc=1, ncol=2, borderaxespad=0.8)
    if SHOW: plt.show()

    save_fig(plt, "q_values_" + nameFileCsv)
    print(nameFileCsv + "saved")


def plot_rewards(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(RESULTS_DIR + nameFileCsv + '.csv', delimiter=',')
        rewards = csv_file[:, 1]

        steps = range(0, len(rewards))

        plt.plot(steps, rewards, label=nameFileCsv[:14], linewidth=0.4)

    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(loc=1, ncol=2, borderaxespad=0.8)
    if SHOW: plt.show()

    save_fig(plt, "rewards")


def plot_sessions(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(SESSIONS_DIR + nameFileCsv, delimiter=',')
        rewards = csv_file[:, 1]

        steps = range(0, len(rewards))

        episodes = re.findall('_e(.*).csv', nameFileCsv)

        plt.plot(steps, rewards, label=episodes[0], linewidth=0.4)

    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt, "session")


def plot_mean_sessions(files):
    plt.clf()

    r_means = []
    x = []
    std_array = []

    for nameFileCsv in files:
        csv_file = genfromtxt(SESSIONS_DIR + nameFileCsv, delimiter=',')
        rewards = csv_file[:, 1]

        episodes = re.findall('_e(.*).csv', nameFileCsv)

        r_means.append(rewards[1:].mean())

        x.append(int(re.search(r'\d+', episodes[0]).group()))

        std_array.append(rewards[1:].std())

    plt.xticks(x)
    plt.errorbar(x, r_means, std_array, linestyle='None', marker='o')

    plt.title('Reward Mean')
    plt.xlabel('Episodes of training')
    plt.ylabel('Reward Mean')
    # plt.legend()
    if SHOW: plt.show()

    save_fig(plt, "mean_session")


def plot_loss(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(RESULTS_DIR + nameFileCsv + '.csv', delimiter=',')
        loss = csv_file[:, 4]

        steps = range(0, len(loss))

        plt.plot(steps, loss, label=nameFileCsv[:14], linewidth=0.4)

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if SHOW: plt.show()

    save_fig(plt, "loss")


def main():

    files = [
        # DQN 1

        'DQN1-lambda-42-06-26T21-03',
        # 'DQN1-lambda-42-06-26T21-03-epoch',
        'DQN1-lambda-52-06-27T09-40',
        # 'DQN1-lambda-52-06-27T09-40-epoch',
        'DQN1-seed-42-06-26T16-56',
        # 'DQN1-seed-42-06-26T16-56-epoch',
        'DQN1-seed-52-2018-06-26T15-41-06',
        # 'DQN1-seed-52-2018-06-26T15-41-06-epoch',

        # DQN 2

        # 'DQN2-32-06-28T17-45-epoch',
        'DQN2-32-06-28T17-45',
        # 'DQN2-32-lambda-06-27T17-06-epoch',
        'DQN2-32-lambda-06-27T17-06',
        # 'DQN2-42-07-02T13-41-epoch',
        'DQN2-42-07-02T13-41',
        # 'DQN2-42-07-02T15-47-epoch',
        'DQN2-42-07-02T15-47',

        # 'DQN2-dd-32-06-29T14-11-epoch',
        'DQN2-dd-32-06-29T14-11',
        # 'DQN2-dd-42-06-29T19-52-epoch',
        'DQN2-dd-42-06-29T19-52',
        # 'DQN2-dd-42-07-02T11-39-epoch',
        'DQN2-dd-42-07-02T11-39',
        # 'DQN2-dd-42-07-02T12-46-epoch',
        'DQN2-dd-42-07-02T12-46',
        # 'DQN2-dd-52-06-29T10-48-epoch',
        'DQN2-dd-52-06-29T10-48',
        #
        # 'DQN2-deep-32-06-28T15-13-epoch',
        'DQN2-deep-32-06-28T15-13',
        # 'DQN2-deep-52-06-28T09-55-epoch',
        'DQN2-deep-52-06-28T09-55',
        #
        # 'DQN2-lambda-42-06-27T14-01-epoch',
        # 'DQN2-lambda-42-06-27T14-01',
        # 'DQN2-lambda-52-06-27T15-39-epoch',
        # 'DQN2-lambda-52-06-27T15-39',
        #
        # 'DQN2-seed-42-2018-06-26T10-48-11-epoch',
        # 'DQN2-seed-42-2018-06-26T10-48-11',
        # 'DQN2-seed-52-2018-06-26T13-56-28-epoch',
        # 'DQN2-seed-52-2018-06-26T13-56-28'
    ]

    OVERLAY = False

    if OVERLAY:
        plot_q_values(files)
    else:
        for f in files:
            plot_q_values([f])

    # game_sessions = [f for f in listdir(SESSIONS_DIR) if isfile(join(SESSIONS_DIR, f))]
    # game_sessions.sort(key=len)
    # plot_mean_sessions(game_sessions)


if __name__ == "__main__":
    main()
