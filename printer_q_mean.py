import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import statistics


indeces_epoch   = [1, 2]

RESULTS_DIR = 'DQN/results/'

TITLE = "Double DQN Deep"


def plot_q_values(q_value_mean, q_value_median, q_value_min, q_value_max):
    plt.clf()

    namePlot = "q_values_median"

    steps = range(0, len(q_value_mean))

    plt.fill_between(steps, q_value_min, q_value_max,  facecolor='red')
    #plt.plot(steps, q_value_mean, label="", linewidth=0.4, color='#1f77b4')
    plt.plot(steps, q_value_median, label="", linewidth=0.4, color='#1f77b4')


    plt.title(TITLE)
    plt.xlabel("epoch")
    plt.ylabel('Q-value')
    plt.show()



def main():

    files = [

        # 'res_experiment_5/DQN-32-07-04T19-36-epoch',
        # 'res_experiment_0/DQN-42-07-04T14-10-epoch',
        # 'res_experiment_1/DQN-52-07-04T14-28-epoch'

        # 'res_experiment_6/DQN-more_deep-32-07-04T19-34-epoch',
        # 'res_experiment_2/DQN-more_deep-42-07-04T14-40-epoch',
        # 'res_experiment_7/DQN-more_deep-52-07-04T19-53-epoch'

        # 'res_experiment_8/DQN-deep-32-07-04T20-31-epoch',
        # 'res_experiment_3/DQN-deep-42-07-04T15-01-epoch',
        # 'res_experiment_9/DQN-deep-52-07-04T20-30-epoch'


        # 'res_experiment_10/DoubleDQN-32-07-05T21-04-epoch',
        # 'res_experiment_11/DoubleDQN-42-07-04T20-56-epoch',
        # 'res_experiment_12/DoubleDQN-52-07-05T21-24-epoch'

        'res_experiment_15/DoubleDQN-deep-32-07-05T21-41-epoch',
        'res_experiment_16/DoubleDQN-deep-42-07-04T21-14-epoch',
        'res_experiment_17/DoubleDQN-deep-52-07-05T22-04-epoch'
    ]

    q_value_online = []

    for f in files:
        csv_file = genfromtxt(RESULTS_DIR + f + '.csv', delimiter=',')
        q_value_online.append(csv_file[1:, indeces_epoch[0]])

    min_len = 9999999

    for q in q_value_online:
        if min_len > len(q):
            min_len = len(q)

    print(str(min_len))

    q_value_median = [statistics.median(x) for x in zip(q_value_online[0], q_value_online[1], q_value_online[2])]

    c = [sum(x) for x in zip(q_value_online[0], q_value_online[1], q_value_online[2])]

    q_value_online_min = list(map(lambda pair: min(pair), zip(q_value_online[0], q_value_online[1], q_value_online[2])))
    q_value_online_max = list(map(lambda pair: max(pair), zip(q_value_online[0], q_value_online[1], q_value_online[2])))

    q_value_online_mean = [x / 3 for x in c]

    plot_q_values(q_value_online_mean[:min_len], q_value_median[:min_len], q_value_online_min[:min_len], q_value_online_max[:min_len])


if __name__ == "__main__":
    main()
