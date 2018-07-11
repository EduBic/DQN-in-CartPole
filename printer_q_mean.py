import matplotlib.pyplot as plt
from numpy import genfromtxt
import numpy as np
import statistics


indeces_epoch   = [1, 2]

RESULTS_DIR = 'DQN/results/'

TITLE = "DQN Shallow"


def plot_q_values(q_value_mean, q_value_median, q_value_min, q_value_max):
    plt.clf()

    namePlot = "q_values_median"

    steps = range(0, len(q_value_mean))

    plt.fill_between(steps, q_value_min, q_value_max,  facecolor='red', alpha=0.5)
    #plt.plot(steps, q_value_mean, label="", linewidth=0.4, color='#1f77b4')
    plt.plot(steps, q_value_median, label="", linewidth=0.6, color='#1f77b4')

    plt.plot([135000, -200], [100, 100], 'g', linewidth=0.6)

    axes = plt.gca()
    axes.set_xlim([-10, 1050])
    axes.set_ylim([-10, 350])

    #axes.get_xaxis().set_visible(False)
    #axes.get_yaxis().set_visible(False)


    plt.title(TITLE, fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel('Q-value estimates', fontsize=14)
    plt.show()



def main():

    files = [

        'res_experiment_0/DQN-32-tau1000-07-10T19-53-epoch-restricted-32',
        'res_experiment_1/DQN-42-tau1000-07-10T21-47-epoch-restricted-32',
        'res_experiment_2/DQN-52-tau1000-07-10T22-18-epoch-restricted-32'

        # 'res_experiment_3/DQN-deep-32-tau1000-07-11T10-04-epoch-restricted-32',
        # 'res_experiment_4/DQN-deep-42-tau1000-07-11T11-32-epoch-restricted-32',
        # 'res_experiment_5/DQN-deep-52-tau1000-07-11T12-01-epoch-restricted-32'

        # 'res_experiment_6/DoubleDQN-32-tau1000-07-11T12-46-epoch-restricted-32',
        # 'res_experiment_7/DoubleDQN-42-tau1000-07-11T13-17-epoch-restricted-32',
        # 'res_experiment_8/DoubleDQN-52-tau1000-07-11T13-51-epoch-restricted-32'

        # 'res_experiment_9/DoubleDQN-deep-32-tau1000-07-11T14-25-epoch-restricted-32',
        # 'res_experiment_10/DoubleDQN-deep-42-tau1000-07-11T14-59-epoch-restricted-32',
        # 'res_experiment_11/DoubleDQN-deep-52-tau1000-07-11T15-47-epoch-restricted-32'
    ]

    q_value_online = []

    for f in files:
        csv_file = genfromtxt(RESULTS_DIR + f + '.csv', delimiter=',')
        q_value_online.append(csv_file[1:, indeces_epoch[0]])

    min_len = 9999999000

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
