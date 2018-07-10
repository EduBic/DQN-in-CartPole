
# Plotters
import os
import numpy as np
import matplotlib.pyplot as plt

'''
*     0         1                   2               3               4
* episode,  reward,           q-online-value,   q-target-value
*
* epoch,    q-online-value,   q-target-value,   epsilon,        loss_mean
'''

def save_fig(folderFile, namePlot):
    #if not os.path.exists(folderFile + "/" + namePlot):
        #os.makedirs(folderFile + "/" + namePlot)

    plt.savefig(folderFile + "/" + namePlot + ".png")


def plot_file(name_file, folder_data, folder_plot):

    csv_file = np.genfromtxt(folder_data + name_file, delimiter=',')

    if "epoch" in name_file:
        # y-axe
        q_online_value = csv_file[:, 1]
        q_target_value = csv_file[:, 2]
        epsilon = csv_file[:, 3]
        loss = csv_file[:, 4]
        # x-axe
        epoch = range(0, len(q_online_value))

        plot_(epoch,  q_online_value,   "Q value per epoch: " + name_file, "Epochs", "Q-value estimates")
        save_fig(folder_plot, name_file + "-q-value-per-epoch")

        plot_(epoch,  loss, "Loss: " + name_file, "Epochs", "Loss")
        save_fig(folder_plot, name_file + "-loss")

    else:
        # y-axe
        rewards = csv_file[:, 1]
        q_online_value = csv_file[:, 2]
        q_target_value = csv_file[:, 3]
        # x-axe
        episode = range(0, len(q_online_value))

        plot_(episode, q_online_value, "Q value per episode: " + name_file, "Episodes", "Q-value estimates")
        save_fig(folder_plot, name_file + "-q-value-per-ep")

        plot_(episode, rewards, "Rewards: " + name_file, "Episodes", "Rewards")
        save_fig(folder_plot, name_file + "-rewards")


def plot_(xs, ys, name_png, xlabel, ylabel):
    plt.clf()

    plt.plot(xs, ys, linewidth=0.4)

    plt.title(name_png)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend(loc=1, ncol=2, borderaxespad=0.8)