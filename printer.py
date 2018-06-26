import matplotlib.pyplot as plt
from numpy import genfromtxt


FOLDER = 'DQN/results/'

def plot_rewards(files):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(FOLDER + nameFileCsv + '.csv', delimiter=',')
        rewards = csv_file[:, 1]

        steps = range(0, len(rewards))

        plt.plot(steps, rewards, label='Reward ' + nameFileCsv[:10])

    plt.title('Titolone')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

def plot_q_values(files, indeces):
    plt.clf()

    for nameFileCsv in files:
        csv_file = genfromtxt(FOLDER + nameFileCsv + '.csv', delimiter=',')
        q_value_online = csv_file[:, indeces[0]]
        q_value_target = csv_file[:, indeces[1]]

        steps = range(0, len(q_value_online))

        plt.plot(steps, q_value_online, label='Q-value ' + nameFileCsv[:10])

    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


def main():

    files = [
        "mse-DQN-seed-52-CartPole-v0-2018-06-26T13-56-28", 
        "mse-DQN-seed-42-CartPole-v0-2018-06-26T10-48-11"
    ]

    # step , reward, q-online, q-target
    plot_rewards(files)
    plot_q_values(files, indeces=[2, 3])

    files_epochs = [csv_file + '-epoch' for csv_file in files]

    # epoch, q_online, q_target
    plot_q_values(files_epochs, indeces=[1, 2])
    

if __name__ == "__main__":
    main()
