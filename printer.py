import matplotlib.pyplot as plt
from numpy import genfromtxt


FOLDER = 'DQN/results/'

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
    plt.show()

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
    plt.show()


def main():

    files = [
        # DQN
        #"DQN-seed-52-2018-06-26T15-41-06",
        #"DQN-seed-42-06-26T16-56",

        # Deep DDQN
        #"DDQN-deep-52-06-28T09-55",
        #"DDQN-deep-32-06-28T15-13",

        # More Deep DDQN
        "DDQN-dd-52-06-29T10-48",
        "DDQN-dd-32-06-29T14-11",

        # Double DQN
        "DDQN-seed-52-2018-06-26T13-56-28",
        #"DDQN-seed-42-2018-06-26T10-48-11",
        
        # Lambda = 0.00001 -> Epsilon need more steps to decay to 0.01
        # "DQN-lambda-42-06-26T21-03",
        # "DDQN-32-lambda-06-27T17-06",
        # "DQN-lambda-52-06-27T09-40"
    ]

    # step , reward, q-online, q-target
    plot_rewards(files)
    plot_q_values(files, indeces=[2, 3], xlabel='episode')

    files_epochs = [csv_file + '-epoch' for csv_file in files]

    # epoch, q_online, q_target
    plot_q_values(files_epochs, indeces=[1, 2], xlabel='epoch')
    

if __name__ == "__main__":
    main()
