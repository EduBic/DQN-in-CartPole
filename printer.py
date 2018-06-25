import matplotlib.pyplot as plt
from numpy import genfromtxt

epoche = 50


def print_rewards(rewards_mat):
    plt.clf()

    len1 = len(rewards_mat[0])
    len2 = len(rewards_mat[1])

    minimum = min(len1, len2)

    steps = range(0, minimum)

    plt.plot(steps, rewards_mat[0][:minimum], 'b', label='Batch size 64')
    plt.plot(steps, rewards_mat[1][:minimum], 'r', label='Batch size D')

    plt.title('Titolone')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()


def print_q_value(q_mat):
    plt.clf()

    len1 = len(q_mat[0])
    len2 = len(q_mat[1])

    minimum = min(len1, len2)
    print(len(q_mat[0]))
    steps = range(0, minimum)

    plt.plot(steps, q_mat[0][:minimum], 'b', label='q-online')
    plt.plot(steps, q_mat[1][:minimum], 'r', label='q-target')

    plt.title('Titolone')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.show()


def print_meam_epoch_q_value(mean_q_online, mean_q_target):
    plt.clf()

    x = len(mean_q_online)

    epochs = range(0, x)

    plt.plot(epochs, mean_q_online, 'r', label='q_online')
    plt.plot(epochs, mean_q_target, 'b--', label='q_target')

    plt.title('Mean epoch q_value')
    plt.xlabel('Epoch')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.show()

def main():
    # header:
    # step , reward

    '''num_files = 2
    rewards = [0 for i in range(num_files)]
    q_value = [0 for i in range(num_files)]

    nameFileCsv = "doubleDQN-2-seed-52-CartPole-v0-2018-06-24T11-55-06"

    my_csv = genfromtxt('DQN/results/' + nameFileCsv + '.csv', delimiter=',')
    rewards[0] = my_csv[0:, 2]
    q_value[0] = my_csv[0:, 3]

    my_csv = genfromtxt('DQN/results/' + nameFileCsv + '.csv', delimiter=',')
    rewards[1] = my_csv[0:, 2]
    q_value[1] = my_csv[0:, 4]

    print_rewards(rewards)
    print_q_value(q_value)'''

    q_target = []
    q_online = []


    folder = 'csvDQN/'
    name_file = 'ccc64.csv'

    # epoch, q_online, q_target

    my_csv = genfromtxt(folder + name_file, delimiter=',')
    q_online = my_csv[:, 1]
    q_target = my_csv[:, 2]

    print_meam_epoch_q_value(q_online, q_target)


if __name__ == "__main__":
    main()
