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


def main():
    # header:
    # step , reward

    num_files = 2
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
    print_q_value(q_value)


if __name__ == "__main__":
    main()
