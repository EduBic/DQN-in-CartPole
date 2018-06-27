import matplotlib.pyplot as plt
import csv
import math
import numpy as np

NUM_STEP = 1000000 # steps of the agent
MAX_EPS = 1
MIN_EPS = 0.01
MY_LAMBDA = 0.00001

def make_csv():

    fileCsvPath = "eps-gen-Min" + str(MIN_EPS) + "-Max" + str(MAX_EPS) + "-Lambda" + str(MY_LAMBDA) + ".csv"

    with open(fileCsvPath, 'w', newline='') as csvfile:
        fieldnames = ['step', 'epsilon']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        const_ = MIN_EPS + (MAX_EPS - MIN_EPS)

        for step in range(NUM_STEP):
            writer.writerow({
                fieldnames[0]: step + 1,
                fieldnames[1]: const_ * math.exp(- MY_LAMBDA * step)   # epsilon
            })

def plot_eps():
    plt.clf()

    const_ = MIN_EPS + (MAX_EPS - MIN_EPS)

    steps = range(1, NUM_STEP + 1)
    results = np.zeros(NUM_STEP)

    for step in range(NUM_STEP):
        results[step] = const_ * math.exp(- MY_LAMBDA * step) # epsilon

    plt.plot(steps, results, 'b')

    plt.title("Epsilon")
    plt.xlabel('steps')
    plt.ylabel('eps-value')
    plt.show()



def main():

    plot_eps()



if __name__ == "__main__":
    main()