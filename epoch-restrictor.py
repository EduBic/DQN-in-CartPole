import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Global settings

# epoch of 32 steps with restrictor factor of 16 means: 32*16 = 512 steps per epoch
RESTRICTOR_FACTOR = 32

PARENT_FOLDER = "EpochsCSV/"
SAVED_FOLDER = "NewEpochsCSV/"


# Utility functions

def save_new_q_values_in_csv(data_file, new_q_values):

    if not os.path.exists(SAVED_FOLDER):
        os.makedirs(SAVED_FOLDER)

    name_file = SAVED_FOLDER + data_file.replace('.csv', '') + "-restricted-" + str(RESTRICTOR_FACTOR)

    with open(name_file + ".csv", 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'q-online-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(new_q_values)):
            writer.writerow({
                fieldnames[0]: epoch,
                fieldnames[1]: new_q_values[epoch]
            })

def plot_new_q_values(new_q_values):
    plt.clf()

    xs = range(0, len(new_q_values))
    ys = new_q_values

    plt.plot(xs, ys, linewidth=0.4)
    plt.show()

def restrict_epochs(data_file):
    print()
    print("*", data_file)

    # open csv file
    # epoch, q-online-vale, q-target-value, ...
    file_csv = np.genfromtxt(PARENT_FOLDER + data_file, delimiter=',')

    epochs = file_csv[1:, 0] # select from 1 row (skip header) for 0 column
    q_values = file_csv[1:, 1]

    # Remove element until the restrictor factor division return 0
    while not len(epochs) % RESTRICTOR_FACTOR == 0:
        print("Update length", len(epochs))
        epochs = epochs[:-1]
        q_values = q_values[:-1]

    # generate new epoch
    num_new_epochs = int(np.floor(len(epochs) / RESTRICTOR_FACTOR))

    print("Num of epoch", len(epochs))
    print("New num of epochs:", num_new_epochs)

    # compute new q-values
    splitted_q_values = np.split(q_values, num_new_epochs)
    new_q_values = np.mean(splitted_q_values, 1) # for each row compute the mean

    # save_new_q_values_in_csv(data_file, new_q_values)
    plot_new_q_values(new_q_values)



# ** MAIN **

# get the epoch files
data_files = [f for f in os.listdir(PARENT_FOLDER) if "-epoch" in str(f)]

# iterate the csv file
#for data_file in data_files:
restrict_epochs(data_files[0])




    