import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Global settings
RESTRICTOR_FACTOR = 8 
PARENT_FOLDER = "Data/"


# Utility functions

def save_new_q_values_in_csv(new_q_values):
    with open(str(data_file) + "-restricted.csv", 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'q-online-value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(len(new_q_values)):
            writer.writerow({
                fieldnames[0]: epoch,
                fieldnames[1]: new_q_values[epochs]
            })

def plot_new_q_values(new_q_values):
    plt.clf()

    xs = range(0, len(new_q_values))
    ys = new_q_values

    plt.plot(xs, ys, linewidth=0.4)
    plt.show()


# MAIN

# get the epoch files
data_files = [f for f in os.listdir(PARENT_FOLDER) if "-epoch" in str(f)]

# iterate the csv file
#for data_file in data_files:
data_file = data_files[0]
print("\n",data_file)

# open csv file
# epoch, q-online-vale, q-target-value, ...
file_csv = np.genfromtxt(PARENT_FOLDER + data_file, delimiter=',')

epochs = file_csv[1:, 0] # select from 1 row (skip header) for 0 column
q_values = file_csv[1:, 1]

# Particular case
while not len(epochs) % RESTRICTOR_FACTOR == 0:
    print("Update length", len(epochs))
    epochs = epochs[:-1]
    q_values = q_values[:-1]


# generate new epoch

num_new_epochs = int(np.floor(len(epochs) / RESTRICTOR_FACTOR))
print("Num of epoch", len(epochs))
print("New num of epochs:", num_new_epochs)

# Init new array
new_epochs = np.zeros(num_new_epochs)
new_q_values = np.zeros(num_new_epochs)

# compute new q-values
splitted_q_values = np.split(q_values, num_new_epochs)
new_q_values = np.mean(splitted_q_values, 1) # for each row compute the mean

# save_new_q_values_in_csv(new_q_values)
plot_new_q_values(new_q_values)


    