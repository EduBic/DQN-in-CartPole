
import os
import numpy as np
import csv


parent_fold = "sessions/"

sessions = [
    "mod_20b_DQN",
    "mod_23b_DQN_deep",
    "mod_26b_DDQN",
    "mod_29b_DDQN_deep"
]

with open("sess-results", 'w', newline='') as csvfile:
    fieldnames = ['session', 'file', 'avg_reward', 'num_solver', 'first_solver']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Load all file for each folders
    for sess_type in sessions:

        folder = parent_fold + sess_type + "/"
        files = os.listdir(folder)

        files.pop(0) # remove model with 0 steps of training

        # General data on sess_type
        num_model = 0
        avg_reward = 0
        num_solved = 0
        first_win = None


        print("\nSession:", sess_type)

        for f in files:
            file_csv = np.genfromtxt(folder + f, delimiter=',')

            # Episode , Total Reward
            eisodes = file_csv[1:, 0] # from 1 to 100
            rewards = file_csv[1:, 1]
            
            file_avg_rew = np.mean(rewards)

            avg_reward += 1 / (num_model + 1) * (file_avg_rew - avg_reward)

            solved = False
            if file_avg_rew >= 195:
                solved = True
                num_solved += 1
                if first_win is None:
                    first_win = f

            print(f)
            print("Solved:", solved)
            print("Mean reward:", file_avg_rew)
            print("")

            num_model += 1

        writer.writerow({
            fieldnames[0]: sess_type,
            fieldnames[1]: str(f),
            fieldnames[2]: avg_reward,
            fieldnames[3]: num_solved,
            fieldnames[4]: str(first_win)
        })
            
        print("Conclusion")
        print("\tFirst win", first_win)
        print("\tAvg Reward", avg_reward)
        print("\tNum solved", num_solved)
        
        print(sess_type + " Finished\n")


