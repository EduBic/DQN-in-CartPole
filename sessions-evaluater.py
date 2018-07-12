
import os
import numpy as np
import csv


PARENT_FOLDER = "sessions/"

with open("sess-results.csv", 'w', newline='') as csvfile:
    fieldnames = ['model', 'seed', 'avg_reward', 'std_reward', 'num_solver', 'first_solver']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    sessions_folders = os.listdir(PARENT_FOLDER)

    # Load all file for each folders
    for session in sessions_folders:

        folder = PARENT_FOLDER + session + "/"
        files = os.listdir(folder)
        files.sort(key=len)

        files.pop(0) # remove model with 0 steps of training
        files = files[:83]

        # General data on session
        num_model = 0
        avg_reward = 0
        num_solved = 0
        first_win = None

        all_rewards = np.array([])


        print("\nSession:", session)

        for f in files:
            file_csv = np.genfromtxt(folder + f, delimiter=',')

            # Episode , Total Reward
            eisodes = file_csv[1:, 0] # from 1 to 100
            rewards = file_csv[1:, 1]
            
            file_avg_rew = np.mean(rewards)
            all_rewards = np.append(all_rewards, rewards)

            avg_reward += 1 / (num_model + 1) * (file_avg_rew - avg_reward)

            solved = False
            if file_avg_rew >= 195:
                solved = True
                num_solved += 1
                if first_win is None:
                    print("FIRST WIN")
                    first_win = f

            print(f)
            print("Solved:", solved)
            print("Mean reward:", file_avg_rew)
            print("")

            num_model += 1
            #input("Continue? Press any keyes")

        # Extract info from name of model
        session = session[7:].replace("_", '')

        model_type = ''.join([c for c in session if not c.isdigit()]).replace("d", " deep")
        seed = session.replace("DDQN", '').replace("DQN", '').replace("d", '')

        writer.writerow({
            fieldnames[0]: model_type,
            fieldnames[1]: seed,
            fieldnames[2]: avg_reward,
            fieldnames[3]: np.std(all_rewards),
            fieldnames[4]: num_solved,
            fieldnames[5]: first_win.replace('_e','').replace('.csv', '')
        })
            
        print("Conclusion")
        print("\tFirst win", first_win)
        print("\tAvg Reward", avg_reward)
        print("\tNum solved", num_solved)
        
        print(session + " Finished\n")


