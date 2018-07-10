import matplotlib.pyplot as plt
import numpy as np

FILE_RES = "sess-results.csv"

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data


# session, avg_reward, std_reward, num_solver, first_solver
# (str, float, float, int, int)
file_csv = np.genfromtxt(FILE_RES, delimiter=',', dtype=None)

print(file_csv)

sessions = file_csv[1:, 0]
avg_reward = [float(n) for n in file_csv[1:, 1]]
std_reward = file_csv[1:, 2]

print(avg_reward)

y_pos = np.arange(len(sessions))

ax.barh(y_pos, avg_reward, xerr=std_reward, align='center', color='green', ecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(sessions)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Average Reward')
ax.set_title('How fast do you want to go today?')

plt.show()