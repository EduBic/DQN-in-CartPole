
import gym
import numpy as np

# init environment
env = gym.make("MountainCarContinuous-v0")

action_set = env.action_space
print("Action space\n", dir(action_set))

# Execution options
tot_episodes = 1

# Init arbitrarily Q function


# Execute
for episode in range(tot_episodes):

    # Init state
    observation = env.reset()

    for _ in range(1000):
        env.render()

        # take a random action np.array() of 1-dim
        action = env.action_space.sample() 
        env.step(action)