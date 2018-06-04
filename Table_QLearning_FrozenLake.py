
import gym
import numpy as np

env = gym.make("FrozenLake-v0")

#print("obs space n", env.observation_space)
#print("action space n", env.action_space)

obs_space_dim = env.observation_space.n
action_space_dim = env.action_space.n

# Init action value function
# A table of dimension: (row, col)
Q = np.zeros([obs_space_dim, action_space_dim])

# hyperparams
learn_rate = 0.8
gamma = 0.95

num_episodes = 2000

rewards = []

def getRandomAction():
    return np.random.rand(1, action_space_dim)

for episode in range(num_episodes):
    # get init state
    obs = env.reset() 
    total_reward = 0

    for step in range(99):
        Q_value = Q[obs, :]
        action = np.argmax(Q_value + 
            getRandomAction() * (1. / (episode + 1)))

        new_obs, reward, done, _ = env.step(action)

        # Update Q-table
        td_target = learn_rate * (reward + gamma * np.max(Q[new_obs, :]))
        td_error = td_target - Q[obs, action]
        Q[obs, action] += td_error

        total_reward += reward
        obs = new_obs

        if done: break
    
    rewards.append(total_reward)
    
        
print("Score over time:", sum(rewards) / num_episodes)
print ("Final Q tables values:", Q)
