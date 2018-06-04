import gym
import numpy as np

from gym import wrappers

env = gym.make("CartPole-v0")

best_length = 0
episode_lengths = []

best_weights = np.zeros(4)

def main_loop(env, new_weights):
    obs = env.reset()
    done = False
    cnt = 0 

    while not done:
        env.render()
        cnt += 1
        action = 1 if np.dot(obs, new_weights) > 0 else 0
        obs, reward, done, _ = env.step(action)

    return cnt


for i in range(100):
    print(i)
    new_weights = np.random.uniform(-1.0, 1.0, 4)
    
    length = []
    for j in range(100):
        print(j)
        obs = env.reset()
        done = False
        cnt = 0 

        while not done:
            env.render()
            cnt += 1
            action = 1 if np.dot(obs, new_weights) > 0 else 0
            obs, reward, done, _ = env.step(action)

        length.append(cnt)

    avg_length = float(sum(length) / len(length))

    if avg_length > best_length:
        best_length = avg_length
        best_weights = new_weights
    
    episode_lengths.append(avg_length)

    if i % 10 == 0:
        print("Best length", best_length)


print("Start video")

env = wrappers.Monitor(env, "MovieFile2", force=True)

obs = env.reset()
done = False
cnt = 0 

while not done:
    env.render()
    cnt += 1
    action = 1 if np.dot(obs, new_weights) > 0 else 0
    obs, reward, done, _ = env.step(action)

print("game lasted moves", cnt)