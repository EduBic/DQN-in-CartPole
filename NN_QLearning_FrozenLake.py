
import gym
import numpy as np
import random

import tensorflow as tf

import matplotlib.pyplot as plt

# init env and tensorflow
env = gym.make("FrozenLake-v0")

tf.reset_default_graph()

# Declare the network
# input neurons 
inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))

Q_out = tf.matmul(inputs, W)
predict = tf.argmax(Q_out, 1)    # which action?


Q_next = tf.placeholder(shape=[1, 4], dtype=tf.float32)

# Compute the loss function
loss = tf.reduce_sum(tf.square(Q_next - Q_out))

# tools of NN learning
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


# Training the network
init = tf.initialize_all_variables()

gamma = 0.99
epsilon = 0.1
num_episodes = 2000

rewards = []

with tf.Session() as sess:
    sess.run(init)

    for episode in range(num_episodes):
        state = env.reset()
        tot_reward = 0

        for step in range(99):

            action, Q = sess.run([predict, Q_out], 
                feed_dict={inputs : np.identity(16)[state : state + 1]})

            # with epsilon probability explore
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()

            new_state, reward, done, _ = env.step(action[0])

            # compute the action value 
            # from all possible action from new_state
            Q_a_s = sess.run(Q_out, 
                feed_dict={inputs : np.identity(16)[new_state : new_state + 1]})

            max_Q = np.max(Q_a_s)   # select the max action

            Q_target = Q
            Q_target[0, action[0]] = reward + gamma * max_Q

            _, W1 = sess.run([updateModel, W], 
                feed_dict={inputs : np.identity(16)[state : state + 1], 
                            Q_next : Q_target})

            tot_reward += reward
            state = new_state

            if done: 
                epsilon = 1 / (episode / 50 + 10)
                break

    rewards.append(tot_reward)

print("Percent of succesfull episodes:", sum(rewards) / num_episodes)
plt.plot(rewards)
