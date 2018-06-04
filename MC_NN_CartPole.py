
import gym
import random
import numpy as np
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import keras
from keras.layers import Dense
from keras.layers import Dropout

from statistics import mean, median
from collections import Counter

# hyperparam
LR = 1e-3 # learning rate

env = gym.make("CartPole-v0") # create environment
env.reset()

goal_steps = 500 # upper bound to reach goal state
score_requirement = 50 # throws away value <= 50
initial_games = 10000 # training set size
episodes = 5

def rand_game():
    for episode in range(episodes):
        env.reset()

        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample() # gets random action
            observation, reward, done, info = env.step(action) # returns new state (obs), rew, is it done?, info

            if done: break # ending constraint


def get_episode_samples():
    train_data = []
    scores = []
    acccepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []    # action value function
        prev_observation = []   # init state

        # first step
        action = random.randrange(0, 2) 
        observation, reward, done, info = env.step(action)
        prev_observation = observation

        for _ in range(goal_steps - 1):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward

            if done: break

        if score >= score_requirement:
            acccepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                train_data.append([data[0], output])

        env.reset()
        scores.append(score)

    train_data_save = np.array(train_data)
    np.save("saved.npy", train_data_save)

    print("Avg acc score:", mean(acccepted_scores))
    print("Median acc score:", median(acccepted_scores))
    print(Counter(acccepted_scores))

    return train_data

def neural_network_model_tflearn(input_size):
    batch_size = None # default
    input_data_shape = [batch_size, input_size, 1]

    network = input_data(shape = input_data_shape, name='input')

    network = fully_connected(network, 128, activation="relu")
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation="relu")
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation="relu")
    network = dropout(network, 0.8)

    # 2 neurons are the possible actions -> Left or Right
    network = fully_connected(network, 2, activation="softmax") # output neuron
    network = regression(network, optimizer="adam", learning_rate=LR,
                        loss="categorical_crossentropy", name="targets")

    model = tflearn.DNN(network, tensorboard_dir="log")

    return model
    
def train_model_tflearn(train_data, model=False):

    print("\nData sample:", train_data[0])
    extract_train_data = np.array([data[0] for data in train_data])

    print("\nTraining Data before:", extract_train_data.shape)
    X = extract_train_data.reshape(-1, len(train_data[0][0]), 1)  # observation
    y = np.array([data[1] for data in train_data])

    print("X shape:", X.shape)
    print("y shape:", len(y))

    print("Inputs newtork dimension:", len(X[0]))

    if not model:
        model = neural_network_model_tflearn(input_size = len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500,
                show_metric=True, run_id="openaistuff")

    return model

def neural_network_model_keras(input_size):
    input_data_shape = (input_size, )

    model = keras.Sequential()

    # input dim (*, input_size)
    model.add(Dense(units=128, activation='relu', input_shape=input_data_shape))
    # output dim (*, 128)
    #model.add(Dropout(rate=0.8))

    model.add(Dense(units=256, activation='relu'))
    #model.add(Dropout(rate=0.8))

    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(rate=0.8))

    model.add(Dense(units=256, activation='relu'))
    #model.add(Dropout(rate=0.8))
    
    model.add(Dense(units=128, activation='relu'))
    #model.add(Dropout(rate=0.8))

    model.add(Dense(units=2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

    return model

def train_model_keras(train_data, model=False):
    X = np.array([i[0] for i in train_data]) #.reshape(-1, len(train_data[0][0]), 1)  # observation
    y = np.array([i[1] for i in train_data])

    print("\nX shape:", X.shape)
    print("y shape:", len(y))

    if not model:
        model = neural_network_model_keras(input_size = len(X[0]))

    model.fit(x=X, y=y, batch_size=64, epochs=3)

    return model


def play_game(model):
    scores = []
    choices = []

    for _ in range(10):     # each_game
        score = 0
        game_memory = []
        prev_obs = np.array([])

        env.reset()

        for _ in range(goal_steps):
            env.render()

            if len(prev_obs) == 0:
                action = random.randrange(0, 2)
            else:
                # For TFlearn
                prev_obs_reshape_tflearn = prev_obs.reshape(-1, len(prev_obs), 1)
                #print("prev obs reshaped:", prev_obs_reshape_tflearn)

                # For keras
                prev_obs_reshaped_keras = prev_obs.reshape(1, 4)
                #print("prev obs reshaped 2:", prev_obs_reshaped_keras)

                prediction = model.predict(prev_obs_reshaped_keras)
                #prediction = model.predict(prev_obs_reshape_tflearn)[0]

                action = np.argmax(prediction)

            choices.append(action)
            
            new_obs, reward, done, info = env.step(action)
            prev_obs = new_obs

            # need for retrain
            game_memory.append([new_obs, action])
            score += reward

            if done: break

        scores.append(score)

    print("\nAvg Score", sum(scores) / len(scores))
    print("Choice 1: {}, Choice 0: {}".format(choices.count(1) / len(choices),
        choices.count(0) / len(choices)))


# Execution
print("\n")
training_data = get_episode_samples()
print("\n")
model = train_model_keras(training_data)
print("\n")
play_game(model)