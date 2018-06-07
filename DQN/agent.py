
import math
import random
import numpy as np

from brain import Brain
from memory import Memory

class Agent:
    #MAX_EPSILON = 1
    #MIN_EPSILON = 0.01

    LAMBDA = 0.001 # speed of decay

    MEMORY_CAPACITY = 10000
    MEM_BATCH_SIZE = 64

    GAMMA = 0.99

    def __init__(self, stateDataCount, actionCount, 
                max_eps=1, min_eps=0.01, update_target_frequency=1000):
        self.steps = 0
        self.stateDataCount = stateDataCount
        self.actionCount = actionCount

        self.max_eps = max_eps
        self.min_eps = min_eps

        self.update_target_frequency = update_target_frequency

        self.epsilon = max_eps

        self.brain = Brain(stateDataCount, actionCount)
        self.memory = Memory(Agent.MEMORY_CAPACITY)

    def act(self, curr_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(curr_state))
            
    def observe(self, sample): # (s, a, r, s') tupla
        self.memory.add(sample)

        if self.steps % self.update_target_frequency == 0:
            self.brain.update_target_model()
            print("Steps", self.steps)

        # Decay the learning
        self.steps += 1
        self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(-Agent.LAMBDA * self.steps)


    def replay(self):
        batch = self.memory.get_rand_samples(Agent.MEM_BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.stateDataCount)

        curr_states = np.array([obs[0] for obs in batch])
        new_states = np.array([(no_state if obs[3] is None else obs[3]) for obs in batch])

        p = self.brain.predict(curr_states)
        p_ = self.brain.predict_target(new_states)

        x = np.zeros((batchLen, self.stateDataCount))
        y = np.zeros((batchLen, self.actionCount))

        # For each samples compute target
        for i in range(batchLen):
            observation = batch[i]

            state = observation[0]
            action = observation[1]
            reward = observation[2]
            new_state = observation[3]

            t = p[i]
            if new_state is None:
                t[action] = reward
            else:
                t[action] = reward + Agent.GAMMA * np.amax(p_[i])
            
            x[i] = state
            y[i] = t

        self.brain.train(x, y)

    