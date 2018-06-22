
import math
import random
import numpy as np

from brain import Brain
from memory import Memory

'''
% Agent
 max_eps=1
 min_eps=0.01
 update_target_frequency=1000
 mLambda=0.001
 memory_capacity=10000
 mem_batch_size=64
 gamma=0.99
%
'''

class Agent:

    memory_capacity=10000

    def __init__(self, stateDataCount, actionCount,
                 double_q_learning=True,
                 max_eps=1,
                 min_eps=0.01,
                 update_target_frequency=800,
                 mLambda=0.001,
<<<<<<< HEAD
                 memory_capacity=10000,
                 mem_batch_size=128,
=======
                 memory_capacity=100000,
                 mem_batch_size=64,
>>>>>>> c06aa74988e9d66b773777f4f5846177c81bc1eb
                 gamma=0.99):

        self.double_q_learning = double_q_learning

        self.mLambda = mLambda
        self.memory_capacity = memory_capacity
        self.mem_batch_size = mem_batch_size
        self.gamma = gamma

        self.steps = 0
        self.stateDataCount = stateDataCount
        self.actionCount = actionCount

        self.max_eps = max_eps
        self.min_eps = min_eps

        self.update_target_frequency = update_target_frequency

        self.epsilon = max_eps

        self.brain = Brain(stateDataCount, actionCount)
        self.memory = Memory(self.memory_capacity)

        # state to test q value
        self.q_state = np.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
        self.q_target_results = np.array([])
        self.q_online_results = np.array([])

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

        if self.double_q_learning:
            pred_target = self.brain.predictOne_target(self.q_state)
            self.q_target_results = np.append(self.q_target_results, pred_target)

        pred_online = self.brain.predictOne(self.q_state)
        self.q_online_results = np.append(self.q_online_results, pred_online)


        # Decay the learning
        self.steps += 1
        self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(- self.mLambda * self.steps)


    def replay(self):
        batch = self.memory.get_rand_samples(self.mem_batch_size)
        batchLen = len(batch)

        no_state = np.zeros(self.stateDataCount)

        curr_states = np.array([obs[0] for obs in batch])
        new_states = np.array([(no_state if obs[3] is None else obs[3]) for obs in batch])

        p = self.brain.predict(curr_states)

        if self.double_q_learning:
            p_ = self.brain.predict_target(new_states)
        else:
            p_ = self.brain.predict(new_states)

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
                t[action] = reward + self.gamma * np.amax(p_[i])

            x[i] = state
            y[i] = t

        self.brain.train(x, y)

    def get_and_reinit_q_target_results(self):
        res = np.copy(self.q_target_results)
        self.q_target_results = np.array([])
        return res

    def get_and_reinit_q_online_results(self):
        res = np.copy(self.q_online_results)
        self.q_online_results = np.array([])
        return res
