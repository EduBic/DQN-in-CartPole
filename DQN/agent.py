
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

    def __init__(self, stateDataCount, actionCount,
                 double_q_learning,
                 max_eps,
                 min_eps,
                 update_target_frequency,
                 mLambda,
                 memory_capacity,
                 mem_batch_size,
                 gamma):

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

        self.env_max_step = 200 # Only for CartPole environment
        self.index_results = 0

        self.q_target_results = np.zeros(self.env_max_step)
        self.q_online_results = np.zeros(self.env_max_step)

        self.epoch = mem_batch_size # DEBUG
        self.q_target_epoch_results = np.zeros(self.epoch)
        self.q_online_epoch_results = np.zeros(self.epoch)
        self.mean_q_target_epoch = np.array([])
        self.mean_q_online_epoch = np.array([])

    def act(self, curr_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(curr_state))

    def observe(self, sample): # (s, a, r, s') tupla
        self.memory.add(sample)
        self.steps += 1

        curr_state = sample[0]

        if self.double_q_learning:  # For Double DQN

            if self.steps % self.update_target_frequency == 0:
                self.brain.update_target_model()
                print("Steps (double)", self.steps) # DEBUG

            max_pred_target = np.amax(self.brain.predictOne_target(curr_state))
            self.q_target_results[self.index_results] = max_pred_target
            self.q_target_epoch_results[(self.steps - 1) % self.epoch] = max_pred_target
        
        else: # For DQN
            if self.steps % 1000 == 0:
                print("Steps", self.steps)

        max_pred_online = np.amax(self.brain.predictOne(curr_state))
        self.q_online_results[self.index_results] = max_pred_online
        self.q_online_epoch_results[(self.steps - 1) % self.epoch] = max_pred_online

        self.index_results += 1

        # Decay the learning
        self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(- self.mLambda * self.steps)

        if self.steps % self.epoch == 0:
            self.write_epoch()


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

            target = p[i]
            if new_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(p_[i])

            x[i] = state
            y[i] = target

        self.brain.train(x, y)

    # Logging methods

    def get_and_reinit_q_target_results(self):
        res = np.copy(self.q_target_results)
        self.q_target_results = np.zeros(self.env_max_step)
        self.index_results = 0
        return res

    def get_and_reinit_q_online_results(self):
        res = np.copy(self.q_online_results)
        self.q_online_results = np.zeros(self.env_max_step)
        self.index_results = 0
        return res

    def set_writer_epochs(self, writer):
        self.writer = writer

    def write_epoch(self):
        #print("Write epochs", value)
        self.writer.writerow({
            self.writer.fieldnames[0]: self.steps / self.epoch,
            self.writer.fieldnames[1]: np.mean(self.q_online_epoch_results),
            self.writer.fieldnames[2]: np.mean(self.q_target_epoch_results),
            self.writer.fieldnames[3]: self.epsilon
        })
