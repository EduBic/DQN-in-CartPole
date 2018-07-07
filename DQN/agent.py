
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
                 diff_target_network,
                 double_DQN,
                 max_eps,
                 min_eps,
                 update_target_frequency,
                 mLambda,
                 memory_capacity,
                 mem_batch_size,
                 gamma,
                 deep_set):

        self.diff_target_network = diff_target_network
        self.double_DQN = double_DQN

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

        self.brain = Brain(stateDataCount, actionCount, deep_set=deep_set)
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

        self.loss_hystory_epoch = np.zeros(self.epoch)

    def act(self, curr_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(curr_state))

    def observe(self, sample): # (s, a, r, s') tupla
        self.memory.add(sample)
        self.steps += 1

        curr_state = sample[0]

        if self.diff_target_network:  # For Double DQN

            if self.steps % self.update_target_frequency == 0:
                self.brain.update_target_model()
                print("Steps (double)", self.steps) # DEBUG

            max_pred_target = np.amax(self.brain.predictOne_target(curr_state))
            self.q_target_results[self.index_results] = max_pred_target
            self.q_target_epoch_results[(self.steps - 1) % self.epoch] = max_pred_target
        
        else: # For DQN
            if self.steps % self.update_target_frequency == 0:
                print("Steps", self.steps)

        max_pred_online = np.amax(self.brain.predictOne(curr_state))
        self.q_online_results[self.index_results] = max_pred_online
        self.q_online_epoch_results[(self.steps - 1) % self.epoch] = max_pred_online

        self.index_results += 1

        DECAY_EXP = False
        print("BE CAREFUL!! Decay exponential:", DECAY_EXP)
        
        # Decay the learning
        if DECAY_EXP:
            self.epsilon = self.min_eps + (self.max_eps - self.min_eps) * math.exp(- self.mLambda * self.steps)
        else:
            self.epsilon = max(- 0.000099 * self.steps + self.max_eps, self.min_eps)


    def replay(self):
        batch = self.memory.get_rand_samples(self.mem_batch_size)
        batchLen = len(batch)

        no_state = np.zeros(self.stateDataCount)

        curr_states = np.array([obs[0] for obs in batch])
        new_states = np.array([(no_state if obs[3] is None else obs[3]) for obs in batch])

        online_pred_curr_state = self.brain.predict(curr_states)

        if self.diff_target_network:
            target_pred_new_state = self.brain.predict_target(new_states)
        else:
            target_pred_new_state = self.brain.predict(new_states)  # No target network

        if self.double_DQN:
            online_pred_new_state = self.brain.predict(new_states)  # Double Target network

        x = np.zeros((batchLen, self.stateDataCount))
        y = np.zeros((batchLen, self.actionCount))

        # For each samples compute target
        for i in range(batchLen):
            observation = batch[i]

            state = observation[0]
            action = observation[1]
            reward = observation[2]
            new_state = observation[3]

            target = online_pred_curr_state[i]

            if new_state is None:
                target[action] = reward
            elif self.double_DQN: # DoubleDQN
                target_action = np.argmax(online_pred_new_state[i])
                target[action] = reward + self.gamma * target_pred_new_state[i][target_action]
            else: # DQN
                target[action] = reward + self.gamma * np.amax(target_pred_new_state[i])

            x[i] = state
            y[i] = target

        loss_history = self.brain.train(x, y, self.mem_batch_size)

        self.loss_hystory_epoch[(self.steps - 1) % self.epoch] = loss_history[0]

        if self.steps % self.epoch == 0:
            self.write_epoch()

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
            self.writer.fieldnames[3]: self.epsilon,
            self.writer.fieldnames[4]: np.mean(self.loss_hystory_epoch)
        })
