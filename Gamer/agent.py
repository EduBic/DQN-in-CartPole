
import math
import random
import numpy as np

from brain import Brain
from memory import Memory


class Agent:

    def __init__(self, stateDataCount, actionCount, model):

        self.brain = Brain(stateDataCount, actionCount, model)

        self.steps = 0
        self.stateDataCount = stateDataCount
        self.actionCount = actionCount

        self.epsilon = 0.01

    def act(self, curr_state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCount - 1)
        else:
            return np.argmax(self.brain.predictOne(curr_state))