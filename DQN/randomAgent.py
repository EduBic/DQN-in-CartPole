
import random

from memory import Memory
from agent import Agent

""" 
Fill a memory with a random policy. 
This speed up the learning process.
"""
class RandomAgent:

    def __init__(self, actionsCount, memory_capacity):
        self.memory = Memory(memory_capacity)
        self.actionsCount = actionsCount

    def act(self, state):
        return random.randint(0, self.actionsCount - 1)
    
    def observe(self, sample): # Sample = (s, a, r, s')
        self.memory.add(sample)

    def replay(self):
        pass