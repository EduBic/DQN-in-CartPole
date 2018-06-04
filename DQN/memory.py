
import random

# Experience replay
class Memory:

    def __init__(self, max_capacity):
        self.samples = []
        self.max_capacity = max_capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.max_capacity:
            self.samples.pop(0)

    def get_rand_samples(self, n):
        n = min(n, len(self.samples))

        return random.sample(self.samples, n)

    def is_full(self):
        return len(self.samples) >= self.max_capacity