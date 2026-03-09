import random

class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity

    def add(self, obs, action, reward, next_obs, done):

        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append((obs, action, reward, next_obs, done))


    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


    def size(self):
        return len(self.buffer)