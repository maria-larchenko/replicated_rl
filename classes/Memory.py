from collections import namedtuple, deque
import random

import numpy as np
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final'))


class ReplayMemory:

    def __init__(self, device, seed=None, capacity=10000):
        random.seed(seed)
        self.device = device
        self.memory = deque([], maxlen=capacity)

    def to_tensor(self, x, dtype=torch.float32,):
        return torch.as_tensor(np.array(x), dtype=dtype).to(self.device)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample_batch(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch, see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))

    def sample(self, batch_size):
        batch = self.sample_batch(batch_size)
        states = self.to_tensor(batch.state)
        actions = self.to_tensor(batch.action, dtype=torch.int64)
        rewards = self.to_tensor(batch.reward)
        next_states = self.to_tensor(batch.next_state)
        finals = self.to_tensor(batch.final)
        return states, actions, rewards, next_states, finals

    def __len__(self):
        return len(self.memory)


class BackwardMemory:

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def pop(self):
        return self.memory.pop()

    def clear(self):
        self.memory = deque([], maxlen=self.capacity)

    def __len__(self):
        return len(self.memory)

