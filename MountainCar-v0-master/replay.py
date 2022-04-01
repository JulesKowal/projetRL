
import numpy as np
from collections import namedtuple, deque
import random

class ReplayBuffer:

    def __init__(self, bufferSize, batchSize):
        self.memory = deque(maxlen=bufferSize)  # internal memory (deque)
        self.batchSize = batchSize
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])

    def add(self, state, action, reward, nextState, done):
        e = self.experience(state, action, reward, nextState, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batchSize)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        nextStates = np.vstack([e.nextState for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, nextStates, dones)

    def __len__(self):
        return len(self.memory)