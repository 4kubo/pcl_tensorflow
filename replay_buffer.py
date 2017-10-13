import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_length=10000, batch_size=100, alpha=0.5):
        self.max_length = max_length
        self.batch_size = batch_size
        self.alpha = alpha
        self.buffer = []
        self.priority = []

    def add(self, rollout):
        self.buffer.append(rollout)

        sum_r = self.alpha*sum(rollout.rewards)
        self.priority += [0.9*np.exp(sum_r)]
        if self.max_length < len(self.buffer):
            del self.buffer[0]
            del self.priority[0]

    def sample(self, n_items):
        probs = 0.1/len(self.priority) + np.asarray(self.priority)
        probs /= probs.sum()

        n_items = n_items if n_items < len(self.buffer) else len(self.buffer)
        return np.random.choice(self.buffer, n_items, replace=False, p=probs)

    @property
    def trainable(self):
        if self.batch_size <= len(self.buffer):
            return True
        else:
            return False