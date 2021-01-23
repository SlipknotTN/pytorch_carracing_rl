from collections import deque

import numpy as np


class ExperienceBuffer(object):

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def size(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        size = min(batch_size, self.size())
        sample_indexes = np.random.choice(np.arange(self.size()),
                                          size=size,
                                          replace=False)
        return [self.buffer[i] for i in sample_indexes]