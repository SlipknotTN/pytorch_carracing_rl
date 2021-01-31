from collections import deque

import numpy as np


class ExperienceBuffer(object):

    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = self.buffer.maxlen
        # Weight of the last frames of first state in the experience, for more accurate sampling.
        # Higher weight, higher probability to be chosen
        self.frame_weight = deque(maxlen=max_size)

    @property
    def size(self):
        return len(self.buffer)

    def add(self, experience):
        experience_weight = ExperienceBuffer.calc_frame_weight(experience)
        # Add weight to experience for easier debugging
        experience.append(experience_weight)
        self.buffer.append(experience)
        self.frame_weight.append(experience_weight)

    def is_full(self) -> bool:
        return self.size == self.max_size

    def sample(self, batch_size):
        size = min(batch_size, self.size)
        # TODO: Increase differences with softmax? Using temperature because most of the weights remain similar. WIP.
        temp = 100
        probs = np.exp(np.multiply(self.frame_weight, temp)) / np.sum(np.exp(np.multiply(self.frame_weight, temp)))
        sample_indexes = np.random.choice(np.arange(self.size),
                                          size=size,
                                          replace=False,
                                          p=probs)
        return [self.buffer[i] for i in sample_indexes]

    def __getitem__(self, item):
        return self.buffer[item]

    @classmethod
    def calc_frame_weight(cls, experience):
        # Get the last frame of the first state s of s, a, r, s'
        # The value of the frame is preprocessed for the network
        state = experience[0]
        last_frame = state[-1]
        last_frame_np = last_frame.clone().cpu().data.numpy()[0]
        last_frame_np += 1.0  # range [0.0, 2.0]
        last_frame_mean = np.mean(last_frame_np)
        # Magnitude of weight to be selected (1.0 - np.mean).
        # More road (black) you have, the more is possible to selected.
        # TODO: Manage out of the screen. Penalize mean < 0.0? Set max to 1.0 out of 2.0?. WIP
        last_frame_weight = min(2.0, max(0.0, 2.0 - last_frame_mean))
        # Try to penalize high weights (mostly black images)
        if last_frame_weight > 1.0:
            last_frame_weight /= 2.0
        return last_frame_weight
