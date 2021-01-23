from collections import deque
from typing import List, Optional

import cv2
import numpy as np
import torch

from qlearning.common.input_processing import transform_input


class InputStates(object):
    """
    Class containing the input states considering the concatenation of multiple frames
    """
    def __init__(self, num_frames: int):
        self.processed_frames = deque()
        self.bw_frames = deque()
        self.max_length = num_frames

    def prepare_starting_input_states(self, state: np.ndarray):
        for _ in range(0, self.max_length):
            self.add_state(state)

    def add_state(self, state: np.ndarray):
        """
        Add a new state to internal deques considering max length
        """
        state_bw = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        self.bw_frames.append(state_bw)
        self.processed_frames.append(transform_input()(state_bw))
        # Remove the oldest frame
        if len(self.processed_frames) > self.max_length:
            self.processed_frames.popleft()
            self.bw_frames.popleft()

    def get_last_bw_frame(self) -> Optional[np.ndarray]:
        if len(self.bw_frames) > 0:
            return self.bw_frames[-1]
        else:
            return None

    def as_list(self) -> List[torch.Tensor]:
        return list(self.processed_frames)
