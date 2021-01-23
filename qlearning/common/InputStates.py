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
        self.processed_frames = deque(maxlen=num_frames)
        self.bw_frames = deque(maxlen=num_frames)

    def add_state(self, state: np.ndarray):
        """
        Add a new state to internal deques considering max length
        """
        state_bw = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        self.bw_frames.append(state_bw)
        self.processed_frames.append(transform_input()(state_bw))

    def get_last_bw_frame(self) -> Optional[np.ndarray]:
        if len(self.bw_frames) > 0:
            return self.bw_frames[-1]
        else:
            return None

    def as_list(self) -> List[torch.Tensor]:
        return list(self.processed_frames)
