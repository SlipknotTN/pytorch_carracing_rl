from typing import Deque, List

import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def transform_input():
    # Input from [0, 255] to [-1.0, 1.0]
    # We don't do Grayscale transformation here, because it doesn't accept numpy array as input,
    # but only PIL Images
    return transforms.Compose([
        transforms.ToTensor(),  # FloatTensor [0.0, 1.0]
        transforms.Normalize(mean=0.5, std=0.5)
    ])


def get_input_tensor_list(input_tensors: List[List[Tensor]]) -> torch.cuda.FloatTensor:
    """
    Create a tensor of shape [batch_size, num_frames, width, height] from a list of list of tensors
    """
    # Prepare input: concat multiple frames
    batch_input = []
    for single_input in input_tensors:
        # Concat multiple frames and add a dimension for the successive cat
        batch_input.append(torch.cat(single_input, dim=0).unsqueeze(dim=0))
    # Concat for a single input with batch size dimension
    input_tensor = torch.cat(batch_input, dim=0)
    # Move to GPU
    input_tensor = input_tensor.type(torch.cuda.FloatTensor)
    return input_tensor
