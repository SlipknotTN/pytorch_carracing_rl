from typing import Union, Tuple

import numpy as np

"""Discrete to continuous actions conversion"""
steer_to_continuous = {0: 0.0, 1: -1.0, 2: 1.0}
gas_to_continuous = {0: 0.0, 1: 1.0}
brake_to_continuos = {0: 0.0, 1: 0.8}  # 0.8 for easier control, see also https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py

"""Continuous to discrete actions conversion"""
steer_to_discrete = {0.0: 0, -1.0: 1, 1.0: 2}
gas_to_discrete = {0.0: 0, 1.0: 1}
brake_to_discrete = {0.0: 0, 0.8: 1}

"""Single dimension action space mapping, full combinations """
complex_encoded_actions = {
    0: (0, 0, 0),
    1: (0, 0, 1),
    2: (0, 1, 0),
    3: (0, 1, 1),

    4: (1, 0, 0),
    5: (1, 0, 1),
    6: (1, 1, 0),
    7: (1, 1, 1),

    8: (2, 0, 0),
    9: (2, 0, 1),
    10: (2, 1, 0),
    11: (2, 1, 1)
}

complex_decoded_actions = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 2,
    (0, 1, 1): 3,

    (1, 0, 0): 4,
    (1, 0, 1): 5,
    (1, 1, 0): 6,
    (1, 1, 1): 7,

    (2, 0, 0): 8,
    (2, 0, 1): 9,
    (2, 1, 0): 10,
    (2, 1, 1): 11
}

"""
Single dimension action space mapping, simple combinations (only one command max):
- no actions
- turn left
- turn right
- gas
- brake
"""
simple_encoded_actions = {
    0: (0, 0, 0),  # no action
    1: (1, 0, 0),  # turn left
    2: (2, 0, 0),  # turn right
    3: (0, 1, 0),  # gas
    4: (0, 0, 1)   # brake
}

simple_decoded_actions = {
    (0, 0, 0): 0,  # no action
    (1, 0, 0): 1,  # turn left
    (2, 0, 0): 2,  # turn right
    (0, 1, 0): 3,  # gas
    (0, 0, 1): 4   # brake
}

"""
Single dimension action space mapping, simple combinations (only one command max excluding the brake):
- no actions
- turn left
- turn right
- gas
"""
basic_encoded_actions = {
    0: (0, 0, 0),  # no action
    1: (1, 0, 0),  # turn left
    2: (2, 0, 0),  # turn right
    3: (0, 1, 0),  # gas
}

basic_decoded_actions = {
    (0, 0, 0): 0,  # no action
    (1, 0, 0): 1,  # turn left
    (2, 0, 0): 2,  # turn right
    (0, 1, 0): 3,  # gas
}


def get_encoded_actions(action_complexity: str):
    if action_complexity == "complex":
        return complex_encoded_actions
    if action_complexity == "simple":
        return simple_encoded_actions
    if action_complexity == "basic":
        return basic_encoded_actions
    raise Exception(f"Unknown action complexity {action_complexity}")


def get_decoded_actions(action_complexity: str):
    if action_complexity == "complex":
        return complex_decoded_actions
    if action_complexity == "simple":
        return simple_decoded_actions
    if action_complexity == "basic":
        return basic_decoded_actions
    raise Exception(f"Unknown action complexity {action_complexity}")


def get_continuous_actions(discrete_actions: Union[np.ndarray, Tuple[int]]) -> np.ndarray:
    """
    Convert from discrete to continuous action space
    Steer: [0, 1, 2] -> [0.0, -1.0, 1.0]
    Gas: [0, 1] -> [0.0, 1.0]
    Brake: [0, 1] -> [0.0, 1.0]
    :param discrete_actions: actions in the discrete space (integers)
    :return: actions in the continuous space (float)
    """
    steer_cont = steer_to_continuous[discrete_actions[0]]
    gas_cont = gas_to_continuous[discrete_actions[1]]
    brake_cont = brake_to_continuos[discrete_actions[2]]
    return np.asarray([steer_cont, gas_cont, brake_cont])
