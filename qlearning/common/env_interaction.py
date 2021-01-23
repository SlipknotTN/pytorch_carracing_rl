from collections import deque
from typing import Tuple

import gym
import numpy as np

from qlearning.common.input_processing import get_input_tensor_list
from qlearning.common.space import encoded_actions, get_continuous_actions
from qlearning.common.InputStates import InputStates
from qlearning.model.ModelBaseline import ModelBaseline


def take_most_probable_action(env: gym.Env, input_states: InputStates, model: ModelBaseline) -> Tuple[bool, np.ndarray, float]:
    input_tensor_explore = get_input_tensor_list([input_states.as_list()])
    # Choose the action with higher confidence
    state_action_values = model(input_tensor_explore)
    state_action_values_np = state_action_values.cpu().data.numpy()[0]
    # If state_action_values doesn't change over input states, it is because
    # input state is the same, every time it is taken the same action with no effect -> stuck
    action_id = np.argmax(state_action_values_np)
    # Convert to continuous action space
    action_discrete = encoded_actions[action_id]
    action = get_continuous_actions(action_discrete)
    # Apply action
    next_state, reward, done, _ = env.step(action)
    return done, next_state, reward