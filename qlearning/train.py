"""
TODO:
- Discretize action space (source file claims on/off is possible).
  - Steer: left, no, right.
  - Gas: no, yes.
  - Brake: no, yes.
- Implement model, 96x96x4 input should be ok (4 BW frames)
"""
import argparse
from collections import deque

import cv2
import gym
import numpy as np
import torch

from qlearning.common import encoded_actions, get_continuous_actions, transform_input
from qlearning.config import ConfigParams
from qlearning.model.ModelBaseline import ModelBaseline


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Q-Learning PyTorch training script")
    parser.add_argument("--config_file", required=True, type=str, help="Path to the config file")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    config = ConfigParams(args.config_file)

    env = gym.make('CarRacing-v0')

    model = ModelBaseline(
        input_size=env.observation_space.shape[0],
        input_frames=config.input_num_frames,
        output_size=len(encoded_actions)
    )
    print(model)
    model.cuda()
    model.train()

    input_states = deque()

    # First implementation without experience replay, learning while exploring
    epsilon = config.initial_epsilon
    for num_episode in range(0, config.num_episodes):
        state = env.reset()
        state_bw = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        for _ in range(0, config.input_num_frames):
            # Apply transform composition to convert input to float [-1.0, 1.0] range
            input_states.append(transform_input()(state_bw))

        # Reply the first frame config.input_num_frames times
        done = False
        while not done:
            # Prepare input
            input_tensor = torch.cat(list(input_states), dim=0)
            # Add batch size dimension
            input_tensor = input_tensor.unsqueeze(dim=0)
            # Move to GPU
            input_tensor = input_tensor.type(torch.cuda.FloatTensor)

            # Choose action from epsilon-greedy policy
            state_action_values = model(input_tensor)
            state_action_values_np = state_action_values.cpu().data.numpy()[0]
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action_id = np.argmax(state_action_values_np)
                # Convert to continuous action space
                next_action_discrete = encoded_actions[next_action_id]
                next_action = get_continuous_actions(next_action_discrete)

            # TODO: Apply action

            # TODO: Update model weights according to new state and taken action
            # The target is the reward + gamma * max q(new_state, a, w)

            # TODO: Update the deque

            # TODO: Epsilon decay
            pass


if __name__ == "__main__":
    main()
