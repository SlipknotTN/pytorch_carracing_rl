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
import torch
from torchvision import transforms

from qlearning.common import encoded_actions
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
    for num_episode in range(0, config.num_episodes):
        state = env.reset()
        state_bw = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        for _ in range(0, config.input_num_frames):
            # TODO: Add transform composition to convert input to float [-1.0, 1.0] range
            input_states.append(transforms.ToTensor()(state_bw))

        # Reply the first frame config.input_num_frames times
        done = False
        while not done:
            # Prepare input
            input_tensor = torch.cat(list(input_states), dim=0)
            # Add batch size dimension
            input_tensor = input_tensor.unsqueeze(dim=0)
            input_tensor.cuda()

            # TODO: Choose action from epsilon-greedy policy
            state_action_values = model(input_tensor)

            # TODO: Convert to continuous space

            # TODO: Apply action

            # TODO: Update model weights according to new state and taken action
            # The target is the reward + gamma * max q(new_state, a, w)

            # TODO: Update the deque
            pass



if __name__ == "__main__":
    main()
