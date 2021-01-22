import argparse
from collections import deque

import cv2
import gym
import numpy as np
import torch

from qlearning.common import encoded_actions, get_continuous_actions, transform_input, get_input_tensor
from qlearning.config import ConfigParams
from qlearning.model.ModelBaseline import ModelBaseline


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Q-Learning PyTorch test script")
    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to the config file used to train the model")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the model path")
    parser.add_argument("--test_episodes", required=True, type=int, help="Number of episodes to run")
    parser.add_argument("--env_render", action="store_true", help="Render environment in GUI")
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
    model.load_state_dict(torch.load(args.model_path))

    print(model)
    model.cuda()

    for num_episode in range(0, args.test_episodes):

        total_reward = 0.0
        print(f"Start episode {num_episode + 1}")
        state = env.reset()

        # Deque to store num_frames as input, reset at every episode
        input_states = deque()
        state_bw = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        for _ in range(0, config.input_num_frames):
            # Apply transform composition to convert input to float [-1.0, 1.0] range
            input_states.append(transform_input()(state_bw))

        # Reply the first frame config.input_num_frames times
        done = False
        while not done:

            input_tensor_explore = get_input_tensor(input_states)

            # Choose the action with higher confidence
            state_action_values = model(input_tensor_explore)
            state_action_values_np = state_action_values.cpu().data.numpy()[0]
            action_id = np.argmax(state_action_values_np)
            # Convert to continuous action space
            action_discrete = encoded_actions[action_id]
            action = get_continuous_actions(action_discrete)

            # Apply action
            next_state, reward, done, _ = env.step(action)
            if args.env_render:
                env.render()

            # Update the deque
            next_state_bw = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            if args.env_render:
                cv2.imshow("State", next_state_bw)
                cv2.waitKey(1)
            input_states.append(transform_input()(next_state_bw))
            # Remove the oldest frame
            input_states.popleft()

            # Update the episode reward
            total_reward += reward

        # End of episode, epsilon decay
        print(f"End of episode {num_episode + 1}, total_reward: {total_reward}\n")


if __name__ == "__main__":
    main()
