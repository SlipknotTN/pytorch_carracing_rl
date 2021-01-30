"""
Test script to evaluate a trained mode
"""
import argparse

import cv2
import gym
import torch

from qlearning.common.env_interaction import take_most_probable_action
from qlearning.common.space import get_encoded_actions, get_continuous_actions
from qlearning.common.input_states import InputStates
from qlearning.common.config import ConfigParams
from qlearning.model.model_baseline import ModelBaseline


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Q-Learning PyTorch test script")
    parser.add_argument("--config_file", required=True, type=str,
                        help="Path to the config file used to train the model")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the model path")
    parser.add_argument("--test_episodes", required=True, type=int, help="Number of episodes to run")
    parser.add_argument("--env_render", action="store_true", help="Render environment in GUI")
    parser.add_argument("--debug_state", action="store_true", help="Show last state frame in GUI")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    config = ConfigParams(args.config_file)

    env = gym.make('CarRacing-v0')

    available_actions = get_encoded_actions(config.action_complexity)
    model = ModelBaseline(
        input_size=env.observation_space.shape[0],
        input_frames=config.input_num_frames,
        output_size=len(available_actions)
    )
    model.load_state_dict(torch.load(args.model_path))

    print(model)
    model.cuda()
    model.eval()

    for num_episode in range(0, args.test_episodes):

        total_reward = 0.0
        print(f"Start episode {num_episode + 1}")
        state = env.reset()

        # Prepare starting input states
        input_states = InputStates(config.input_num_frames)
        input_states.add_state(state)
        # Warmup: Fill the input
        for _ in range(0, config.input_num_frames - 1):
            no_action_discrete = available_actions[0]
            no_action = get_continuous_actions(no_action_discrete)
            next_state, reward, done, _ = env.step(no_action)
            input_states.add_state(next_state)

        # Reply the first frame config.input_num_frames times
        done = False
        while not done:
            done, next_state, reward = take_most_probable_action(env, input_states, model, available_actions)
            if args.env_render:
                env.render()

            # Update the input states
            input_states.add_state(next_state)
            if args.debug_state:
                last_frame_bw = input_states.get_last_bw_frame()
                cv2.imshow("State", last_frame_bw)
                cv2.waitKey(1)

            # Update the episode reward
            total_reward += reward

        # End of episode, epsilon decay
        print(f"End of episode {num_episode + 1}, total_reward: {total_reward}\n")


if __name__ == "__main__":
    main()
