"""
TODO:
- Assert everything is running as expected
- Implement test/run script from trained model
- Implement experience replay and mini-batch
- Implement fixed Q-Target
"""
import argparse
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from qlearning.common import encoded_actions, get_continuous_actions, transform_input, get_input_tensor
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.alpha)

    # First implementation without experience replay, learning while exploring
    for num_episode in range(0, config.num_episodes):

        total_reward = 0.0
        print(f"Start episode {num_episode + 1}")
        epsilon = config.min_epsilon + (config.initial_epsilon - config.min_epsilon) * np.exp(-config.eps_decay_rate * num_episode)
        print(f"epsilon: {epsilon}")
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

            # Choose action from epsilon-greedy policy
            state_action_values = model(input_tensor_explore)
            state_action_values_np = state_action_values.cpu().data.numpy()[0]
            if np.random.rand() < epsilon:
                action_id = np.random.randint(0, len(encoded_actions))
            else:
                action_id = np.argmax(state_action_values_np)
            # Convert to continuous action space
            action_discrete = encoded_actions[action_id]
            action = get_continuous_actions(action_discrete)

            # Apply action
            next_state, reward, done, _ = env.step(action)
            env.render()

            # Update the deque
            next_state_bw = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
            input_states.append(transform_input()(next_state_bw))
            input_states.popleft()

            # Use the same reached state to train
            input_tensor_train = get_input_tensor(input_states)
            next_state_action_values = model(input_tensor_train)
            new_state_action_values_np = next_state_action_values.cpu().data.numpy()[0]

            # Update model weights according to new state and taken action (batch_size is 1)
            # The target is the reward + gamma x max q(new_state, any_action, w)
            target = reward + config.gamma * torch.max(next_state_action_values[0])
            # td_error = target - q(state, action, w)
            # Weights update = alfa x td_error x gradient_w.r.t._w(q(state, action, w))
            # With PyTorch we use learning_rate and MSE error
            # calculate the loss between predicted and target class
            loss = criterion(target, state_action_values[0][action_id])
            # Reset the parameters (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # Update the episode reward
            total_reward += reward

        # End of episode, epsilon decay
        print(f"End of episode {num_episode + 1}, total_reward: {total_reward}\n")

        if (num_episode + 1) % 10 == 0:
            print("Saving model")
            torch.save(model.state_dict(), f"model_baseline_{num_episode + 1}.pth")


if __name__ == "__main__":
    main()
