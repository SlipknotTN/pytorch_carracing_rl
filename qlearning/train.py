"""
TODO:
- Solve system out of memory -> Temporary fix https://github.com/openai/gym/pull/2096
- Assert everything is running as expected (~)
- Implement experience recorded from human interaction
- Implement experience replay (DONE with random sampling)
- Implement fixed Q-Target
"""
import argparse
import pickle

import cv2
import gym
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from qlearning.common.env_interaction import take_most_probable_action
from qlearning.common.input_processing import get_input_tensor_list
from qlearning.common.space import encoded_actions, get_continuous_actions
from qlearning.common.ExperienceBuffer import ExperienceBuffer
from qlearning.common.InputStates import InputStates
from qlearning.config import ConfigParams
from qlearning.model.ModelBaseline import ModelBaseline


def run_validation_episode(env, config, model, env_render=True, debug_state=False):
    """
    Run a validation episode taking the most probable action at every step
    """
    total_reward = 0.0
    state = env.reset()
    # Eval mode
    model.eval()

    # Prepare starting input states
    input_states = InputStates(config.input_num_frames)
    input_states.add_state(state)

    # Warmup: Fill the input
    for _ in range(0, config.input_num_frames - 1):
        no_action_discrete = encoded_actions[4]
        no_action = get_continuous_actions(no_action_discrete)
        next_state, reward, done, _ = env.step(no_action)
        input_states.add_state(next_state)

    # Reply the first frame config.input_num_frames times
    done = False
    while not done:
        done, next_state, reward = take_most_probable_action(env, input_states, model)
        if env_render:
            env.render()

        # Update the input states
        input_states.add_state(next_state)
        if debug_state:
            last_frame_bw = input_states.get_last_bw_frame()
            cv2.imshow("Validation State", last_frame_bw)
            cv2.waitKey(1)

        # Update the episode reward
        total_reward += reward

    print(f"End of validation episode, total_reward: {total_reward}")
    # Restore train mode
    model.train()


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Q-Learning PyTorch training script")
    parser.add_argument("--config_file", required=True, type=str, help="Path to the config file")
    parser.add_argument("--env_render", action="store_true", help="Render environment in GUI")
    parser.add_argument("--debug_state", action="store_true", help="Show last state frame in GUI")
    parser.add_argument("--save_experience", action="store_true", help="Save experience memory for future analysis")
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

    # Experience buffer
    experience_buffer = ExperienceBuffer(max_size=config.experience_buffer_size)

    # First implementation without experience replay, learning while exploring
    for num_episode in range(0, config.num_episodes):

        total_reward = 0.0
        losses = []
        print(f"\nStart episode {num_episode + 1}")
        epsilon = config.min_epsilon + (config.initial_epsilon - config.min_epsilon) * np.exp(-config.eps_decay_rate * num_episode)
        print(f"epsilon: {epsilon}")
        print(f"Experience buffer length: {experience_buffer.size()}")
        state = env.reset()

        # Prepare starting input states
        input_states = InputStates(config.input_num_frames)
        input_states.add_state(state)
        # Warmup: Fill the input
        for _ in range(0, config.input_num_frames - 1):
            no_action_discrete = encoded_actions[4]
            no_action = get_continuous_actions(no_action_discrete)
            next_state, reward, done, _ = env.step(no_action)
            input_states.add_state(next_state)

        # Reply the first frame config.input_num_frames times
        done = False
        while not done:

            # EXPLORATION STEP
            input_tensor_explore = get_input_tensor_list([input_states.as_list()])

            # Choose action from epsilon-greedy policy
            state_action_values_explore = model(input_tensor_explore)
            state_action_values_explore_np = state_action_values_explore.cpu().data.numpy()[0]
            if np.random.rand() < epsilon:
                action_id = np.random.randint(0, len(encoded_actions))
            else:
                action_id = np.argmax(state_action_values_explore_np)
            # print(state_action_values_np)
            # Convert to continuous action space
            action_discrete = encoded_actions[action_id]
            action = get_continuous_actions(action_discrete)

            # Apply action
            next_state, reward, done, _ = env.step(action)
            if args.env_render:
                env.render()

            # Update the input states
            s = input_states.as_list()
            input_states.add_state(next_state)
            s1 = input_states.as_list()
            # Store the experience (s, a, r, s1) if episode not finished.
            # State size is #config.num_input_frames frames.
            if not done:
                experience_buffer.add([s, action_id, reward, s1])
            if args.debug_state:
                last_frame_bw = input_states.get_last_bw_frame()
                cv2.imshow("State", last_frame_bw)
                cv2.waitKey(1)

            # TRAINING STEP

            # Sample experience
            # TODO: Sample the best tuples by ranking by reward? Especially at the beginning? Lots
            # of tuples seems very similar and useless (car out of the road)).
            # Or better by lower reward? But we can't learn when the car is way out of track.
            # We want to learn to turn, probably we should "cluster" the states and avoid duplicates/similarity.
            # Use color average to detect the quantity of road in the image?
            sampled_experience = experience_buffer.sample(batch_size=config.batch_size)

            # Reshape from list of (s, a, r, s') to list(s), list(a), list(r), list(r')
            state_train, action_train, reward_train, next_state_train, _ = [list(elem) for elem in zip(*sampled_experience)]

            input_tensor_train_1 = get_input_tensor_list(state_train)
            state_action_values_train = model(input_tensor_train_1)

            input_tensor_train_2 = get_input_tensor_list(next_state_train)
            next_state_action_values_train = model(input_tensor_train_2)

            # Update model weights according to new state and taken action (batch_size is 1)
            # The target is the reward + gamma x max q(new_state, any_action, w)
            reward_train_t_cuda = torch.Tensor(reward_train).cuda()
            target = reward_train_t_cuda + config.gamma * torch.max(next_state_action_values_train, dim=1).values
            # td_error = target - q(state, action, w)
            # Weights update = alfa x td_error x gradient_w.r.t._w(q(state, action, w))
            # With PyTorch we use learning_rate and MSE error
            # calculate the loss between predicted and target class
            # Retrieve the state value for every action taken in the batch
            state_action_values_train_filtered = \
                torch.cat([state_action_values_train[batch_id][action_id].unsqueeze(0)
                           for batch_id, action_id in enumerate(action_train)], dim=0)

            # Update the weights
            loss = criterion(target, state_action_values_train_filtered)
            losses.append(loss.clone().cpu().data.numpy())
            # Reset the parameters (weights) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # Update the episode reward
            total_reward += reward

        # End of episode, epsilon decay
        print(f"End of episode {num_episode + 1}, total_reward: {total_reward}, avg_loss: {np.mean(losses)}")

        if args.save_experience:
            experience_dump_file = f"experience_{experience_buffer.size()}.pkl"
            with open(experience_dump_file, "wb") as out_fp:
                pickle.dump(experience_buffer, out_fp)
            print(f"ExperienceBuffer dump saved to \"{experience_dump_file}\"")

        if (num_episode + 1) % config.validation_frequency == 0:
            print(f"\nRun validation episode after {num_episode + 1} episodes")
            run_validation_episode(env, config, model, args.env_render, args.debug_state)

        if (num_episode + 1) % config.save_model_frequency == 0:
            print("Saving model")
            torch.save(model.state_dict(), f"model_baseline_{num_episode + 1}.pth")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
