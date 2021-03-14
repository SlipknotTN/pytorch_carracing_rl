import argparse
import os
import pickle

import gym
import numpy as np
from pyglet.window import key

from qlearning.common.space import steer_to_discrete, gas_to_discrete, brake_to_discrete, get_decoded_actions
from qlearning.common.input_states import InputStates
from qlearning.common.experience_buffer import ExperienceBuffer


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Record human interaction with the environment")
    parser.add_argument("--experience_file", required=True, type=str, help="Experience pkl filepath")
    parser.add_argument("--experience_size", required=True, type=int, help="Maximum number of trajectories to save")
    parser.add_argument("--action_complexity", required=True, type=str, choices=["full", "simple", "basic"],
                        help="Discrete action space complexity of the discretization")
    parser.add_argument("--input_num_frames", required=True, type=int,
                        help="Number of frames to be considered as input state")
    parser.add_argument("--num_episodes", required=False, default=10, type=int, help="Number of episodes")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    cont_action = np.array([0.0, 0.0, 0.0])
    discrete_action = [0, 0, 0]

    # Experience buffer
    experience_buffer = ExperienceBuffer(max_size=args.experience_size)

    available_actions_decoded = get_decoded_actions(args.action_complexity)

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            cont_action[0] = -1.0
            discrete_action[0] = steer_to_discrete[cont_action[0]]
        if k == key.RIGHT:
            cont_action[0] = +1.0
            discrete_action[0] = steer_to_discrete[cont_action[0]]
        if k == key.UP:
            cont_action[1] = +1.0
            discrete_action[1] = gas_to_discrete[cont_action[1]]
        if k == key.DOWN:
            cont_action[2] = +0.8  # set 1.0 for wheels to block to zero rotation
            discrete_action[2] = brake_to_discrete[cont_action[2]]

    def key_release(k, mod):
        if k == key.LEFT and cont_action[0] == -1.0:
            cont_action[0] = 0
            discrete_action[0] = steer_to_discrete[0]
        if k == key.RIGHT and cont_action[0] == +1.0:
            cont_action[0] = 0
            discrete_action[0] = steer_to_discrete[0]
        if k == key.UP:
            cont_action[1] = 0
            discrete_action[1] = gas_to_discrete[0]
        if k == key.DOWN:
            cont_action[2] = 0
            discrete_action[2] = brake_to_discrete[0]

    env = gym.make('CarRacing-v0')
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    for num_episode in range(0, args.num_episodes):
        done = False
        state = env.reset()
        # Prepare starting input states
        input_states = InputStates(args.input_num_frames)
        input_states.add_state(state)
        total_reward = 0.0
        while done is False:
            action_to_take = np.copy(cont_action)
            discrete_action_to_take = np.copy(discrete_action)
            next_state, reward, done, info = env.step(action_to_take)
            total_reward += reward

            if done is False:
                # Update the input states that will be used to train model (num_frames input)
                s_length = input_states.size
                s = input_states.as_list()
                input_states.add_state(next_state)
                s1 = input_states.as_list()

                try:
                    action_id_to_take = available_actions_decoded[tuple(discrete_action_to_take.tolist())]
                    # Only reproducible tuples are added to the experience buffer
                    # print(f"action c: {action_to_take}, id: {action_id_to_take}")
                    # Store the experience trajectory (s, a, r, s1)
                    if s_length == args.input_num_frames:
                        experience_buffer.add([s, action_id_to_take, reward, s1])
                except KeyError:
                    # We don't support every combination of actions,
                    # it depends on action space complexity discretization
                    # print(f"action c: {action_to_take} not available in the discrete action space, skipping...")
                    pass

                env.render()
        else:
            print(f"Total reward: {total_reward}")

    env.close()

    # Save experience buffer to file
    assert experience_buffer.size == args.experience_size, \
        f"Recorded experience buffer too small {experience_buffer.size} vs {args.experience_size}"

    print("Saving ExperienceBuffer to file...")
    dirname = os.path.dirname(args.experience_file)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(args.experience_file, "wb") as out_fp:
        pickle.dump(experience_buffer, out_fp)
    print(f"ExperienceBuffer dump saved to \"{args.experience_file}\"")


if __name__ == "__main__":
    main()
