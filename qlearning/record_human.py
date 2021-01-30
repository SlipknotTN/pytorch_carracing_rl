import argparse

import gym
import numpy as np
from pyglet.window import key

from qlearning.common.space import steer_to_discrete, gas_to_discrete, brake_to_discrete, get_decoded_actions


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Record human interaction with the environment")
    parser.add_argument("--experience_file", required=True, type=str, help="Experience pkl filepath")
    parser.add_argument("--experience_size", required=True, type=int, help="Maximum number of trajectories to save")
    parser.add_argument("--action_complexity", required=True, type=str, choices=["simple", "full", "simple_no_brake"],
                        help="Discrete action space complexity of the discretization")
    args = parser.parse_args()
    return args


def main():
    args = do_parsing()
    print(args)

    cont_action = np.array([0.0, 0.0, 0.0])
    discrete_action = [0, 0, 0]

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
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            # TODO: Retrieve state as list of input frames
            action_to_take = np.copy(cont_action)
            discrete_action_to_take = np.copy(discrete_action)
            s, r, done, info = env.step(action_to_take)
            total_reward += r
            action_id_to_take = 0
            try:
                action_id_to_take = available_actions_decoded[tuple(discrete_action_to_take.tolist())]

                # Only reproducible tuples are added to the experience buffer
                print(f"action c: {action_to_take}, id: {action_id_to_take}")
                # TODO: Save experience trajectory here
            except KeyError:
                # We don't support every combination of actions, it depends on action space complexity discretization
                print(f"action c: {action_to_take} not available in the discrete action space, skipping...")

            steps += 1
            isopen = env.render()
            if done or restart or isopen is False:
                break
    env.close()

if __name__ == "__main__":
    main()