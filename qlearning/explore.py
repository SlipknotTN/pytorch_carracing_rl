import cv2
import gym

env = gym.make('CarRacing-v0')
"""
More details here: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
"""

print("State space description: image 96x96 RGB images uint8 format")
print(f"State space (env.observation_space): {env.observation_space}")

print("\nActions description: steer from -1.0 to +1.0, gas from 0.0 to +1.0, brake from 0.0 to +1.0")
print(f"Action space (env.action_space): {env.action_space}, "
      f"min: {env.action_space.low}, max: {env.action_space.high}")


print("Running a random episode")
state = env.reset()
score = 0
done = False
while not done:
    action = env.action_space.sample()
    env.render()
    state, reward, done, _ = env.step(action)
    # Show the actual state
    cv2.imshow("State", cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    score += reward

print('Final score:', score)
env.close()
