import gym_space_racer
import gym

env = gym.make('gym_space_racer:rocket-racer-v0')

while True:
    env.reset()
    env.render()
    input()
