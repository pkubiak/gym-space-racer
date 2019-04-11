import gym_space_racer
import gym

env1 = gym.make('gym_space_racer:rocket-racer-v0')
env1.seed(123)
env1.spaceship.color = (0,1,0)
env1.reset()

# env2 = gym.make('gym_space_racer:rocket-racer-v0')
# env2.seed(123)
# env1.spaceship.color = (0,0,1)
# env2.reset()

done1 = done2 = False
import random

while True:
    if not done1:
        _, _, done1, _ = env1.step(random.choice([0,1,2,3,4]))
        env1.render()
    else:
        done1 = False
        env1.reset()
    #
    # if not done2:
    #     _, _, done2, _ = env2.step(random.choice([0,1,2,3,4]))
    #     env2.render()

input("DONE")
env1.close()
# env2.close()
