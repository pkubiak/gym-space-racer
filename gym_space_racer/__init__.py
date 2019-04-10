from gym.envs.registration import register

register(
    id='rocket-racer-v0',
    entry_point='gym_space_racer.envs:RocketRacerEnv',
)
