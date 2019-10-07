from gym.envs.registration import register

register(
    id='noisy-lander-v0',
    entry_point='noisy_lander.envs:NoisyLander',
)
