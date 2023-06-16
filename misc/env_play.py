import gym
import numpy as np
from gym.utils.play import play

env = gym.make('MountainCarContinuous-v0')
env = gym.make("BipedalWalker-v3", )
env = gym.make('CartPole-v1')
env = gym.make("LunarLander-v2", render_mode="human")


def get_dim(env_space):
    if isinstance(env_space, gym.spaces.Discrete):
        return env_space.n
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape[0]


print(get_dim(env.action_space))
print(get_dim(env.observation_space))

# play(gym.make('LunarLander-v2', render_mode='rgb_array'), keys_to_action={"w": 0,
#                                                                           "a": 1,
#                                                                           "s": 2,
#                                                                           "d": 3, })
