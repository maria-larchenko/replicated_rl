import gym
import numpy as np
from gym.utils.play import play

env = gym.make('CartPole-v1')
env = gym.make('MountainCarContinuous-v0')
env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("BipedalWalker-v3", hardcore=True)



play(gym.make('LunarLander-v2', render_mode='rgb_array'), keys_to_action={"w": 0,
                                                                          "a": 1,
                                                                          "s": 2,
                                                                          "d": 3, })
