from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from torch.multiprocessing import Manager, Pool, set_start_method
from torch import float32
from classes.Agents import DqnAgent
from classes.Models import DQN
from drawing import plot_results, plot_result_frames
from gym.wrappers import TimeLimit

seed = np.random.randint(10_000)
env_name = 'Acrobot-v1'  # LunarLander-v2 LunarLander-v2 CartPole-v1 Acrobot-v1
saved_model = './output/dqn/2023.02.21 20-14-15_dqn_3.pth'

hidden = 512
max_frames = 300_000
max_episode_steps = 500 if env_name != "MountainCar-v0" else 200
avg_frames = 1000
temperature = 1
processes = 1
agents = 1


def to_tensor(x, dtype=float32):
    return torch.as_tensor(np.array(x), dtype=dtype).to(device)


def get_dim(env_space):
    if isinstance(env_space, gym.spaces.Discrete):
        return env_space.n
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape[0]


def main_dqn(agent_number, env_s, env_a, device, device_name):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = DQN(env_s, hidden, env_a).to(device)
    model.load_state_dict(torch.load(saved_model))
    weights = sum(p.numel() for p in model.parameters())
    print(f'{weights} weights, model: {model}')
    print(f'Using {device} device: {device_name}')
    env = TimeLimit(gym.make(env_name, render_mode="human"), max_episode_steps=max_episode_steps)
    agent = DqnAgent(env.action_space, model, local_seed, device)
    agent.set_softmax_greediness(temperature=1)

    final, truncated = False, False
    try:
        state, info = env.reset(seed=local_seed)
        for t in range(max_frames):
            if final or truncated:
                state, _ = env.reset()
            action = agent.get_action(state)
            next_state, reward, final, truncated, _ = env.step(action)
            state = next_state
    except ValueError as error:
        print(error)
    env.close()


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"SEED: {seed}")
    try:
        set_start_method('spawn')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'
    except RuntimeError:
        print("failed to use spawn for CUDA")
        device = torch.device('cpu')
        device_name = '-'
    env_s = get_dim(gym.make(env_name).observation_space)
    env_a = get_dim(gym.make(env_name).action_space)
    with Manager() as manager, Pool(processes=processes) as pool:
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        if saved_model.__contains__('dqn'):
            pool_args = [(agent, env_s, env_a, device, device_name) for agent in range(agents)]
            agent_results = pool.starmap(main_dqn, pool_args)
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")


