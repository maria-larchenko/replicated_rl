import os
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from gym import wrappers
from torch import from_numpy, as_tensor, float32, int64

from drawing import plot_results

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = '-' #torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 0.07
eps_0 = 1.0
eps_min = 0.0
eps_decay = 0.99
gamma = 0.9
episode_count = 700
batch_size = 50

elasticity = 0.05
N = 3
commute_t = None


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 30),
            nn.ReLU(),
            nn.Linear(30, 2),
        )

    # x could be a single state or a batch
    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, eps):
        self.action_space = action_space
        self.model = model
        self.eps = eps

    def get_action(self, state):
        model = self.model
        state = to_tensor(state)
        with torch.no_grad():
            if np.random.uniform() < self.eps:
                return self.action_space.sample()
            else:
                return model(state).argmax().item()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'non_final'))


class ReplayMemory:
    """From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch,
        # see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


def to_tensor(x, dtype=float32):
    return as_tensor(x, dtype=dtype).to(device)


def main():
    mse_loss = nn.MSELoss()
    memory = [ReplayMemory() for _ in range(0, N)]

    master = DQN().to(device)
    models = [DQN().to(device) for _ in range(0, N)]
    tmp = DQN().to(device)
    # # AESGD init:
    # for model in models:
    #     model.load_state_dict(master.state_dict())

    # rand init (RSGD init?):
    master.load_state_dict(models[0].state_dict())
    with torch.no_grad():
        for i in range(1, N):
            for param, master_param in zip(models[i].parameters(), master.parameters()):
                new_master_param = master_param + param
                master_param.copy_(new_master_param)
        for master_param in master.parameters():
            master_param.divide_(N)

    weights = sum(p.numel() for p in master.parameters())
    print(f'{weights} weights, model: {master}')
    print(f'Using {device} device: {device_name}')

    environments = [gym.make('CartPole-v0') for _ in range(0, N)]
    agents = [Agent(env.action_space, model, eps_0) for env, model in zip(environments, models)]
    agent_states = [env.reset() for env in environments]
    episode_durations = [[] for _ in range(0, N)]
    episode_counter = [0 for _ in range(0, N)]
    epsilon = []

    for t in tqdm(count()):
        if len(episode_durations[0]) > episode_count:
            break

        for i in range(0, N):
            episode_counter[i] += 1
            agent = agents[i]
            env = environments[i]
            state = agent_states[i]
            model = models[i]

            action = agent.get_action(state)
            next_state, reward, final, _ = env.step(action)
            memory[i].push(state, action, next_state, reward, int(not final))
            agent_states[i] = next_state

            if len(memory[i]) > batch_size:
                batch = memory[i].sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                non_final = to_tensor(batch.non_final)

                tmp.load_state_dict(model.state_dict())

                # q_update = r,  for final s'
                #            r + gamma * max_a Q(s', :), otherwise
                indices = torch.stack((actions, actions))
                q_values = model(states).t().gather(0, indices)[0]
                # if commute_t is None:  #or rand init
                q_update = rewards + non_final * gamma * model(next_states).amax(dim=1)
                # else:
                #     q_update = rewards + non_final * gamma * master(next_states).amax(dim=1)

                loss = mse_loss(q_values, q_update)

                grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
                with torch.no_grad():
                    for param, param_grad in zip(model.parameters(), grad):
                        new_param = param - learning_rate * param_grad
                        param.copy_(new_param)

                if commute_t is not None and t % commute_t == 0:
                    with torch.no_grad():
                        for param, tmp_param, master_param in zip(model.parameters(), tmp.parameters(),
                                                                  master.parameters()):
                            new_param = param - elasticity * (tmp_param - master_param)
                            new_master_param = master_param + elasticity * (tmp_param - master_param)

                            param.copy_(new_param)
                            master_param.copy_(new_master_param)
            if final:
                episode_durations[i].append(episode_counter[i])
                episode_counter[i] = 0
                agent_states[i] = env.reset()
                agent.eps = agent.eps * eps_decay if agent.eps > eps_min else agent.eps
                if i == 0:
                    epsilon.append(agent.eps)

    # Close the env and write monitor result to disk
    [env.close() for env in environments]
    title = f'Distributed Asynch AESGD (rand init)\n' \
            f'agents: {N}, commute: {commute_t}, elasticity: {elasticity}, ' \
            f'lr: {learning_rate}, gamma: {gamma}, batch: {batch_size}, 2l {weights}w'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp_easgd_distributed_asynch/{time}_training_dist_asynch rand.png'
    plot_results(episode_durations, epsilon, title, info, filename)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
