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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 0.05
eps_0 = 1.0
eps_min = 0.0
eps_decay = 0.99
gamma = 0.9
episode_count = 700
batch_size = 50

elasticity = 0.1
workers = 3
commute_t = 10


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

    def __init__(self, action_space, model, eps_0, eps_min, eps_decay=1.0, N=1):
        self.action_space = action_space
        self.model = model
        # linear
        # self.eps = eps_0 - (eps_0 - eps_min) / N * np.arange(0, N)
        # exponential
        self.eps = np.full(N, eps_0) * np.full(N, eps_decay) ** np.arange(0, N)
        self.eps = np.where(self.eps < eps_min, eps_min, self.eps)

    def get_action(self, state, episode):
        state = to_tensor(state)
        with torch.no_grad():
            if np.random.uniform() < self.eps[episode]:
                return self.action_space.sample()
            else:
                return self.model(state).argmax().item()


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
    memory = ReplayMemory()
    mse_loss = nn.MSELoss()

    master_model = DQN().to(device)
    models = [DQN().to(device) for i in range(0, workers)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=learning_rate) for model in models]

    weights = sum(p.numel() for p in master_model.parameters())
    print(f'{weights} weights, model: {master_model}')
    print(f'Using {device} device: {device_name}')

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory='./tmp_easgd', force=True)
    agent = Agent(env.action_space, master_model, eps_0, eps_min, eps_decay, episode_count)
    episode_durations = []

    for episode in tqdm(range(episode_count)):
        state = env.reset()

        for t in count():
            # env.render()
            # actions are sampled from the master net
            action = agent.get_action(state, episode)
            next_state, reward, final, _ = env.step(action)
            memory.push(state, action, next_state, reward, int(not final))
            state = next_state

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                non_final = to_tensor(batch.non_final)

                # gradient update for N workers with the master net as regularisation
                master_w = [w.flatten() for w in master_model.state_dict().values()]
                master_w = torch.cat(master_w)
                for i, model in enumerate(models):
                    optimizer = optimizers[i]

                    # q_update = r,  for final s'
                    #            r + gamma * max_a Q(s', :), otherwise
                    indices = torch.stack((actions, actions))
                    q_values = model(states).t().gather(0, indices)[0]
                    q_update = rewards + non_final * gamma * master_model(next_states).amax(dim=1)

                    # only trainable
                    # model_parameters = filter(lambda param: param.requires_grad, model.parameters())
                    worker_w = [w.flatten() for w in model.state_dict().values()]
                    worker_w = torch.cat(worker_w)
                    loss = mse_loss(q_values, q_update) + elasticity * mse_loss(worker_w, master_w)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    # for param in model.parameters():
                    #     param.grad.data.clamp_(-1, 1)
                    optimizer.step()

                # master update - moving toward mean of workers
                if t % commute_t == 0:
                    # print(master_model.state_dict()['model.0.weight'][0])

                    for key in master_model.state_dict().keys():
                        for i, w in enumerate(master_model.state_dict()[key]):
                            workers_sum = 0
                            for model in models:
                                workers_sum += model.state_dict()[key][i]
                            # w = (1 - learning_rate * elasticity * workers) * w + learning_rate * elasticity * workers_sum
                            w.multiply_(1 - learning_rate * elasticity * workers)
                            w.add_(learning_rate * elasticity * workers_sum)

            if final:
                episode_durations.append(t + 1)
                break

    # Close the env and write monitor result to disk
    env.close()
    title = f'{weights} weights, workers: {workers}, elasticity: {elasticity}, tau: {commute_t}, ' \
            f'batch: {batch_size}, lr: {learning_rate}, gamma: {gamma}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp_easgd/{time}_training_qnn_easgd.png'
    plot_results(episode_durations, agent.eps, title, info, filename)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
