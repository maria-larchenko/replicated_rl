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

learning_rate = 0.01
eps_0 = 1.0
eps_min = 0.0
eps_decay = 0.99
gamma = 0.9
episode_count = 1300
batch_size = 50

elasticity = 0.5
N = 3
commute_t = 1


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
    memory = ReplayMemory()
    mse_loss = nn.MSELoss()

    master_model = DQN().to(device)
    models = [DQN().to(device) for _ in range(0, N)]
    for model in models:
        model.load_state_dict(master_model.state_dict())

    weights = sum(p.numel() for p in master_model.parameters())
    print(f'{weights} weights, model: {master_model}')
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
            memory.push(state, action, next_state, reward, int(not final))
            agent_states[i] = next_state

            # gradient update for N workers with the master net as regularisation
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                non_final = to_tensor(batch.non_final)
                # master_params = filter(lambda param: param.requires_grad, master_model.parameters())
                # master_w = torch.cat([w.flatten() for w in master_params])

                # q_update = r,  for final s'
                #            r + gamma * max_a Q(s', :), otherwise
                indices = torch.stack((actions, actions))
                q_values = model(states).t().gather(0, indices)[0]
                q_update = rewards + non_final * gamma * master_model(next_states).amax(dim=1)

                loss = mse_loss(q_values, q_update)

                # Alexander Lobashev:
                grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
                with torch.no_grad():
                    for param, master_param, param_grad in zip(model.parameters(), master_model.parameters(), grad):
                        new_param = param - learning_rate * (param_grad + elasticity * (param - master_param))
                        param.copy_(new_param)
            if final:
                episode_durations[i].append(episode_counter[i])
                episode_counter[i] = 0
                agent_states[i] = env.reset()
                agent.eps = agent.eps * eps_decay if agent.eps > eps_min else agent.eps
                if i == 0:
                    epsilon.append(agent.eps)

        # master update - moving toward mean of workers
        if t % commute_t == 0:
            # print(master_model.state_dict()['model.0.weight'][0])

            with torch.no_grad():
                # i - model, j - param num
                params = [[p for p in model.parameters()] for model in models]
                average_params = []
                for j in range(0, len(params[0])):
                    avg = params[0][j]
                    for i in range(1, N):
                        avg += params[i][j] / N
                    average_params.append(avg)

                for average_param, master_param in zip(average_params, master_model.parameters()):
                    new_master_param = master_param * (1 - elasticity) + average_param * elasticity
                    master_param.copy_(new_master_param)


    # Close the env and write monitor result to disk
    [env.close() for env in environments]
    title = f'{weights} weights, agents: {N}, elasticity: {elasticity}, tau: {commute_t}, ' \
            f'batch: {batch_size}, lr: {learning_rate}, gamma: {gamma}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp_easgd_multiagent/{time}_training_multiagent.png'
    plot_results(episode_durations, epsilon, title, info, filename)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
