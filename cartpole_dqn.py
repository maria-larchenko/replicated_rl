from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from tqdm import tqdm
from torch import float32, int64

from drawing import plot_result_frames

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 0.02
eps_0 = 1.0
eps_min = 0.1
eps_decay = 0.0
N = 1000
gamma = 0.99
max_frames = 10_000
batch_size = 32
update_frequency = 32
polyak_coef = 0.1
clamp = 1.0


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, eps_0, eps_min, eps_decay=0.0, N=1):
        self.action_space = action_space
        self.model = model
        # linear
        self.eps = eps_0 - (eps_0 - eps_min) / N * np.arange(0, N)
        # exponential
        if eps_decay != 0.0:
            self.eps = np.full(N, eps_0) * np.full(N, eps_decay) ** np.arange(0, N)
        self.eps = np.where(self.eps < eps_min, eps_min, self.eps)

    def get_action(self, state, frame):
        state = to_tensor(state)
        with torch.no_grad():
            n = frame if frame < N-1 else N-1
            if np.random.uniform() < self.eps[n]:
                return self.action_space.sample()
            else:
                return self.model(state).argmax().item()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final'))


class ReplayMemory:

    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch, see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


def main():
    model = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(model.state_dict())

    memory = ReplayMemory()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    weights = sum(p.numel() for p in model.parameters())
    print(f'{weights} weights, model: {model}')
    print(f'Using {device} device: {device_name}')

    env = gym.make('CartPole-v0')
    agent = Agent(env.action_space, model, eps_0, eps_min, eps_decay, N)
    score = np.zeros(max_frames)
    epsilon = np.zeros(max_frames)
    learning_rates = np.zeros(max_frames)

    final = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state = env.reset()

    for t in tqdm(range(max_frames)):
        if final:
            prev_score = current_score
            current_score = 0
            episodes += 1
            state = env.reset()
        action = agent.get_action(state, t)
        next_state, reward, final, _ = env.step(action)
        memory.push(state, action, next_state, reward, final)
        state = next_state

        current_score += 1
        score[t] = prev_score
        epsilon[t] = agent.eps[t if t < N-1 else N-1]  # test
        learning_rates[t] = optimizer.param_groups[0]['lr']

        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states = to_tensor(batch.state)
            actions = to_tensor(batch.action, dtype=int64)
            rewards = to_tensor(batch.reward)
            next_states = to_tensor(batch.next_state)
            finals = to_tensor(batch.final)

            # q_update = r,  for final s'
            #            r + gamma * max_a Q(s', :), otherwise
            indices = torch.stack((actions, actions))
            q_values = model(states).t().gather(0, indices)[0]
            with torch.no_grad():
                new_q_values = target(next_states)
            q_update = rewards + (1.0 - finals) * gamma * new_q_values.max(dim=1).values
            # q_update = rewards + (1.0 - finals) * gamma * new_q_values.amax(dim=1)
            loss = mse_loss(q_update, q_values)

            optimizer.zero_grad()
            loss.backward()
            if clamp:
                for param_grad in model.parameters():
                    param_grad.data.clamp_(-clamp, clamp)
            optimizer.step()

            if t % update_frequency == 0:
                with torch.no_grad():
                    for model_param, target_param in zip(model.parameters(), target.parameters()):
                        new_param = polyak_coef * model_param + (1 - polyak_coef) * target_param
                        target_param.copy_(new_param)

    env.close()
    print(f'episodes: {episodes}')
    title = f'{weights} weights, batch: {batch_size}, lr: {learning_rate}, gamma: {gamma}, clamp: {clamp}: polyak: {polyak_coef}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp/{time}_training_qnn.png'
    plot_result_frames([score], epsilon, title, info, filename, learning_rate=learning_rates)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
