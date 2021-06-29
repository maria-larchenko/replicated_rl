import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from itertools import count
from gym import wrappers
from torch import from_numpy, as_tensor, float32, int64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0005
eps_0 = 1.0
eps_N = 0.1
gamma = 0.9
episode_count = 1000
batch_size = 10


class DQN(nn.Module):
    def __init__(self, hidden1=6, hidden2=4):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    # x could be a single state or a batch
    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, eps_0, eps_N, N):
        self.action_space = action_space
        self.model = model
        self.eps = eps_0 - (eps_0 - eps_N) / N * np.arange(0, N)

    def get_action(self, state, episode):
        state = to_tensor(state)
        with torch.no_grad():
            if np.random.uniform() < self.eps[episode]:
                return self.action_space.sample()
            else:
                action = self.model(state).argmax()
                return action.item()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'non_final'))


class ReplayMemory:
    """Taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

    def __init__(self, capacity=2000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch,
        # see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))


def to_tensor(x, dtype=float32):
    return as_tensor(x, dtype=dtype).to(device)


def main():
    model = DQN().to(device)
    memory = ReplayMemory()
    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Using {device} device: {torch.cuda.get_device_name(device=device)}')
    print(model)

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory='./tmp', force=True)

    agent = Agent(env.action_space, model, eps_0, eps_N, episode_count)
    episode_durations = []

    for i in range(episode_count):
        # state = to_tensor(env.reset())
        state = env.reset()
        for t in count():
            # env.render()
            action = agent.get_action(state, i)
            next_state, reward, final, _ = env.step(action)

            memory.push(state, action, next_state, reward, int(not final))

            state = next_state

            # Experience Replay
            if t > batch_size:
                batch = memory.sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                non_final = to_tensor(batch.non_final)

                q_values = model(states)
                q_values = q_values.gather(0, actions)
                q_update = reward + non_final * gamma * model(next_state).max()

            # next_state, reward = to_tensor(next_state), to_tensor(reward)
            # q_update = reward.broadcast_to(2).requires_grad_()
            # if not final:
            #     q_update = q_update + gamma * model(next_state).max()
            # loss = huber_loss(q_values, q_update)

            # Backpropagation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if final:
                episode_durations.append(t + 1)
                break

    # Close the env and write monitor result to disk
    env.close()
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(episode_durations)
    ax0.set_xlabel('Episode')
    ax0.set_ylabel('Duration')
    ax1.plot(agent.eps, color='r')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Epsilon')
    plt.tight_layout()
    plt.savefig('./tmp/training.png')
    plt.show()


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
