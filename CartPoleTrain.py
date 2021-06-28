import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from itertools import count
from gym import wrappers
from torch import from_numpy, as_tensor, float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.0005
eps_0 = 0.9
eps_N = 0.01
gamma = 0.999
episode_count = 600


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 2),
        )

    # x could be a single state or a batch
    def forward(self, x):
        return self.model(x)


class Agent(object):

    def __init__(self, action_space, model, eps_0, esp_N, N):
        self.action_space = action_space
        self.model = model
        n = np.arange(0, N)
        self.eps = eps_0 - (eps_0 - eps_N) / N * n

    def get_action(self, state, episode):
        with torch.no_grad():
            q_values = self.model(state)
            if np.random.uniform() < self.eps[episode]:
                return self.action_space.sample(), q_values
            else:
                action = q_values.argmax()
                return action.item(), q_values


def to_tensor(x):
    return as_tensor(x, dtype=float32).to(device)


def main():

    model = DQN().to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Using {device} device: {torch.cuda.get_device_name(device=device)}')
    print(model)

    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory='./tmp', force=True)

    agent = Agent(env.action_space, model, eps_0, eps_N, episode_count)
    episode_durations = []

    for i in range(episode_count):
        state = to_tensor(env.reset())
        for t in count():
            # env.render()
            action, q_values = agent.get_action(state, i)
            next_state, reward, done, _ = env.step(action)

            # Compute prediction error
            next_state, reward = to_tensor(next_state), to_tensor(reward)
            q_update = reward + gamma * model(next_state).max()
            q_update = q_update.broadcast_to(2)
            loss = mse_loss(q_values, q_update)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                episode_durations.append(t + 1)
                break

    # Close the env and write monitor result info to disk
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
