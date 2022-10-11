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

seed = 1  # np.random.randint(10_000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')  #
device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  # 'cpu'  #

lr_v = 0.01
lr_pi = 0.01
hidden = 254
gamma = 0.99
max_frames = 50_000
avg_frames = 1000
batch_size = 32


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x):
        return self.model(x).reshape(batch_size)


class PolicyNet(nn.Module):
    def __init__(self):
        self.hidden = hidden
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model):
        self.action_space = np.arange(0, action_space.n)
        self.model = model

    def get_action(self, state):
        state = to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        with torch.no_grad():
            return np.random.choice(self.action_space, p=prob)


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


def main():
    print(f"SEED: {seed}")

    value_net = ValueNet().to(device)
    policy_net = PolicyNet().to(device)

    memory = ReplayMemory()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    weights = sum(p.numel() for p in policy_net.parameters())
    print(f'{weights} weights, model: {policy_net}')
    print(f'Using {device} device: {device_name}')

    env = gym.make('CartPole-v0')
    env.seed(seed)
    agent = Agent(env.action_space, policy_net)
    score = np.zeros(max_frames)
    # epsilon = np.zeros(max_frames)

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
        action = agent.get_action(state)
        next_state, reward, final, _ = env.step(action)
        memory.push(state, action, next_state, reward, final)
        state = next_state

        current_score += 1
        score[t] = prev_score
        # epsilon[t] = agent.eps[t if t < N-1 else N-1]  # test
        # learning_rates[t] = optimizer.param_groups[0]['lr']

        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states = to_tensor(batch.state)
            actions = to_tensor(batch.action, dtype=int64)
            rewards = to_tensor(batch.reward)
            next_states = to_tensor(batch.next_state)
            finals = to_tensor(batch.final)

            # 1-step Actor-Critic
            # ----------- gradient step 1: CRITIC
            advance = rewards + (1 - finals) * gamma * value_net(next_states) - value_net(states)
            critic_loss = (1/2 * advance ** 2).mean()
            critic_grad = torch.autograd.grad(critic_loss, value_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(value_net.parameters(), critic_grad):
                    param.copy_(param - lr_v * param_grad)
            # ----------- gradient step 2: ACTOR
            indices = torch.stack((actions, actions))
            act_prob = policy_net(states).T.gather(0, indices)[0]
            actor_loss = (- advance * torch.log(act_prob)).mean()
            actor_grad = torch.autograd.grad(actor_loss, policy_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(policy_net.parameters(), actor_grad):
                    param.copy_(param - lr_pi * param_grad)

    env.close()
    print(f'episodes: {episodes}')
    title = f'hidden: {hidden}, batch: {batch_size}, lr_v: {lr_v}, lr_pi: {lr_pi}, gamma: {gamma}, softmax, seed: {seed}'
    # info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./output/tmp_ac/{time}_training.png'
    plot_result_frames([score], None, title, None, filename, lr=None, mean_window=avg_frames)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
