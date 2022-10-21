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

device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

lr_v = 0.01
lr_pi = 0.01
hidden = 256
capacity = 200
gamma = 0.99
max_frames = 50_000
avg_frames = 1000


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
        # return self.model(x).reshape(batch_size)
        return self.model(x)

    def zeros_like(self):
        zeros = []
        for p in self.parameters():
            zeros.append(torch.zeros_like(p))
        return zeros


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
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.model(x)

    def zeros_like(self):
        zeros = []
        for p in self.parameters():
            zeros.append(torch.zeros_like(p))
        return zeros


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


class EpisodicMemory:

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch, see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))

    def clear(self):
        self.memory = deque([], maxlen=self.capacity)

    def pop(self):
        return self.memory.pop()

    def __len__(self):
        return len(self.memory)


def main():
    print(f"SEED: {seed}")
    global_value_net = ValueNet().to(device)
    global_policy_net = PolicyNet().to(device)
    value_net = ValueNet().to(device)
    policy_net = PolicyNet().to(device)

    memory = EpisodicMemory(capacity=capacity)
    print(f'model: {policy_net}')
    print(f'Using {device} device: {device_name}')
    env = gym.make('CartPole-v0')
    env.seed(seed)
    agent = Agent(env.action_space, policy_net)
    score = np.zeros(max_frames)

    current_score = 0
    prev_score = 0
    episodes = 0
    T = 0

    while T < max_frames:
        value_grad = value_net.zeros_like()
        policy_grad = policy_net.zeros_like()
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())
        state = env.reset()
        final = False
        memory.clear()

        while not final and T < max_frames:
            action = agent.get_action(state)
            next_state, reward, final, _ = env.step(action)
            memory.push(state, action, next_state, reward, final)
            state = next_state

            score[T] = prev_score
            current_score += 1
            T += 1

            if final:
                prev_score = current_score
                current_score = 0
                episodes += 1

        if T % 10 == 0: print(f'T: {T}, score: {prev_score}')

        # Advantage Actor-Critic: A(s, a) = Q(s, a) - V(s) = r + V(s') - V(s)
        memory_length = len(memory)
        last_transition = memory.pop()
        retain_graph = not last_transition.final
        R = 0 if last_transition.final else value_net(to_tensor(last_transition.state))
        for transition in reversed(memory.memory):
            s = to_tensor(transition.state)
            a = transition.action
            r = to_tensor(transition.reward)
            R = r + gamma * R
            # ----------- accumulate gradients: ACTOR
            actor_loss = - torch.log(policy_net(s)[a]) * (R - value_net(s))
            actor_grad = torch.autograd.grad(actor_loss, policy_net.parameters())
            with torch.no_grad():
                for accumulated_grad, grad in zip(policy_grad, actor_grad):
                    accumulated_grad += grad / memory_length
            # ----------- accumulate gradients: CRITIC
            critic_loss = (R - value_net(s)) ** 2
            critic_grad = torch.autograd.grad(critic_loss, value_net.parameters(), retain_graph=retain_graph)
            with torch.no_grad():
                for accumulated_grad, grad in zip(value_grad, critic_grad):
                    accumulated_grad += grad / memory_length
        # ----------- Asynch Update global nets
        with torch.no_grad():
            for param, param_grad in zip(global_value_net.parameters(), value_grad):
                param.copy_(param - lr_v * param_grad)
            for param, param_grad in zip(global_policy_net.parameters(), policy_grad):
                param.copy_(param - lr_pi * param_grad)

    env.close()
    print(f'episodes: {episodes}')
    title = f'hidden: {hidden}, batch: episode, lr_v: {lr_v}, lr_pi: {lr_pi}, gamma: {gamma}, softmax, seed: {seed}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./output/tmp_ac/{time}_training.png'
    plot_result_frames([score], None, title, None, filename, lr=None, mean_window=avg_frames)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
