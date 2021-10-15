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

seed = 75  # np.random.randint(low=0, high=2**10)  # 7, 8
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

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
clamp = 0.5
units = 64


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, units),
            nn.SELU(),
            nn.Linear(units, 2),
        )
        self.train()        # <----- SLM

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
        with torch.no_grad():    # for efficiency, only calc grad in algorithm.train
            n = frame if frame < N-1 else N-1
            if np.random.uniform() < self.eps[n]:
                return random.sample([0, 1], 1)[0]
                # return self.action_space.sample()
                # return random.sample(self.action_space, 1)
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
    return torch.as_tensor(x, dtype=dtype).to(device)        # initial
    # return torch.from_numpy(x.astype(np.float32)).to(device)   # <----- SLM


def main():
    model = DQN().to(device)
    target = DQN().to(device)
    # target.load_state_dict(model.state_dict())   # from SLM debug: not the same initial weights (for polyak update)

    memory = ReplayMemory()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'{sum(p.numel() for p in model.parameters())} weights, model: {model}')
    print(f'Using {device} device: {device_name}')

    env = gym.make('CartPole-v0')
    env.seed(seed)

    agent = Agent(env.action_space, model, eps_0, eps_min, eps_decay, N)
    score = np.zeros(max_frames)
    epsilon = np.zeros(max_frames)
    learning_rates = np.zeros(max_frames)
    weights = []

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
        # memory.push(state, action, next_state, reward, final)
        # SLM converts state into float16 for some reason
        slm = '_float16'
        memory.push(state.astype(np.float16), action, next_state.astype(np.float16), reward, final)
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

            # ---------------- initial variant:
            # q_update = r,  for final s'
            #            r + gamma * max_a Q(s', :), otherwise
            slm = ''
            indices = torch.stack((actions, actions))
            q_values = model(states).t().gather(0, indices)[0]
            with torch.no_grad():
                new_q_values = target(next_states)
            q_update = rewards + (1.0 - finals) * gamma * new_q_values.max(dim=1).values
            loss = mse_loss(q_update, q_values)
            # ---------------- SLM update:
            # slm = '_slm'
            # q_preds = model(states)
            # act_q_preds = q_preds.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # Q(S_t, A_t)
            # with torch.no_grad():
            #     # Use online_net to select actions in next state
            #     online_next_q_preds = target(next_states)                                        # Q'(S_{t+1}, :)
            #     # Use eval_net to calculate next_q_preds for actions chosen by online_net
            #     next_q_preds = target(next_states)                                               # Q'(S_{t+1}, :)
            # online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)          # argmax_a Q'(S_{t+1}, a)
            # max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)     # Q'(S_{t+1}, argmax_a Q'(S_{t+1}, a)) = max_a Q'(S_{t+1}, a)
            # max_q_targets = rewards + gamma * (1 - finals) * max_next_q_preds
            # loss = mse_loss(act_q_preds, max_q_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % update_frequency == 0:
                # ---------------- initial variant:
                with torch.no_grad():
                    for model_param, target_param in zip(model.parameters(), target.parameters()):
                        new_param = polyak_coef * model_param + (1 - polyak_coef) * target_param
                        target_param.copy_(new_param)
                # ---------------- SLM update:
                # for src_param, tar_param in zip(model.parameters(), target.parameters()):
                #     tar_param.data.copy_(polyak_coef * src_param.data + (1.0 - polyak_coef) * tar_param.data)
            # weights.append(np.array(model.state_dict()['model.2.bias'].cpu().data))

    env.close()
    print(f'episodes: {episodes}')
    title = f'hidden units: {units}, batch: {batch_size}, lr: {learning_rate}, gamma: {gamma}, clamp: {clamp}: ' \
            f'polyak: {polyak_coef}, seed: {seed}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp/{time}_training_qnn{slm}.png'
    plot_result_frames([score], epsilon, title, info, filename, learning_rate=learning_rates)
    # np.savetxt(f'weights{slm}.txt', weights)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
