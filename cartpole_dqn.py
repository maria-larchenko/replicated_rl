from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from multiprocessing import Manager, Pool
from collections import namedtuple, deque
from tqdm import tqdm
from torch import float32, int64

from drawing import plot_results, plot_result_frames

seed = 1  # np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

processes = 8
agents = 4

lr = 0.02
hidden = 64
eps_0 = 1.0
eps_min = 0.1
eps_steps = 1000
eps_decay = 0.0
gamma = 0.99
max_frames = 10_000
avg_frames = 1_000
batch_size = 32
update_frequency = 32
polyak_coef = 0.1
clamp = 1.0


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden = hidden
        # self.model = nn.Sequential(
        #     nn.Linear(4, units),
        #     nn.SELU(),
        #     nn.Linear(units, 2),
        # )
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, 2),
        )
        self.train()

    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, eps_0, eps_min, eps_steps, eps_decay=0.0, seed=None):
        self.action_space = np.arange(0, action_space.n)
        self.model = model
        self.N = eps_steps
        # linear
        self.eps = eps_0 - (eps_0 - eps_min) / self.N * np.arange(0, self.N)
        # exponential
        if eps_decay != 0.0:
            self.eps = np.full(self.N, eps_0) * np.full(self.N, eps_decay) ** np.arange(0, self.N)
        self.eps = np.where(self.eps < eps_min, eps_min, self.eps)
        self._rng = np.random.default_rng(seed)

    def get_eps(self, frame):
        n = frame if frame < self.N-1 else self.N-1
        return self.eps[n]

    def get_action(self, state, frame):
        state = to_tensor(state)
        with torch.no_grad():    # for efficiency, only calc grad in algorithm.train
            if self._rng.uniform() < self.get_eps(frame):
                return self._rng.choice(self.action_space)
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


def main(agent_number):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = DQN().to(device)
    target = DQN().to(device)
    target.load_state_dict(model.state_dict())

    memory = ReplayMemory()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #
    # weights = sum(p.numel() for p in model.parameters())
    # print(f'{weights} weights, model: {model}')
    # print(f'Using {device} device: {device_name}')

    env = gym.make('CartPole-v0')
    env.seed(local_seed)
    agent = Agent(env.action_space, model, eps_0, eps_min, eps_steps, eps_decay, local_seed)
    score = np.zeros(max_frames)
    epsilon = np.zeros(max_frames)
    learning_rates = np.zeros(max_frames)

    final = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state = env.reset()

    # for t in tqdm(range(max_frames)):
    for t in range(max_frames):
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
        epsilon[t] = agent.get_eps(t)  # test
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
            # if clamp:                                     # there is a mistake here!
            #     for param_grad in model.parameters():
            #         param_grad.data.clamp_(-clamp, clamp)
            optimizer.step()

            if t % update_frequency == 0:
                with torch.no_grad():
                    for model_param, target_param in zip(model.parameters(), target.parameters()):
                        new_param = polyak_coef * model_param + (1 - polyak_coef) * target_param
                        target_param.copy_(new_param)

    env.close()
    return episodes, score, epsilon


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    # main()

    print(f"MULTIPROCESSING DQN, processes: {processes}")
    print(f"SEED: {seed}")

    with Manager() as manager, Pool(processes=processes) as pool:
        print(f"------------------------------------ started: {datetime.now().strftime('%H-%M-%S')}")
        pool_args = [(agent, ) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score, eps), (episodes, score, eps)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        epsilons = [result[2] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'DQN {agents} agents\n ' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'polyak: {polyak_coef}, ' \
                f'freq: {update_frequency}, ' \
                f'seed: {seed}'
        info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/tmp_dqn_elastic/{timestamp}_dqn_{agents}.png'
        plot_result_frames(scores, epsilon=epsilons[0], title=title, info=info,
                           filename=filename, lr=lr, mean_window=avg_frames)

