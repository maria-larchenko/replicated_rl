from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from collections import namedtuple, deque
from tqdm import tqdm
from torch import from_numpy, as_tensor, float32, int64

from drawing import plot_results, plot_result_frames

seed = 1  # np.random.randint(10_000)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

learning_rate = 0.01
eps_0 = 1.0
eps_min = 0.1
eps_decay = 0.0
eps_steps = 5000
gamma = 0.99
max_frames = 10_000
batch_size = 32
clamp = False
hidden = 64
mean_window = 1000

draw_master = False
N = 4
update_frequency = 10  # 32

# choose only one coef, another should be set to 0
elasticity = 0.05      # 0.1 |  2 x learning rate ? ~ 1 / agents
polyak_coef = 0        # 0.1


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
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

    # x could be a single state or a batch
    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, eps_0, eps_min, eps_steps, eps_decay=0.0,):
        self.action_space = action_space
        self.model = model
        self.N = eps_steps
        # linear
        self.eps = eps_0 - (eps_0 - eps_min) / self.N * np.arange(0, self.N)
        # exponential
        if eps_decay != 0.0:
            self.eps = np.full(self.N, eps_0) * np.full(self.N, eps_decay) ** np.arange(0, self.N)
        self.eps = np.where(self.eps < eps_min, eps_min, self.eps)

    def get_eps(self, frame):
        n = frame if frame < self.N-1 else self.N-1
        return self.eps[n]

    def get_action(self, state, frame):
        state = to_tensor(state)
        with torch.no_grad():    # for efficiency, only calc grad in algorithm.train
            if np.random.uniform() < self.get_eps(frame):
                return random.sample([0, 1], 1)[0]
                # return self.action_space.sample()
                # return random.sample(self.action_space, 1)
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
        # Transpose the batch of Transitions to Transition of batch, see https://stackoverflow.com/a/19343/3343043.
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

    if draw_master:
        models.append(master)

    tmp = DQN().to(device)

    # # AESGD init:
    # for model in models:
    #     model.load_state_dict(master.state_dict())

    # # rand init (RSGD init):
    # master.load_state_dict(models[0].state_dict())
    # with torch.no_grad():
    #     for i in range(1, N):
    #         for param, master_param in zip(models[i].parameters(), master.parameters()):
    #             new_master_param = master_param + param
    #             master_param.copy_(new_master_param)
    #     for master_param in master.parameters():
    #         master_param.divide_(N)

    print(f'{sum(p.numel() for p in master.parameters())} weights, model: {master}')
    print(f'Using {device} device: {device_name}')

    # number of agents is N+1 - the last model will be master model
    environments = [gym.make('CartPole-v0') for _ in range(0, len(models))]
    for env in environments:
        env.seed(seed)
    agents = [Agent(env.action_space, model, eps_0, eps_min, eps_steps) for env, model in zip(environments, models)]
    agent_states = [env.reset() for env in environments]
    finals = [False for _ in range(0, len(models))]
    current_scores = [0 for _ in range(0, len(models))]
    previous_scores = [0 for _ in range(0, len(models))]
    episode_counter = [0 for _ in range(0, len(models))]
    scores = np.zeros((len(models), max_frames))
    epsilon = []
    learning_rates = []

    for t in tqdm(range(max_frames)):

        for i in range(0, len(models)):
            agent = agents[i]
            env = environments[i]
            state = agent_states[i]
            model = models[i]
            final = finals[i]

            if final:
                previous_scores[i] = current_scores[i]
                current_scores[i] = 0
                episode_counter[i] += 1
                state = env.reset()

            action = agent.get_action(state, t)
            next_state, reward, final, _ = env.step(action)
            if i != N:      # <---- no memory for master model
                memory[i].push(state.astype(np.float16), action, next_state.astype(np.float16), reward, int(not final))
            agent_states[i] = next_state
            finals[i] = final

            current_scores[i] += 1
            scores[i, t] = previous_scores[i]
            if i == 0:
                epsilon.append(agent.get_eps(t))  # test
                learning_rates.append(learning_rate)

            if i != N and len(memory[i]) > batch_size:  # <---- do not run update for master model
                batch = memory[i].sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                non_finals = to_tensor(batch.non_final)

                if t % update_frequency == 0 and elasticity > 0:
                    tmp.load_state_dict(model.state_dict())

                indices = torch.stack((actions, actions))
                q_values = model(states).t().gather(0, indices)[0]
                with torch.no_grad():
                    new_q_values = master(next_states)
                q_update = rewards + non_finals * gamma * new_q_values.max(dim=1).values
                loss = mse_loss(q_update, q_values)

                grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
                with torch.no_grad():
                    if clamp:  # <------ probably a error, drops performance! set to false
                        for param_grad in model.parameters():
                            param_grad.data.clamp_(-clamp, clamp)
                    for param, param_grad in zip(model.parameters(), grad):
                        param.copy_(param - learning_rate * param_grad)

                if t % update_frequency == 0 and elasticity > 0:  # ---------------- elastic update:
                    with torch.no_grad():
                        for param, tmp_param, master_param in zip(model.parameters(), tmp.parameters(),
                                                                  master.parameters()):
                            param.copy_(param - elasticity * (tmp_param - master_param))
                            master_param.copy_(master_param + elasticity * (tmp_param - master_param))

                if t % update_frequency == 0 and polyak_coef > 0:  # ---------------- SLM polyak update:
                    with torch.no_grad():
                        for model_param, master_param in zip(model.parameters(), master.parameters()):
                            master_param.copy_(master_param + polyak_coef * (model_param - master_param))

    [env.close() for env in environments]
    type = 'elastic' if elasticity else 'polyak'
    time = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
    filename = f'./output/tmp_dqn_elastic/{time}_dqn_{type}.png'
    title = f'{f"DQN Elastic {N} agents" if elasticity else f"DQN Polyak {N} agents"}\n' \
            f'update_frequency: {update_frequency}, ' \
            f'elasticity: {elasticity}, ' \
            f'polyak_coef: {polyak_coef},\n ' \
            f'hidden: {hidden}(selu), ' \
            f'batch: {batch_size}, ' \
            f'lr: {learning_rate}, ' \
            f'gamma: {gamma}, ' \
            f'clamp: {clamp}, ' \
            f'seed: {seed}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'

    master_score = scores[N, :] if draw_master else None
    model_scores = scores[:-1] if draw_master else scores
    plot_result_frames(model_scores, epsilon, title, info, filename,
                       lr=learning_rates, mean_window=mean_window, master_score=master_score, txt=episode_counter)


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
