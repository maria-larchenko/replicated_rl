import random
from collections import namedtuple, deque
from datetime import datetime
from multiprocessing import Manager, Pool

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import float32

from drawing import plot_result_frames

# Seed
seed = 333  # np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

# params
lr = 0.01
hidden = 64
capacity = 200
gamma = 0.99
max_frames = 10_000
avg_frames = 1000
processes = 8
agents = 5


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


def print_progress(agent_number, message):
    if agent_number == 0:
        print(message)


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
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
        super(PolicyNet, self).__init__()
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final'))


class EpisodicMemory:

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory = deque([], maxlen=self.capacity)

    def pop(self):
        return self.memory.pop()

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, action_space, model, seed=None):
        self.action_space = np.arange(0, action_space.n)
        self._rng = np.random.default_rng(seed)
        self.model = model

    def get_action(self, state):
        state = to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        with torch.no_grad():
            return self._rng.choice(self.action_space, p=prob)


def main(number, global_value_net, global_policy_net, update_lock):
    local_seed = number + seed
    # print_progress(number, f"agent: {number}, local seed: {local_seed}")
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    value_net = ValueNet().to(device)
    policy_net = PolicyNet().to(device)

    memory = EpisodicMemory(capacity=capacity)
    env = gym.make('CartPole-v0')
    env.seed(local_seed)
    agent = Agent(env.action_space, policy_net, local_seed)
    score = np.zeros(max_frames)

    current_score = 0
    prev_score = 0
    episodes = 0
    T = 0

    while T < max_frames:
        # if 0 < episodes and episodes % 10 == 0:
        #     actions_str = ''
        #     for a in actions:
        #         actions_str += str(a)
        #     print(f"agent: {number} episode {episodes}, policy_net: {policy_net.state_dict()['model.4.bias']}\n")
        #     print(f"agent: {number} episode {episodes}, actions: {actions_str}\n")
        # if T > 0 and episodes < 4:
        #     actions_str = ''
        #     for a in actions:
        #         actions_str += str(a)
        #     print(f"agent: {number} episode {episodes}, policy_net: {policy_net.state_dict()['model.4.bias']}\n")
        #     print(f"agent: {number} episode {episodes}, actions: {actions_str}\n")
        #     if episodes == 3:
        #         env.close()
        #         return episodes, score

        value_grad = value_net.zeros_like()
        policy_grad = policy_net.zeros_like()
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())
        state = env.reset()
        final = False
        memory.clear()
        # actions = []

        while not final and T < max_frames:
            action = agent.get_action(state)
            next_state, reward, final, _ = env.step(action)
            memory.push(state, action, next_state, reward, final)
            state = next_state

            score[T] = prev_score
            current_score += 1
            T += 1
            # actions.append(action)

            if final:
                prev_score = current_score
                current_score = 0
                episodes += 1
        if T % 10 == 0:
            print_progress(number, f"score: {prev_score}")

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
        with update_lock:
            with torch.no_grad():
                for param, param_grad in zip(global_value_net.parameters(), value_grad):
                    param.copy_(param - lr / memory_length * param_grad)
                for param, param_grad in zip(global_policy_net.parameters(), policy_grad):
                    param.copy_(param - lr / memory_length * param_grad)

    env.close()
    return episodes, score


if __name__ == '__main__':
    print(f"MULTIPROCESSING A3C, processes: {processes}")
    print(f"SEED: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    global_value_net = ValueNet().to(device)
    global_policy_net = PolicyNet().to(device)

    with Manager() as manager, Pool(processes=processes) as pool:
        update_lock = manager.Lock()

        print(f"------------------------------------ started: {datetime.now().strftime('%H-%M-%S')}")
        pool_args = [(agent, global_value_net, global_policy_net, update_lock)
                     for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score), (episodes, score)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'A3C {agents} agents\n ' \
                f'hidden: {hidden}(selu), batch: episode, lr: {lr}, gamma: {gamma}, softmax, seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/tmp_a3c/{timestamp}_a3c_{agents}.png'

        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename, lr=None, mean_window=avg_frames)


