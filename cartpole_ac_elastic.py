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
from multiprocessing import Manager, Pool

seed = 333  # np.random.randint(10_000)

device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #
processes = 5

lr = 0.01
hidden = 64
gamma = 0.99
max_frames = 10_000
avg_frames = 1000
batch_size = 32

agents = 5
update_frequency = 5  # 32
elasticity = 0.05
draw_master = False


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


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
        return self.model(x).reshape(batch_size)


class PolicyNet(nn.Module):
    def __init__(self):
        self.hidden = hidden
        super(PolicyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, 2),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)


class Agent:

    def __init__(self, action_space, model, seed=None):
        self.action_space = np.arange(0, action_space.n)
        self.model = model
        self._rng = np.random.default_rng(seed)

    def get_action(self, state):
        state = to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        with torch.no_grad():
            return self._rng.choice(self.action_space, p=prob)


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


def main(agent_number, global_value_net, global_policy_net, update_lock):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    value_net = ValueNet().to(device)
    policy_net = PolicyNet().to(device)

    tmp_value_net = ValueNet().to(device)
    tmp_policy_net = PolicyNet().to(device)

    if elasticity > 0:
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())

    memory = ReplayMemory()
    env = gym.make('CartPole-v0')
    env.seed(local_seed)
    agent = Agent(env.action_space, policy_net, local_seed)

    score = np.zeros(max_frames)
    final = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state = env.reset()

    for t in range(max_frames):
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

        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states = to_tensor(batch.state)
            actions = to_tensor(batch.action, dtype=int64)
            rewards = to_tensor(batch.reward)
            next_states = to_tensor(batch.next_state)
            finals = to_tensor(batch.final)

            if t % update_frequency == 0 and elasticity > 0:
                tmp_value_net.load_state_dict(value_net.state_dict())
                tmp_policy_net.load_state_dict(policy_net.state_dict())

            # 1-step Actor-Critic
            # ----------- gradient step 1: CRITIC
            advance = rewards + (1 - finals) * gamma * value_net(next_states) - value_net(states)
            critic_loss = (1 / 2 * advance ** 2).mean()
            critic_grad = torch.autograd.grad(critic_loss, value_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(value_net.parameters(), critic_grad):
                    param.copy_(param - lr * param_grad)
            # ----------- gradient step 2: ACTOR
            indices = torch.stack((actions, actions))
            act_prob = policy_net(states).T.gather(0, indices)[0]
            actor_loss = (- advance * torch.log(act_prob)).mean()
            actor_grad = torch.autograd.grad(actor_loss, policy_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(policy_net.parameters(), actor_grad):
                    param.copy_(param - lr * param_grad)

            if t % update_frequency == 0 and elasticity > 0:  # ---------------- elastic update:
                with update_lock:
                    with torch.no_grad():
                        for param, tmp_param, master_param in zip(value_net.parameters(), tmp_value_net.parameters(),
                                                                  global_value_net.parameters()):
                            param.copy_(param - elasticity * (tmp_param - master_param))
                            master_param.copy_(master_param + elasticity * (tmp_param - master_param))
                        for param, tmp_param, master_param in zip(policy_net.parameters(), tmp_policy_net.parameters(),
                                                                  global_policy_net.parameters()):
                            param.copy_(param - elasticity * (tmp_param - master_param))
                            master_param.copy_(master_param + elasticity * (tmp_param - master_param))

    env.close()
    return episodes, score


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING AC, processes: {processes}")
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
        pool_args = [(agent, global_value_net, global_policy_net, update_lock) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score), (episodes, score)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'AC Elastic {agents} agents (asynch)\n' \
                f'update_frequency: {update_frequency}, ' \
                f'elasticity: {elasticity}, \n' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'softmax, ' \
                f'seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/tmp_ac/{timestamp}_ac_elastic_{agents}.png'
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename, lr=lr, mean_window=avg_frames)


