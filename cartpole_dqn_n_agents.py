import os
from datetime import datetime

import gym
from gym import wrappers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import random
from PIL import Image

from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
from torch import from_numpy, as_tensor, float32, int64

from drawing import plot_results

device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = '-' #torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 0.01
eps_0 = 1.0
eps_min = 0.01
eps_decay = 0.999
gamma = 0.999
episode_count = 4000
batch_size = 128

clamp = True
commute_t = 10
N = 3


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class Agent:

    def __init__(self, action_space, model, eps):
        self.action_space = action_space
        self.model = model
        self.eps = eps

    def get_action(self, state):
        model = self.model
        state = to_tensor(state)
        with torch.no_grad():
            if np.random.uniform() < self.eps:
                return self.action_space.sample()
            else:
                return model(state).argmax().item()


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'non_final'))


class ReplayMemory:
    """From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html"""

    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        # Transpose the batch of Transitions to Transition of batch,
        # see https://stackoverflow.com/a/19343/3343043.
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


def to_tensor(x, dtype=float32):
    return as_tensor(x, dtype=dtype).to(device)


resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])


def get_cart_location(screen_width, env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width, env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


def main():
    environments = []
    for i in range(0, N):
        env = gym.make('CartPole-v0')
        # env = wrappers.Monitor(env, directory=f'./tmp_dqn_n_agents/agent{i}', force=True)
        env.reset()
        environments.append(env)

    init_screen = get_screen(environments[0])
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = environments[0].action_space.n

    mse_loss = nn.MSELoss()
    memory = [ReplayMemory() for _ in range(0, N)]

    targets = [DQN(screen_height, screen_width, n_actions).to(device) for _ in range(0, N)]
    policies = [DQN(screen_height, screen_width, n_actions).to(device) for _ in range(0, N)]

    for target, policy in zip(targets, policies):
        target.load_state_dict(policy.state_dict())

    weights = sum(p.numel() for p in targets[0].parameters())
    print(f'{weights} weights, model: {targets[0]}')
    print(f'Using {device} device: {device_name}')

    agents = [Agent(env.action_space, policy, eps_0) for env, policy in zip(environments, policies)]
    episode_durations = [[] for _ in range(0, N)]
    episode_counter = [0 for _ in range(0, N)]
    epsilon = []
    agent_states = []
    agent_screens = []

    for env in environments:
        env.reset()

        last_screen = get_screen(env)
        current_screen = get_screen(env)
        state = current_screen - last_screen
        agent_states.append(state)
        agent_screens.append(current_screen)

    for t in tqdm(count()):
        if len(episode_durations[0]) > episode_count:
            break

        for i in range(0, N):
            episode_counter[i] += 1
            agent = agents[i]
            env = environments[i]
            state = agent_states[i]
            policy = policies[i]
            current_screen = agent_screens[i]
            target = targets[i]

            action = agent.get_action(state)
            _, reward, final, _ = env.step(action)

            last_screen = current_screen
            current_screen = get_screen(env)
            if not final:
                next_state = current_screen - last_screen
            else:
                next_state = None

            memory[i].push(state, action, next_state, reward, int(not final))

            agent_states[i] = next_state
            agent_screens[i] = current_screen

            if len(memory[i]) > batch_size:
                batch = memory[i].sample(batch_size)
                states = to_tensor(torch.cat(batch.state))
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                # next_states = to_tensor(torch.cat(batch.next_state))
                non_final = to_tensor(batch.non_final, dtype=torch.bool)

                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final] = target(non_final_next_states).amax(dim=1)

                # q_update = r,  for final s'
                #            r + gamma * max_a Q(s', :), otherwise
                indices = torch.stack((actions, actions))
                q_values = policy(states).t().gather(0, indices)[0]
                # if commute_t is None:  #or rand init
                q_update = rewards + non_final * gamma * next_state_values
                # else:
                #     q_update = rewards + non_final * gamma * master(next_states).amax(dim=1)

                loss = mse_loss(q_values, q_update)

                grad = torch.autograd.grad(outputs=loss, inputs=policy.parameters())
                with torch.no_grad():
                    for param, param_grad in zip(policy.parameters(), grad):
                        if clamp:
                            param_grad.data.clamp_(-1, 1)
                        new_param = param - learning_rate * param_grad
                        param.copy_(new_param)

                if commute_t is not None and t % commute_t == 0:
                    target.load_state_dict(policy.state_dict())

            if final:
                episode_durations[i].append(episode_counter[i])
                episode_counter[i] = 0

                env.reset()
                last_screen = get_screen(env)
                current_screen = get_screen(env)
                agent_states[i] = current_screen - last_screen
                agent_screens[i] = current_screen

                agent.eps = agent.eps * eps_decay if agent.eps > eps_min else agent.eps
                if i == 0:
                    epsilon.append(agent.eps)

    # Close the env and write monitor result to disk
    [env.close() for env in environments]
    title = f'DQN by AdamPaszke, SGD \n' \
            f'agents: {N}, commute: {commute_t}, ' \
            f'lr: {learning_rate}, gamma: {gamma}, batch: {batch_size}, clamp: {clamp}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp_dqn_n_agents/{time}_dqn.png'
    plot_results(episode_durations, epsilon, title, info, filename, cmap='BuGn')


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
