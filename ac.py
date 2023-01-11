from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from tqdm import tqdm
from torch import float32, int64

from classes.Agents import AcAgent
from classes.Memory import ReplayMemory
from classes.Models import ValueNet, PolicyNet
from drawing import plot_result_frames
from multiprocessing import Manager, Pool

seed = 333  # np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #
# env
env_name = 'LunarLander-v2'  # LunarLander-v2 CartPole-v1
env_actions = 4
env_state_dim = 8
# params
lr = 0.001
hidden = 256
gamma = 0.99
max_frames = 50_000
avg_frames = 1000
batch_size = 32
processes = 5
agents = 3


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


def print_progress(agent_number, message):
    if agent_number == 0:
        print(message)


def main(agent_number):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    value_net = ValueNet(env_state_dim, hidden).to(device)
    policy_net = PolicyNet(env_state_dim, hidden, env_actions).to(device)

    memory = ReplayMemory()
    env = gym.make(env_name)
    agent = AcAgent(env.action_space, policy_net, local_seed)

    score = np.zeros(max_frames)
    final = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state = env.reset(seed=local_seed)[0]

    # for t in tqdm(range(max_frames)):
    for t in range(max_frames):
        if final:
            prev_score = current_score
            current_score = 0
            episodes += 1
            state = env.reset()[0]
        action = agent.get_action(state)
        next_state, reward, final, _, _ = env.step(action)
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

            # 1-step Actor-Critic
            # ----------- gradient step 1: CRITIC
            advance = rewards + (1 - finals) * gamma * value_net(next_states) - value_net(states)
            critic_loss = (1/2 * advance ** 2).mean()
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
            if t % 100 == 0:
                print_progress(agent_number, f"score: {prev_score}, C-loss: {critic_loss}, A-loss: {actor_loss}")

    env.close()
    return episodes, score


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING AC, processes: {processes}")
    print(f"SEED: {seed}")

    with Manager() as manager, Pool(processes=processes) as pool:
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent,) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score, eps), (episodes, score, eps)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'{env_name} AC {agents} agents\n' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'softmax, ' \
                f'seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/ac/{timestamp}_ac_{agents}.png'
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename, lr=lr, mean_window=avg_frames)


