from datetime import datetime

import gym
import numpy as np
import torch
import random

from gym.wrappers import TimeLimit
from torch import float32, int64

from classes.Agents import AcAgent
from classes.Memory import ReplayMemory
from classes.Models import ValueNet, PolicyNet
from drawing import plot_result_frames
from multiprocessing import Manager, Pool

seed = np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

env_name = 'CartPole-v1'  # LunarLander-v2
env_actions = 2
env_state_dim = 4
save_model = False

lr = 0.002
hidden = 256
gamma = 0.99
max_frames = 50_000
max_episode_steps = 500
avg_frames = 1_000
batch_size = 64
clamp = 1e-08
processes = 3
agents = 3


def to_tensor(x, dtype=float32):
    return torch.as_tensor(np.array(x), dtype=dtype).to(device)


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

    memory = ReplayMemory(seed=local_seed)
    # env = TimeLimit(gym.make(env_name, render_mode='human'), max_episode_steps=max_episode_steps)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    agent = AcAgent(env.action_space, policy_net, local_seed)

    score = np.zeros(max_frames)
    final = False
    truncated = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state, info = env.reset(seed=local_seed)

    # for t in tqdm(range(max_frames)):
    for t in range(max_frames):
        if final or truncated:
            prev_score = current_score
            current_score = 0
            episodes += 1
            state, info = env.reset()
        action = agent.get_action(state)
        next_state, reward, final, truncated, info = env.step(action)
        memory.push(state, action, next_state, reward, final)
        state = next_state

        current_score += reward
        score[t] = prev_score

        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            states = to_tensor(batch.state)
            actions = to_tensor(batch.action, dtype=int64)
            rewards = to_tensor(batch.reward)
            next_states = to_tensor(batch.next_state)
            finals = to_tensor(batch.final)

            # ----------- gradient step 1: CRITIC
            advance = rewards + (1 - finals) * gamma * value_net(next_states).T - value_net(states).T
            critic_loss = (1/2 * advance ** 2).mean()
            critic_grad = torch.autograd.grad(critic_loss, value_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(value_net.parameters(), critic_grad):
                    if clamp:
                        param_grad.data.clamp_(-clamp, clamp)
                    param.copy_(param - lr * param_grad)
            # ----------- gradient step 2: ACTOR
            indices = torch.stack((actions, actions))
            act_prob = policy_net(states).T.gather(0, indices)[0]
            actor_loss = (- advance * torch.log(act_prob)).mean()
            actor_grad = torch.autograd.grad(actor_loss, policy_net.parameters())
            with torch.no_grad():
                for param, param_grad in zip(policy_net.parameters(), actor_grad):
                    if clamp:
                        param_grad.data.clamp_(-clamp, clamp)
                    param.copy_(param - lr * param_grad)
            if t % 100 == 0:
                print_progress(agent_number, f"{t}/{max_frames}| score: {current_score}, "
                                             f"critic_loss: {float(critic_loss)}, actor_loss: {float(actor_loss)}")
    env.close()
    return episodes, score, value_net, policy_net


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING AC, processes: {processes}")
    print(f"SEED: {seed}")

    with Manager() as manager, Pool(processes=processes) as pool:
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent,) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        value_nets = [result[2] for result in agent_results]
        policy_nets = [result[3] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'{env_name} {max_episode_steps} AC {agents} agents\n ' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'clamp: {clamp}' \
                f'seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/ac/{timestamp}_ac_{agents}'
        if save_model:
            torch.save(value_nets[0].state_dict(), filename + '_V.pth')
            torch.save(policy_nets[0].state_dict(), filename + '_pi.pth')
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename+'.png', lr=lr, mean_window=avg_frames)




