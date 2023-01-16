from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

from multiprocessing import Manager, Pool
from torch import float32, int64
from classes.Agents import DqnAgent
from classes.Memory import ReplayMemory
from classes.Models import DQN
from drawing import plot_results, plot_result_frames
from gym.wrappers import TimeLimit

seed = 1  # np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

env_name = 'LunarLander-v2' #'LunarLander-v2'  # LunarLander-v2 CartPole-v1
env_actions = 4
env_state_dim = 8

processes = 1
agents = 1
save_model = True

lr = 0.001
hidden = 512
eps_0 = 1.0
eps_min = 0.1
eps_steps = 40_000
eps_decay = 0.0
gamma = 0.99
max_frames = 50_000
max_episode_steps = 500
avg_frames = 1_000
batch_size = 128
update_frequency = 32
polyak_coef = 0.1
clamp = 1.0


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

    model = DQN(env_state_dim, hidden, env_actions).to(device)
    target = DQN(env_state_dim, hidden, env_actions).to(device)
    target.load_state_dict(model.state_dict())

    memory = ReplayMemory(seed=local_seed)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # weights = sum(p.numel() for p in model.parameters())
    # print(f'{weights} weights, model: {model}')
    # print(f'Using {device} device: {device_name}')

    # env = TimeLimit(gym.make(env_name, render_mode="human"), max_episode_steps=max_episode_steps)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)

    agent = DqnAgent(env.action_space, model, local_seed)
    agent.set_lin_greediness(eps_0, eps_min, eps_steps)
    agent.set_softmax_greediness()
    score = np.zeros(max_frames)
    epsilon = np.zeros(max_frames)
    learning_rates = np.zeros(max_frames)

    final = False
    truncated = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state, _ = env.reset(seed=local_seed)

    for t in range(max_frames):
        if final or truncated:
            prev_score = current_score
            current_score = 0
            episodes += 1
            state, _ = env.reset()
        action = agent.get_action(state)
        next_state, reward, final, truncated, _ = env.step(action)
        memory.push(state, action, next_state, reward, final)
        state = next_state

        current_score += reward
        score[t] = prev_score
        epsilon[t] = agent.get_greediness()
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
            loss = mse_loss(q_update, q_values)

            optimizer.zero_grad()
            loss.backward()
            # if clamp:
            #     for param_grad in model.parameters():
            #         param_grad.data.clamp_(-clamp, clamp)
            optimizer.step()

            if t % update_frequency == 0:
                with torch.no_grad():
                    for model_param, target_param in zip(model.parameters(), target.parameters()):
                        new_param = polyak_coef * model_param + (1 - polyak_coef) * target_param
                        target_param.copy_(new_param)
                print_progress(agent_number, f"{t}/{max_frames}| score: {current_score}, loss: {loss}")
    env.close()
    return episodes, score, epsilon, agent


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING DQN, processes: {processes}")
    print(f"SEED: {seed}")

    with Manager() as manager, Pool(processes=processes) as pool:
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent, ) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score, eps), (episodes, score, eps)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        epsilons = [result[2] for result in agent_results]
        agent_objs = [result[3] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'{env_name} {max_episode_steps} DQN {agents} agents\n ' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'polyak: {polyak_coef}, ' \
                f'freq: {update_frequency}, ' \
                f'seed: {seed}'
        info = agent_objs[0].get_eps_info()
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/dqn/{timestamp}_dqn_{agents}'
        if save_model:
            torch.save(agent_objs[0].model.state_dict(), filename+'.pth')
        plot_result_frames(scores, epsilon=epsilons[0], title=title, info=info,
                           filename=filename+'.png', lr=lr, mean_window=avg_frames)



