import random
from datetime import datetime
from multiprocessing import Manager, Pool

import gym
import numpy as np
import torch
import torch.nn as nn
from torch import float32

from classes.Agents import AcAgent
from classes.Memory import BackwardMemory
from classes.Models import ValueNet, PolicyNet
from drawing import plot_result_frames
from gym.wrappers import TimeLimit

seed = np.random.randint(10_000)
device = torch.device('cpu')  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  #

env_name = 'LunarLander-v2'  # LunarLander-v2 CartPole-v1
env_actions = 4
env_state_dim = 8
save_model = False

lr = 0.005
hidden = 256
gamma = 0.99
max_frames = 100_000
max_episode_steps = 500
avg_frames = 1_000
processes = 3
agents = 3


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


def print_progress(agent_number, message):
    if agent_number == 0:
        print(message)


def main(agent_number, global_value_net, global_policy_net, update_lock):
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

    memory = BackwardMemory()
    # env = TimeLimit(gym.make(env_name, render_mode='human'), max_episode_steps=max_episode_steps)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    agent = AcAgent(env.action_space, policy_net, local_seed)
    state = env.reset(seed=local_seed)[0]
    score = np.zeros(max_frames)
    episodes = 0
    current_score = 0
    previous_score = 0
    T = 0

    while T < max_frames:
        value_grad = value_net.zeros_like()
        policy_grad = policy_net.zeros_like()
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())
        memory.clear()
        final = False
        truncated = False
        # actions = []

        while not (final or truncated) and T < max_frames:
            action = agent.get_action(state)
            next_state, reward, final, truncated, _ = env.step(action)
            memory.push(state, action, next_state, reward, final)
            state = next_state

            current_score += reward
            score[T] = previous_score

            T += 1
            # actions.append(action)
        state = env.reset()[0]
        episodes += 1
        previous_score = current_score
        current_score = 0

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
        print_progress(agent_number, f"{T}/{max_frames}| score: {current_score}, "
                                     f"critic_loss: {float(critic_loss)}, actor_loss: {float(actor_loss)}")

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

    global_value_net = ValueNet(env_state_dim, hidden).to(device)
    global_policy_net = PolicyNet(env_state_dim, hidden, env_actions).to(device)

    with Manager() as manager, Pool(processes=processes) as pool:
        update_lock = manager.Lock()
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent, global_value_net, global_policy_net, update_lock)
                     for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score), (episodes, score)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'{env_name} {max_episode_steps} A3C {agents} agents\n ' \
                f'hidden: {hidden}(selu), batch: episode, lr: {lr}, gamma: {gamma}, softmax, seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/a3c/{timestamp}_a3c_{agents}'
        if save_model:
            torch.save(global_value_net.state_dict(), filename + '_V.pth')
            torch.save(global_policy_net.state_dict(), filename + '_pi.pth')
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename+'.png', lr=None, mean_window=avg_frames)


