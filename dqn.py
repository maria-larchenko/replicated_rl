from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import random

# from multiprocessing import Manager, Pool, Process
from torch.multiprocessing import Manager, Pool, Process, set_start_method
from torch import float32
from classes.Agents import DqnAgent
from classes.Memory import ReplayMemory
from classes.Models import DQN
from drawing import plot_results, plot_result_frames
from gym.wrappers import TimeLimit

seed = 9882  # np.random.randint(10_000)
env_name = 'CartPole-v1'   # LunarLander-v2   CartPole-v1  Acrobot-v1  MountainCar-v0
save_model = False

lr = 0.001
hidden = 512
gamma = 0.99
max_frames = 10_000
max_episode_steps = 500 if env_name != "MountainCar-v0" else 200
avg_frames = 1000
batch_size = 128
mem_capacity = 10_000

processes = 3
agents = 3

eps_type = 'linear'  # const exp linear softmax
temperature = 5.0
eps_0 = 1.0
eps_min = 0.05
eps_steps = 50_000
eps_decay = 0.0

update_frequency = 64
polyak_coef = 0.1
elasticity = False  # 0.1


def to_tensor(x, dtype=float32):
    return torch.as_tensor(np.array(x), dtype=dtype).to(device)


def get_dim(env_space):
    if isinstance(env_space, gym.spaces.Discrete):
        return env_space.n
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape[0]


def grad_step(loss, model):
    grad = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for param, param_grad in zip(model.parameters(), grad):
            # if clamp:
            #     param_grad.data.clamp_(-clamp, clamp)
            param.copy_(param - lr * param_grad)
    return param_grad


def grad_step_optimiser(loss, model, optimizer):
    optimizer.zero_grad()
    loss.backward()
    # if clamp:
    #     for param_grad in model.parameters():
    #         param_grad.data.clamp_(-clamp, clamp)
    optimizer.step()


def target_step(model, target):
    with torch.no_grad():
        for model_param, target_param in zip(model.parameters(), target.parameters()):
            new_param = polyak_coef * model_param + (1 - polyak_coef) * target_param
            target_param.copy_(new_param)


def elastic_step(update_lock, model, tmp_model, global_model):
    distance = 0
    with torch.no_grad():
        with update_lock:
            for param, tmp_param, master_param in zip(model.parameters(), tmp_model.parameters(), global_model.parameters()):
                distance += torch.mean(tmp_param - master_param)
                param.copy_(param - elasticity * (tmp_param - master_param))
                master_param.copy_(master_param + elasticity * (tmp_param - master_param))
    return distance


def elastic_step_v2(update_lock, model, global_model, mean_dist=False):
    distance = 0
    with torch.no_grad():
        with update_lock:
            for param, master_param in zip(model.parameters(), global_model.parameters()):
                if mean_dist:
                    distance += torch.mean(param - master_param)
                new_param = param - elasticity * (param - master_param)
                new_master_param = master_param + elasticity * (param - master_param)
                param.copy_(new_param)
                master_param.copy_(new_master_param)
    return distance


def print_progress(agent_number, message, filter=True):
    if not filter or agent_number == 0:
        print(f"[{agent_number}]::" + message)


def get_filename():
    if elasticity:
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        return f'./output/dqn_elastic/{timestamp}_dqn_elastic_{agents}'
    else:
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        return f'./output/dqn/{timestamp}_dqn_{agents}'


def get_title():
    txt = f'{env_name} {max_episode_steps}| {agents} agents| DQN {"Elastic " if elasticity else ""}seed {seed}\n'
    if elasticity:
        txt += f'elasticity: {elasticity}, update_frequency: {update_frequency}\n'
    else:
        txt += f'polyak_coef: {polyak_coef}, update_frequency: {update_frequency}\n'
    txt += f'hidden: {hidden}(selu) ' \
           f'lr: {lr} ' \
           f'T: {temperature} ' \
           f'batch: {batch_size} ' \
           f'gamma: {gamma} ' \
           f'mem: {mem_capacity}'
    return txt


def main(agent_number, update_lock, env_s, env_a, global_model, device):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model = DQN(env_s, hidden, env_a).to(device)
    target = DQN(env_s, hidden, env_a).to(device)
    target.load_state_dict(model.state_dict())
    memory = ReplayMemory(seed=local_seed, device=device, capacity=mem_capacity)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    print_progress(agent_number, f'{sum(p.numel() for p in model.parameters())} weights, model: {model} using {device}')

    # env = TimeLimit(gym.make(env_name, render_mode="human"), max_episode_steps=max_episode_steps)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    agent = DqnAgent(env.action_space, model, local_seed, device)
    agent.set_greediness(eps_type, eps_0, eps_min, eps_steps, eps_decay, temperature)
    score = np.zeros(max_frames)
    epsilon = np.zeros(max_frames)
    learning_rates = np.zeros(max_frames)

    final, truncated = False, False
    current_score, prev_score, episodes = 0, 0, 0
    distance = 0
    try:
        state, info = env.reset(seed=local_seed)
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
            epsilon[t] = agent.eps
            learning_rates[t] = optimizer.param_groups[0]['lr']

            if len(memory) > batch_size:
                states, actions, rewards, next_states, finals = memory.sample(batch_size)
                # q_update = r,  for final s'
                #            r + gamma * max_a Q(s', :), otherwise
                indices = torch.stack((actions, actions))
                q_values = model(states).t().gather(0, indices)[0]
                with torch.no_grad():
                    new_q_values = global_model(next_states) if elasticity else target(next_states)
                q_update = rewards + (1.0 - finals) * gamma * new_q_values.max(dim=1).values
                loss = mse_loss(q_update, q_values)
                # ----------- gradient step
                grad_step_optimiser(loss, model, optimizer)
                # ---------------- target update:
                if not elasticity and t % update_frequency == 0:
                    target_step(model, target)
                # ---------------- elastic update:
                if elasticity and t % update_frequency:
                    distance = elastic_step_v2(update_lock, model, global_model)
                if t % 100 == 0:
                    print_progress(agent_number, f"{t}/{max_frames}| score: {current_score}, loss: {loss}")
    except ValueError as error:
        msg = f"{t}/{max_frames}| score: {prev_score}, loss: {loss}, distances: {distance}, act_prob: {agent.prob}"
        print_progress(agent_number, str(error), filter=False)
        print_progress(agent_number, msg, filter=False)
    env.close()
    if save_model and agent_number == 0:
        torch.save(model.state_dict(), get_filename()+'.pth')
    env.close()
    return episodes, score, epsilon, agent


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING DQN, processes: {processes}")
    print(f"SEED: {seed}")
    try:
        set_start_method('spawn')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'
    except RuntimeError:
        print("failed to use spawn for CUDA")
        device = torch.device('cpu')
        device_name = '-'
    env_s = get_dim(gym.make(env_name).observation_space)
    env_a = get_dim(gym.make(env_name).action_space)
    global_model = DQN(env_s, hidden, env_a).to(device)
    with Manager() as manager, Pool(processes=processes) as pool:
        update_lock = manager.Lock()
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent, update_lock, env_s, env_a, global_model, device) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score, eps), (episodes, score, eps)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        epsilons = [result[2] for result in agent_results]
        agent_objs = [result[3] for result in agent_results]
        print(f"played episodes: {episodes}")

        info = agent_objs[0].get_eps_info()
        filename = get_filename()
        title = get_title()
        plot_result_frames(scores, epsilon=epsilons[0], title=title, info=info,
                           filename=filename+'.png', lr=lr, mean_window=avg_frames)
        np.savetxt(filename+'_scores.txt', np.array(scores).T, header=title)



