from datetime import datetime

import gym
import numpy as np
import torch
import random

from gym.wrappers import TimeLimit
from torch import float32, int64

from classes.Agents import AcAgent
from classes.Memory import ReplayMemory, Memory
from classes.Models import ValueNet, PolicyNet
from drawing import plot_result_frames
# from multiprocessing import Manager, Pool, Process
from torch.multiprocessing import Manager, Pool, Process, set_start_method

seed = 8803  # np.random.randint(10_000)
env_name = 'LunarLander-v2'  # LunarLander-v2 CartPole-v1 BipedalWalker-v3
save_model = False
a3c = False
mem_clear = False

lr = 0.001
hidden = 512
gamma = 0.99
max_frames = 200_000
max_episode_steps = 500
avg_frames = 1000
batch_size = 256
mem_capacity = batch_size
clamp = False  # 1e-8
temperature = 5
processes = 6
agents = 6

update_frequency = 32
elasticity = 0.1
elastic_tmp = False


def get_dim(env_space):
    if isinstance(env_space, gym.spaces.Discrete):
        return env_space.n
    if isinstance(env_space, gym.spaces.Box):
        return env_space.shape[0]


def grad_step(loss, model):
    grad = torch.autograd.grad(loss, model.parameters())
    with torch.no_grad():
        for param, param_grad in zip(model.parameters(), grad):
            if clamp:
                param_grad.data.clamp_(-clamp, clamp)
            param.copy_(param - lr * param_grad)
    return param_grad


def elastic_step(update_lock, model, tmp_model, global_model):
    distance = 0
    with torch.no_grad():
        with update_lock:
            for param, tmp_param, master_param in zip(model.parameters(), tmp_model.parameters(),
                                                      global_model.parameters()):
                distance += torch.mean(tmp_param - master_param)
                param.copy_(param - elasticity * (tmp_param - master_param))
                master_param.copy_(master_param + elasticity * (tmp_param - master_param))
    return distance


def elastic_step_v2(update_lock, model, global_model):
    distance = 0
    with torch.no_grad():
        with update_lock:
            for param, master_param in zip(model.parameters(), global_model.parameters()):
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
    timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
    if a3c:
        return f'./output/a3c/{timestamp}_a3c_{agents}'
    if elasticity:
        return f'./output/ac_elastic/{timestamp}_ac_elastic_{agents}'
    else:
        return f'./output/ac/{timestamp}_ac_{agents}'


def get_title():
    if a3c:
        txt = f'{env_name} {max_episode_steps}| {agents} agents| A3C seed: {seed}\n ' \
              f'hidden: {hidden}(selu) batch: episode lr: {lr} gamma: {gamma} T: {temperature}'
    else:
        txt = f'{env_name} {max_episode_steps}| {agents} agents| AC {"Elastic " if elasticity else ""}seed {seed}\n'
        if mem_clear:
            txt += f'mem: clear, '
        if elasticity:
            txt += f'elasticity: {elasticity}, update_freq: {update_frequency}'
            txt += ' with tmp net\n' if elastic_tmp else '\n'
        txt += f'hidden: {hidden}(selu) lr: {lr} batch: {batch_size} gamma: {gamma} T: {temperature} clamp: {clamp}'
    return txt


def main_ac(agent_number, update_lock, env_s, env_a, global_value_net, global_policy_net, device):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    memory = ReplayMemory(seed=local_seed, device=device, capacity=mem_capacity)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    # env = TimeLimit(gym.make(env_name, render_mode='human'), max_episode_steps=max_episode_steps)
    value_net = ValueNet(env_s, hidden).to(device)
    policy_net = PolicyNet(env_s, hidden, env_a, temperature=temperature).to(device)
    if elasticity:
        if elastic_tmp:
            tmp_value_net = ValueNet(env_s, hidden).to(device)
            tmp_policy_net = PolicyNet(env_s, hidden, env_a).to(device)
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())
    agent = AcAgent(env.action_space, policy_net, local_seed, device)

    score = np.zeros(max_frames)
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
                state, info = env.reset()
            action = agent.get_action(state)
            next_state, reward, final, truncated, _ = env.step(action)
            memory.push(state, action, next_state, reward, final)
            state = next_state
            current_score += reward
            score[t] = prev_score

            if len(memory) >= batch_size:
                if elasticity and elastic_tmp and t % update_frequency == 0:
                    tmp_value_net.load_state_dict(value_net.state_dict())
                    tmp_policy_net.load_state_dict(policy_net.state_dict())
                states, actions, rewards, next_states, finals = memory.sample(batch_size)
                # ----------- gradient step 1: CRITIC
                with torch.no_grad():
                    next_v_values = value_net(next_states)
                advance = rewards + (1 - finals) * gamma * next_v_values.T - value_net(states).T
                critic_loss = (1 / 2 * advance ** 2).mean()
                grad_step(critic_loss, value_net)
                # ----------- gradient step 2: ACTOR
                indices = torch.stack((actions, actions))
                act_prob = policy_net(states).T.gather(0, indices)[0]
                advance = rewards + (1 - finals) * gamma * next_v_values.T - value_net(states).T  # recalculated without target!
                actor_loss = (- advance * torch.log(act_prob)).mean()
                param_grad = grad_step(actor_loss, policy_net)
                if mem_clear:
                    memory.clear()
                # # ---------------- elastic update:
                if elasticity and t % update_frequency == 0:
                    if elastic_tmp:
                        distance = elastic_step(update_lock, value_net, tmp_value_net, global_value_net)
                        distance = elastic_step(update_lock, policy_net, tmp_policy_net, global_policy_net)
                    else:
                        distance = elastic_step_v2(update_lock, value_net, global_value_net)
                        distance = elastic_step_v2(update_lock, policy_net, global_policy_net)
            if t > batch_size and t % 500 == 0:
                print_progress(agent_number, f"{t}/{max_frames}| score: {np.around(prev_score, decimals=3)}, "
                                             f"critic_loss: {np.around(float(critic_loss), decimals=5)}, "
                                             f"actor_loss: {np.around(float(actor_loss), decimals=5)}, "
                                             f"param_grad: {torch.mean(torch.abs(param_grad))}, "
                                             f"distances: {float(distance)}")
    except ValueError as error:
        msg = f"{t}/{max_frames}| score: {prev_score}, critic_loss: {critic_loss}, actor_loss: {actor_loss} distances: {distance}, act_prob: {agent.prob}"
        print_progress(agent_number, str(error), filter=False)
        print_progress(agent_number, msg, filter=False)
    env.close()
    if save_model and agent_number == 0:
        torch.save(value_net.state_dict(), get_filename() + '_v.pth')
        torch.save(policy_net.state_dict(), get_filename() + '_pi.pth')
    return episodes, score


def main_a3c(agent_number, update_lock, env_s, env_a, global_value_net, global_policy_net, device):
    local_seed = agent_number + seed
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    np.random.seed(local_seed)
    random.seed(local_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def to_tensor(x, dtype=float32):
        return torch.as_tensor(np.array(x), dtype=dtype).to(device)

    value_net = ValueNet(env_s, hidden).to(device)
    policy_net = PolicyNet(env_s, hidden, env_a, temperature=temperature).to(device)
    memory = Memory(device)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    agent = AcAgent(env.action_space, policy_net, local_seed, device)
    state = env.reset(seed=local_seed)[0]
    score = np.zeros(max_frames)
    current_score = 0
    previous_score = 0
    episodes = 0
    T = 0

    while T < max_frames:
        value_grad = value_net.zeros_like()
        policy_grad = policy_net.zeros_like()
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())
        memory.clear()
        final = False
        truncated = False

        while not (final or truncated) and T < max_frames:
            action = agent.get_action(state)
            next_state, reward, final, truncated, _ = env.step(action)
            memory.push(state, action, next_state, reward, final)
            state = next_state
            current_score += reward
            score[T] = previous_score
            T += 1
        state = env.reset()[0]
        episodes += 1
        previous_score = current_score
        current_score = 0
        # Advantage Actor-Critic: A(s, a) = Q(s, a) - V(s) = r + V(s') - V(s) = r + sum_i(r_i) - V(s)
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
        print_progress(agent_number, f"{T}/{max_frames}| score: {previous_score}, "
                                     f"critic_loss: {float(critic_loss)}, actor_loss: {float(actor_loss)}")
    env.close()
    return episodes, score


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING, processes: {processes}")
    print(f"SEED: {seed}")
    print(get_title())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cpu')
    device_name = 'cpu'
    try:
        set_start_method('spawn')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # torch.device('cpu')  #
        device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'  # 'cpu'  #
    except RuntimeError:
        pass
    env_s = get_dim(gym.make(env_name).observation_space)
    env_a = get_dim(gym.make(env_name).action_space)
    global_value_net = ValueNet(env_s, hidden).to(device)
    global_policy_net = PolicyNet(env_s, hidden, env_a, temperature=temperature).to(device)
    with Manager() as manager, Pool(processes=processes) as pool:
        update_lock = manager.Lock()
        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(i, update_lock, env_s, env_a, global_value_net, global_policy_net, device) for i in range(agents)]
        if a3c:
            agent_results = pool.starmap(main_a3c, pool_args)
        else:
            agent_results = pool.starmap(main_ac, pool_args)
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")
        filename = get_filename()
        title = get_title()
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename + '.png', lr=lr, mean_window=avg_frames)


