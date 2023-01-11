from datetime import datetime

import gym
import numpy as np
import torch
import random

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
lr = 0.0001
hidden = 256
gamma = 0.99
max_frames = 10_000
avg_frames = 1000
batch_size = 64
processes = 3
agents = 3

update_frequency = 32  # 32
rho = 1.0
elastic_type = 'max'  # max, Dkl


def to_tensor(x, dtype=float32):
    return torch.as_tensor(x, dtype=dtype).to(device)


def distance_max(states, model, master_model):
    return torch.max(torch.abs(master_model(states) - model(states)))


def divergence(states, model, master_model):
    p = master_model(states)
    q = model(states)
    return torch.sum(p * torch.log(p / q))


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

    tmp_value_net = ValueNet(env_state_dim, hidden).to(device)
    tmp_policy_net = PolicyNet(env_state_dim, hidden, env_actions).to(device)

    value_net.load_state_dict(global_value_net.state_dict())
    policy_net.load_state_dict(global_policy_net.state_dict())

    memory = ReplayMemory()
    env = gym.make(env_name)
    agent = AcAgent(env.action_space, policy_net, local_seed)

    score = np.zeros(max_frames)
    final = False
    current_score = 0
    prev_score = 0
    episodes = 0
    state = env.reset(seed=local_seed)[0]

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

            if t % update_frequency == 0:  # ---------------- elastic update:
                with update_lock:
                    tmp_value_net.load_state_dict(global_value_net.state_dict())
                    tmp_policy_net.load_state_dict(global_policy_net.state_dict())
                    if elastic_type == 'max':
                        distance_value = distance_max(states, value_net, tmp_value_net)
                        distance_policy = distance_max(states, policy_net, tmp_policy_net)
                    elif elastic_type == 'Dkl':
                        distance_value = divergence(states, value_net, tmp_value_net)
                        distance_policy = divergence(states, policy_net, tmp_policy_net)
                    grad_dist_value = torch.autograd.grad(distance_value, value_net.parameters())
                    grad_dist_policy = torch.autograd.grad(distance_policy, policy_net.parameters())
                    elasticity_v = 0 if torch.isnan(distance_value) else lr * critic_loss/distance_value * rho
                    elasticity_p = 0 if torch.isnan(distance_policy) else lr * actor_loss/distance_policy * rho
                    with torch.no_grad():
                        for param, master_param, grad_distance in zip(value_net.parameters(),
                                                                      global_value_net.parameters(), grad_dist_value):
                            param.copy_(param - elasticity_v * grad_distance)
                            master_param.copy_(master_param + elasticity_v * grad_distance)
                        for param, master_param, grad_distance in zip(policy_net.parameters(),
                                                                      global_policy_net.parameters(), grad_dist_policy):
                            param.copy_(param - elasticity_p * grad_distance)
                            master_param.copy_(master_param + elasticity_p * grad_distance)
                    print_progress(agent_number, f"score: {prev_score}, C-loss: {critic_loss}, A-loss: {actor_loss}, "
                                   f"elasticity_v: {elasticity_v}, elasticity_p: {elasticity_p}, "
                                   f"distance_v: {distance_value}, distance_p: {distance_policy}")
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

    global_value_net = ValueNet(env_state_dim, hidden).to(device)
    global_policy_net = PolicyNet(env_state_dim, hidden, env_actions).to(device)

    with Manager() as manager, Pool(processes=processes) as pool:
        update_lock = manager.Lock()

        print(f"------------------------------------ started: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        pool_args = [(agent, global_value_net, global_policy_net, update_lock) for agent in range(agents)]
        agent_results = pool.starmap(main, pool_args)  # [(episodes, score), (episodes, score)]
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")

        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")

        title = f'AC Stat. Elastic {agents} agents (asynch)\n' \
                f'update_frequency: {update_frequency}, ' \
                f'stat_elasticity: lr*loss/dist, \n' \
                f'hidden: {hidden}(selu), ' \
                f'batch: {batch_size}, ' \
                f'lr: {lr}, ' \
                f'gamma: {gamma}, ' \
                f'softmax, ' \
                f'seed: {seed}'
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        filename = f'./output/ac_elastic/{timestamp}_ac_stat_elastic_{agents}.png'
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename, lr=lr, mean_window=avg_frames)
