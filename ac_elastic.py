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

env_name = 'LunarLander-v2'  # LunarLander-v2 CartPole-v1
env_actions = 4
env_state_dim = 8
save_model = False

lr = 0.0001
hidden = 256
gamma = 0.99
max_frames = 100_000
max_episode_steps = 500
avg_frames = 1000
batch_size = 64
clamp = 1e-08
processes = 3
agents = 3

update_frequency = 32
elasticity = False


def to_tensor(x, dtype=float32):
    return torch.as_tensor(np.array(x), dtype=dtype).to(device)


def print_progress(agent_number, message, filter=True):
    if not filter or agent_number == 0:
        print(f"[{agent_number}]::" + message)


def get_filename():
    if elasticity:
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        return f'./output/ac_elastic/{timestamp}_ac_elastic_{agents}'
    else:
        timestamp = datetime.now().strftime("%Y.%m.%d %H-%M-%S")
        return f'./output/ac/{timestamp}_ac_{agents}'


def get_title():
    if elasticity:
        return f'{env_name} {max_episode_steps}| AC Elastic {agents} agents\n' \
               f'update_frequency: {update_frequency} ' \
               f'elasticity: {elasticity} \n' \
               f'hidden: {hidden}(selu) ' \
               f'batch: {batch_size} ' \
               f'lr: {lr} ' \
               f'gamma: {gamma} ' \
               f'clamp: {clamp} ' \
               f'seed: {seed}'
    else:
        return f'{env_name} {max_episode_steps}| AC {agents} agents\n ' \
               f'hidden: {hidden}(selu) ' \
               f'batch: {batch_size} ' \
               f'lr: {lr} ' \
               f'gamma: {gamma} ' \
               f'clamp: {clamp} ' \
               f'seed: {seed}'


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

    if elasticity:
        tmp_value_net = ValueNet(env_state_dim, hidden).to(device)
        tmp_policy_net = PolicyNet(env_state_dim, hidden, env_actions).to(device)
        value_net.load_state_dict(global_value_net.state_dict())
        policy_net.load_state_dict(global_policy_net.state_dict())

    memory = ReplayMemory(seed=local_seed)
    # env = TimeLimit(gym.make(env_name, render_mode='human'), max_episode_steps=max_episode_steps)
    env = TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
    agent = AcAgent(env.action_space, policy_net, local_seed)

    score = np.zeros(max_frames)
    final, truncated = False, False
    current_score, prev_score = 0, 0
    episodes = 0
    state, info = env.reset(seed=local_seed)

    try:
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

            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                states = to_tensor(batch.state)
                actions = to_tensor(batch.action, dtype=int64)
                rewards = to_tensor(batch.reward)
                next_states = to_tensor(batch.next_state)
                finals = to_tensor(batch.final)

                if elasticity and t % update_frequency == 0:
                    tmp_value_net.load_state_dict(value_net.state_dict())
                    tmp_policy_net.load_state_dict(policy_net.state_dict())

                # ----------- gradient step 1: CRITIC
                advance = rewards + (1 - finals) * gamma * value_net(next_states).T - value_net(states).T
                critic_loss = (1 / 2 * advance ** 2).mean()
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

                distances = 0
                if elasticity and t % update_frequency == 0:  # ---------------- elastic update:
                    with torch.no_grad():
                        with update_lock:
                            for param, tmp_param, master_param in zip(value_net.parameters(), tmp_value_net.parameters(),
                                                                      global_value_net.parameters()):
                                param.copy_(param - elasticity * (tmp_param - master_param))
                                master_param.copy_(master_param + elasticity * (tmp_param - master_param))
                            for param, tmp_param, master_param in zip(policy_net.parameters(), tmp_policy_net.parameters(),
                                                                      global_policy_net.parameters()):
                                distances += torch.mean(tmp_param - master_param)
                                param.copy_(param - elasticity * (tmp_param - master_param))
                                master_param.copy_(master_param + elasticity * (tmp_param - master_param))
                if t % 100 == 0:
                    print_progress(agent_number, f"{t}/{max_frames}| score: {np.around(prev_score, decimals=3)}, "
                                                 f"critic_loss: {np.around(float(critic_loss), decimals=5)}, "
                                                 f"actor_loss: {np.around(float(actor_loss), decimals=5)}, "
                                                 f"param_grad: {torch.mean(param_grad)}, "
                                                 f"distances: {float(distances)}")
    except ValueError as error:
        msg = f"{t}/{max_frames}| score: {prev_score}, critic_loss: {critic_loss}, actor_loss: {actor_loss} distances: {distances}, act_prob: {agent.prob}"
        print_progress(agent_number, str(error), filter=False)
        print_progress(agent_number, msg, filter=False)
    env.close()
    if save_model and agent_number == 0:
        torch.save(value_net.state_dict(), get_filename()+'_v.pth')
        torch.save(policy_net.state_dict(), get_filename()+'_pi.pth')
    return episodes, score


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    print(f"MULTIPROCESSING AC {'ELASTIC' if elasticity else ''}, processes: {processes}")
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
        agent_results = pool.starmap(main, pool_args)
        print(f"------------------------------------ finished: {datetime.now().strftime('%Y.%m.%d %H-%M-%S')}")
        episodes = [result[0] for result in agent_results]
        scores = [result[1] for result in agent_results]
        print(f"played episodes: {episodes}")
        filename = get_filename()
        title = get_title()
        plot_result_frames(scores, epsilon=None, title=title, info=None,
                           filename=filename+'.png', lr=lr, mean_window=avg_frames)


