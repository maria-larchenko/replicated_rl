import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from gym import wrappers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Linear(8, 2),
        )

    # Called with either one element to determine
    # next action or a batch during optimization.
    def forward(self, x):
        x.to(device)
        return self.model(x)


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def main():
    learning_rate = 0.001
    eps = 0.1
    gamma = 0.999

    model = DQN().to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print('Using {} device'.format(device))
    print(model)

    outdir = './tmp/random-agent-results'
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, directory=outdir, force=True)

    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
