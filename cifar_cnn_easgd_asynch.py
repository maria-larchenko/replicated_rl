from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from itertools import count
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import from_numpy, as_tensor, float32, int64

from drawing import plot_results

seed = 75  # np.random.randint(low=0, high=2**10)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 0.001
epochs = 600
batch_size = 128
clamp = True
elasticity = 0.1
N = 1
commute_t = 1

training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {key: value for (key, value) in enumerate(training_data.classes)}
print(labels_map)

pic_width = 32
pic_height = 32
n_classes = len(labels_map)

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_id = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_id]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    print(img.T.shape)
    plt.imshow(torch.rot90(img.T, -1, [0, 1]))
plt.show()

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for batch, labels in test_dataloader:
    print("Shape of X [N, C, H, W]: ", batch.shape)
    print("Shape of y: ", labels.shape, labels.dtype)
    break


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()

        # Number of linear input connections depends on output of conv2d layers
        # and therefore the input image size, so lets compute it
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = conv_w * conv_h * 32

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(16, affine=False),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(32, affine=False),
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.BatchNorm2d(32, affine=False),
        )
        self.head = nn.Sequential(
            nn.Linear(linear_input_size, outputs),
            # nn.ReLU(),
            # nn.Linear(30, outputs)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = self.model(x)
        return self.head(x.view(x.size(0), -1))


def main():
    mse_loss = nn.MSELoss()
    master = CNN(pic_height, pic_width, n_classes).to(device)
    models = [CNN(pic_height, pic_width, n_classes).to(device) for _ in range(0, N)]
    tmp = CNN(pic_height, pic_width, n_classes).to(device)

    master.load_state_dict(models[0].state_dict())

    # Replicated SGD style:
    # with torch.no_grad():
    #     for i in range(1, N):
    #         for param, master_param in zip(models[i].parameters(), master.parameters()):
    #             new_master_param = master_param + param
    #             master_param.copy_(new_master_param)
    #     for master_param in master.parameters():
    #         master_param.divide_(N)

    # EA SGD style:
    for i in range(1, N):
        models[i].load_state_dict(master.state_dict())

    weights = sum(p.numel() for p in master.parameters())
    print(f'{weights} weights, model: {master}')
    print(f'Using {device} device: {device_name}')

    episode_durations = [[] for _ in range(0, N)]
    episode_counter = [0 for _ in range(0, N)]
    test_grad_0 = []
    test_grad_12 = []
    test_elasticity_0 = []
    test_elasticity_12 = []
    epsilon = []
    agent_states = []
    agent_screens = []

    print(f'START: {datetime.now().strftime("%Y.%m.%d %H-%M-%S")}')

    # for t in tqdm(count()):
    for t in count():
        # if len(episode_durations[0]) > episode_count \
        #         or len(episode_durations[1]) > episode_count \
        #         or len(episode_durations[2]) > episode_count:
        #     break

        if len(episode_durations[0]) > episode_count:
            break

        if t > 0 and t % 500 == 0:
            for n, episode_duration in enumerate(episode_durations):
                print(f'{n} episodes: {len(episode_duration)}, mean score: {np.mean(episode_duration)}')

        for i in range(0, N):
            episode_counter[i] += 1
            agent = agents[i]
            env = environments[i]
            state = agent_states[i]
            model = models[i]
            current_screen = agent_screens[i]

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
                next_state_values[non_final] = master(non_final_next_states).amax(dim=1)

                tmp.load_state_dict(model.state_dict())

                # q_update = r,  for final s'
                #            r + gamma * max_a Q(s', :), otherwise
                indices = torch.stack((actions, actions))
                q_values = model(states).t().gather(0, indices)[0]
                q_update = rewards + non_final * gamma * next_state_values

                loss = mse_loss(q_values, q_update)

                test_grad_val_0 = None
                test_grad_val_12 = None
                grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
                with torch.no_grad():
                    k = 0
                    for param, param_grad in zip(model.parameters(), grad):
                        if clamp:
                            param_grad.data.clamp_(-1, 1)
                        new_param = param - learning_rate * param_grad
                        param.copy_(new_param)
                        if k == 0:
                            test_grad_val_0 = torch.mean(torch.abs(learning_rate * param_grad))
                        if k == 7:
                            test_grad_val_12 = torch.mean(torch.abs(learning_rate * param_grad))
                        k += 1

                test_elasticity_val_0 = None
                test_elasticity_val_12 = None
                if commute_t is not None and t % commute_t == 0:
                    with torch.no_grad():
                        k = 0
                        for param, tmp_param, master_param in zip(model.parameters(), tmp.parameters(),
                                                                  master.parameters()):
                            if k == 0:
                                test_elasticity_val_0 = torch.mean(torch.abs(elasticity * (tmp_param - master_param)))
                            if k == 7:
                                test_elasticity_val_12 = torch.mean(torch.abs(elasticity * (tmp_param - master_param)))
                            k += 1
                            new_param = param - elasticity * (tmp_param - master_param)
                            new_master_param = master_param + elasticity * (tmp_param - master_param)
                            param.copy_(new_param)
                            master_param.copy_(new_master_param)

                if commute_t is not None and t % commute_t == 0:
                    test_grad_0.append(test_grad_val_0.to('cpu'))
                    test_grad_12.append(test_grad_val_12.to('cpu'))
                    test_elasticity_0.append(test_elasticity_val_0.to('cpu'))
                    test_elasticity_12.append(test_elasticity_val_12.to('cpu'))
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
    title = f'Asynch AESGD + DQN by AdamPaszke\n' \
            f'agents: {N}, commute: {commute_t}, elasticity: {elasticity}, ' \
            f'lr: {learning_rate}, gamma: {gamma}, batch: {batch_size}, clamp: {clamp}'
    info = f'eps: {eps_0}\n min: {eps_min}\n decay: {eps_decay}'
    time = datetime.now().strftime("%Y.%m.%d %H-%M")
    filename = f'./tmp_dqn_easgd_asynch/{time}_dqn_easgd_asynch.png'

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(test_grad_0, label='grad_0')
    ax0.plot(test_elasticity_0, label='elasticity_0')
    ax1.plot(test_grad_12, label='grad_7')
    ax1.plot(test_elasticity_12, label='elasticity_7')
    ax0.set_xlabel('elasticity updates')
    ax1.set_xlabel('elasticity updates')
    ax0.legend()
    ax1.legend()
    plt.savefig(f'./tmp_dqn_easgd_asynch/{time}_dqn_easgd_asynch grad.png')

    plot_results(episode_durations, epsilon, title, info, filename)
    # plt.show()


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
