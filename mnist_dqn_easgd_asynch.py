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

seed = 73  # np.random.randint(low=0, high=2**10)
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 1e-7
epochs = 10
batch_size = 60
clamp = False
L2_reg = 0  # 1e-4

elasticity = 0.1
N = 1
commute_t = 1


class DQN(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super(DQN, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100, bias=bias),
            nn.ReLU(),
            nn.Linear(100, 30, bias=bias),
            nn.ReLU(),
            nn.Linear(30, output_size, bias=bias)
        )

    def forward(self, x):
        x = x.float()
        x = self.flatten(x)
        return self.model(x)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch_number, (batch, labels) in enumerate(dataloader):
        batch, labels = batch.to(device), labels.to(device)

        # MSELoss
        labels = labels.float()
        pred = model(batch).max(dim=1).values

        # CrossEntropyLoss
        # pred = model(batch)

        loss = loss_fn(pred, labels)
        grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
        with torch.no_grad():
            for param, param_grad in zip(model.parameters(), grad):
                if clamp:
                    param_grad.data.clamp_(-1, 1)
                new_param = param - learning_rate * param_grad
                # new_param = param - learning_rate * (param_grad + L2_reg * param)  # L2 REG
                param.copy_(new_param)

        if batch_number % 100 == 0:
            loss, current = loss.item(), batch_number * len(batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, label in dataloader:
            batch, label = batch.to(device), label.to(device)
            pred = model(batch)
            # MSELoss
            test_loss += loss_fn(pred.max(dim=1).values, label.float()).item()

            # # CrossEntropyLoss
            # test_loss += loss_fn(pred, label).item()

            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    labels_map = {key: value for (key, value) in enumerate(training_data.classes)}
    print(labels_map)
    n_classes = len(labels_map)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    input_size = 0
    for batch, labels in test_dataloader:
        input_size = np.prod(batch.shape[1:])
        print("Shape of X [N, C, H, W]: ", batch.shape)
        print("Shape of y: ", labels.shape, labels.dtype)
        print("Input size: ", input_size)
        break

    # debug
    # figure, axs = plt.subplots(3, 3, figsize=(8, 8))
    # for ax in axs.reshape(9):
    #     sample_id = torch.randint(len(training_data), size=(1,)).item()
    #     img, label = training_data[sample_id]
    #     ax.set_title(labels_map[label])
    #     ax.axis("off")
    #     img = img.T
    #     img = torch.rot90(img, 1, [0, 1])
    #     img = torch.flip(img, [0])
    #     ax.imshow(img, cmap="gray")
    # plt.show()

    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()

    master = DQN(input_size, n_classes).to(device)
    models = [DQN(input_size, n_classes).to(device) for _ in range(0, N)]
    tmp = DQN(input_size, n_classes).to(device)

    master.load_state_dict(models[0].state_dict())
    optimizer = torch.optim.SGD(master.parameters(), lr=learning_rate)

    weights = sum(p.numel() for p in master.parameters())
    print(f'{weights} weights, model: {master}')
    print(f'Using {device} device: {device_name}')
    print(f'START: {datetime.now().strftime("%Y.%m.%d %H-%M-%S")}')
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, master, loss_fn, optimizer)
        test(test_dataloader, master, loss_fn)
    print("Done!")


# https://google.github.io/styleguide/pyguide.html#317-main
if __name__ == '__main__':
    main()
