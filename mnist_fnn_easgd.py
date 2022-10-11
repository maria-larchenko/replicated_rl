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

device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = 'cpu'  # torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'

learning_rate = 1e-1 * 20
epochs = 10
batch_size = 60

workers = 4
commute_t = 10
elasticity = 0.01

print(f"elasticity as beta / (tau * p * lr): {0.9 / (commute_t * workers * learning_rate)}")
print(f"elasticity is set to: {elasticity}")
print(f"lr {learning_rate}, elasticity {elasticity},  workers {workers}, commute {commute_t}")


class FNN(nn.Module):
    def __init__(self, input_size, output_size, bias=False):
        super(FNN, self).__init__()
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


def train(dataloader, model, loss_fn, n_classes):
    size = len(dataloader.dataset)
    model.train()
    for batch_number, (batch, labels) in enumerate(dataloader):

        target = torch.zeros([batch_size, n_classes], dtype=torch.float32)
        for i, label in enumerate(labels):
            target[i, label] = 1.0
        batch, target = batch.to(device), target.to(device)

        pred = model(batch)
        loss = loss_fn(pred, target)

        grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
        with torch.no_grad():
            for param, param_grad in zip(model.parameters(), grad):
                new_param = param - learning_rate * param_grad
                param.copy_(new_param)
        if batch_number % 100 == 0:
            loss, current = loss.item(), batch_number * len(batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train_elastic(dataloaders, master, models, tmp, loss_fn, n_classes):
    batches = len(dataloaders[0])
    master.train()
    for model in models:
        model.train()

    t = 0
    for batch_number in range(0, batches):

        for n, (dataloader, model) in enumerate(zip(dataloaders, models)):
            batch, labels = next(iter(dataloader))

            target = torch.zeros([batch_size, n_classes], dtype=torch.float32)
            for i, label in enumerate(labels):
                target[i, label] = 1.0
            batch, target = batch.to(device), target.to(device)

            pred = model(batch)
            loss = loss_fn(pred, target)

            tmp.load_state_dict(model.state_dict())
            # gradient step
            grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
            with torch.no_grad():
                for param, param_grad in zip(model.parameters(), grad):
                    param.copy_(param - learning_rate * param_grad)
            # elastic step
            if t % commute_t == 0:
                with torch.no_grad():
                    for param, tmp_param, master_param in zip(model.parameters(), tmp.parameters(), master.parameters()):
                        param.copy_(param - elasticity * (tmp_param - master_param))
                        master_param.copy_(master_param + elasticity * (tmp_param - master_param))
            if n == 0:
                t += 1
                if batch_number % 100 == 0:
                    loss, current = loss.item(), batch_number * len(batch)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataloaders[0].dataset):>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, label in dataloader:
            batch, label = batch.to(device), label.to(device)
            pred = model(batch)
            # # MSELoss
            test_loss += loss_fn(pred.max(dim=1).values, label.float()).item()
            # # CrossEntropyLoss
            # test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def debug(training_data, model):
    labels_map = {key: value for (key, value) in enumerate(training_data.classes)}
    print(labels_map)
    figure, axs = plt.subplots(3, 3, figsize=(8, 8))
    for ax in axs.reshape(9):
        sample_id = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_id]
        ax.set_title(labels_map[model(img).argmax(1).item()])
        ax.axis("off")
        img = img.T
        img = torch.rot90(img, 1, [0, 1])
        img = torch.flip(img, [0])
        ax.imshow(img, cmap="gray")
    plt.show()


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
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    train_dataloaders = [DataLoader(training_data, batch_size=batch_size, shuffle=True) for _ in range(0, workers)]
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    batch, labels = next(iter(test_dataloader))
    n_classes = len(test_data.classes)
    input_size = np.prod(batch.shape[1:])
    print("Shape of X [N, C, H, W]: ", batch.shape)
    print("Shape of y: ", labels.shape, labels.dtype)
    print("Input size: ", input_size)

    # Fully connected NN
    loss_fn = nn.MSELoss()
    master = FNN(input_size, n_classes).to(device)
    models = [FNN(input_size, n_classes).to(device) for _ in range(0, workers)]
    tmp = FNN(input_size, n_classes).to(device)

    # EASGD init:
    for model in models:
        model.load_state_dict(master.state_dict())

    # # Replicated init:
    # master.load_state_dict(models[0].state_dict())
    # with torch.no_grad():
    #     for i in range(1, workers):
    #         for param, master_param in zip(models[i].parameters(), master.parameters()):
    #             master_param.copy_(master_param + param)
    #     for master_param in master.parameters():
    #         master_param.divide_(workers)

    weights = sum(p.numel() for p in master.parameters())
    print(f'{weights} weights, model: {master}')
    print(f'Using {device} device: {device_name}')
    print(f'START: {datetime.now().strftime("%Y.%m.%d %H-%M-%S")}')
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, master, loss_fn, n_classes)
        # train_elastic(train_dataloaders, master, models, tmp, loss_fn, n_classes)
        test(test_dataloader, master, loss_fn)
    print(f'FINISHED: {datetime.now().strftime("%Y.%m.%d %H-%M-%S")}')
    debug(test_data, master)


if __name__ == '__main__':
    main()
