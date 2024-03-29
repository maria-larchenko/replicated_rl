import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(inputs, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, outputs),
        )
        self.train()

    def forward(self, x):
        return self.model(x)


class ValueNet(nn.Module):
    def __init__(self, inputs, hidden):
        super(ValueNet, self).__init__()
        self.hidden = hidden
        self.model = nn.Sequential(
            nn.Linear(inputs, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, x):
        # return self.model(x).reshape(batch_size)
        return self.model(x)

    def zeros_like(self):
        zeros = []
        for p in self.parameters():
            zeros.append(torch.zeros_like(p))
        return zeros


class PolicyNet(nn.Module):
    def __init__(self, inputs, hidden, outputs, temperature=1.0):
        super(PolicyNet, self).__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.model = nn.Sequential(
            nn.Linear(inputs, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SELU(),
            nn.Linear(self.hidden, outputs),
        )

    def softmax(self, x):
        ex = torch.exp(x / self.temperature)
        ex = ex / ex.sum(0)
        return ex

    def forward(self, x):
        return self.softmax(self.model(x))

    def zeros_like(self):
        zeros = []
        for p in self.parameters():
            zeros.append(torch.zeros_like(p))
        return zeros



