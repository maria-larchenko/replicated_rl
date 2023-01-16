import numpy as np
import torch
from torch import nn


class Agent:

    def __init__(self, action_space, model, seed=None, device=None, dtype=torch.float32):
        self.action_space = np.arange(0, action_space.n)
        self._rng = np.random.default_rng(seed)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.prob = None
        if device is None:
            self.device = torch.device('cpu')
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

    def get_action(self, state, *args):
        pass

    def to_tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype).to(self.device)


class AcAgent(Agent):

    def get_action(self, state, *args):
        state = self.to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        self.prob = prob
        with torch.no_grad():
            return self._rng.choice(self.action_space, p=prob)


class DqnAgent(Agent):

    def __init__(self, action_space, model, seed=None, device=None, dtype=torch.float32):
        super().__init__(action_space, model, seed, device, dtype)
        self.eps = None
        self.greedy_type = None
        self.eps_0 = None
        self.eps_min = None
        self.eps_decay = None
        self.eps_decrease = None
        self.softmax = nn.Softmax(dim=0)

    def set_const_greediness(self, eps_0):
        self.greedy_type = 'const'
        self.eps_0 = eps_0

    def set_lin_greediness(self, eps_0, eps_min, eps_steps):
        self.greedy_type = 'linear'
        self.eps_0 = eps_0
        self.eps_min = eps_min
        self.eps_decrease = (eps_0 - eps_min) / eps_steps

    def set_exp_greediness(self, eps_0, eps_min, eps_decay):
        self.greedy_type = 'exp'
        self.eps_0 = eps_0
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def set_softmax_greediness(self):
        self.greedy_type = 'softmax'

    def get_greediness(self):
        if self.eps is None or self.greedy_type == 'const':
            self.eps = self.eps_0
        elif self.greedy_type == 'linear':
            new_eps = self.eps - self.eps_decrease
            self.eps = new_eps if new_eps > self.eps_min else self.eps_min
        elif self.greedy_type == 'exp':
            new_eps = self.eps * self.eps_decay
            self.eps = new_eps if new_eps > self.eps_min else self.eps_min
        return self.eps

    def get_eps_info(self):
        if self.greedy_type == 'softmax':
            return 'eps: softmax'
        else:
            return f'eps_0: {self.eps_0}\n eps_min: {self.eps_min}\n eps_decay: {self.eps_decay}'

    def get_action(self, state, *args):
        state = self.to_tensor(state)
        with torch.no_grad():    # for efficiency, don't calc grad
            if self.greedy_type == 'softmax':
                qval = self.model(state)
                prob = self.softmax(qval).cpu().detach().numpy()
                return self._rng.choice(self.action_space, p=prob)
            elif np.random.uniform() < self.get_greediness():
                return self._rng.choice(self.action_space)
            else:
                return self.model(state).argmax().item()

