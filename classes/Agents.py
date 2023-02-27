import numpy as np
import torch
import gym


class Agent:

    def __init__(self, action_space, model, seed=None, device=None, dtype=torch.float32):
        if isinstance(action_space, gym.spaces.Discrete):
            self._act_type = 0
            self.action_space = np.arange(0, action_space.n)
        if isinstance(action_space, gym.spaces.Box):
            self._act_type = 1
            self.action_space = action_space.shape[0]
        self._rng = np.random.default_rng(seed)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.prob = None
        if device is None:
            self.device = torch.device('cpu')
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

    def _get_action(self, prob=None):
        if self._act_type == 0:
            self.prob = prob
            with torch.no_grad():
                return self._rng.choice(self.action_space, p=self.prob)

    def get_action(self, state, *args):
        """implemented in subclasses"""
        pass

    def get_greedy_action(self, state, *args):
        """implemented in subclasses"""
        pass

    def to_tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype).to(self.device)


class AcAgent(Agent):

    def __init__(self, action_space, model, seed=None, device=None, dtype=torch.float32):
        super().__init__(action_space, model, seed, device, dtype)

    def get_action(self, state, *args):
        state = self.to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        return super()._get_action(prob)

    def get_greedy_action(self, state, *args):
        state = self.to_tensor(state)
        prob = self.model(state).cpu().detach().numpy()
        return super()._get_action(prob)


class DqnAgent(Agent):

    def __init__(self, action_space, model, seed=None, device=None, dtype=torch.float32):
        super().__init__(action_space, model, seed, device, dtype)
        self.eps = None
        self.greedy_type = None
        self.eps_0 = None
        self.eps_min = None
        self.eps_decay = None
        self.eps_decrease = None
        self.temperature = False

    def softmax(self, x):
        ex = torch.exp(x / self.temperature)
        ex = ex / ex.sum(0)
        return ex

    def set_greediness(self, greedy_type, eps_0=0.1, eps_min=0.1, eps_steps=1.0, eps_decay=1.0, temperature=1.0):
        """ types are:  const, linear, exp, softmax """
        self.greedy_type = greedy_type
        if greedy_type == 'const':
            self.eps_0 = eps_0
        if greedy_type == 'linear':
            self.eps_0 = eps_0
            self.eps_min = eps_min
            self.eps_decrease = (eps_0 - eps_min) / eps_steps
        if greedy_type == 'exp':
            self.eps_0 = eps_0
            self.eps_min = eps_min
            self.eps_decay = eps_decay
        if greedy_type == 'softmax':
            self.temperature = temperature

    def greediness_step(self):
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
            return f'softmax T: {self.temperature}'
        elif self.greedy_type == 'linear':
            return f'eps_0: {self.eps_0}\n eps_min: {self.eps_min}\n eps_decay: {np.around(self.eps_decrease, decimals=5)}'
        elif self.greedy_type == 'exp':
            return f'eps_0: {self.eps_0}\n eps_min: {self.eps_min}\n eps_decay: {self.eps_decay}'
        elif self.greedy_type == 'const':
            return f'eps_0: {self.eps_0}'

    def get_greedy_action(self, state, *args):
        return self.model(state).argmax().item()

    def get_action(self, state, *args):
        state = self.to_tensor(state)
        with torch.no_grad():
            if self.greedy_type == 'softmax':
                prob = self.softmax(self.model(state)).cpu().detach().numpy()
                return super()._get_action(prob)
            elif np.random.uniform() < self.greediness_step():
                return super()._get_action()
            else:
                return self.get_greedy_action(state)
