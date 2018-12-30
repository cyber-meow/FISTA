import numpy as np
from copy import deepcopy
import torch.optim


class ForwardBackward(object):

    def __init__(self, params, lr, proximal, regularize_idxs=None):
        """__init__

        :param params: the parameters to update
        :param lr: learning rate
        :param proximal: proximal operator taking in input a tensor
        :regularize_idxs:
            A list of integers to indicate which element of the
            iterable `params` needs to be regularize.
            Regularize all if `regularize_idxs` is None.
        """
        self.params = list(params)
        self.optimizer = torch.optim.SGD(self.params, lr=lr)
        self.proximal = proximal
        self.regularize_idxs = regularize_idxs

    def step(self):
        self.optimizer.step()
        lr = self.optimizer.param_groups[0]['lr']
        for i, param in enumerate(self.params):
            # param.data = torch.tensor(
            #     self.proximal(param.detach().numpy(), self.lr),
            #     dtype=torch.float)
            if self.regularize_idxs is None or i in self.regularize_idxs:
                param.data = self.proximal(param, lr)

    def zero_grad(self):
        self.optimizer.zero_grad()


class FISTA(ForwardBackward):

    def __init__(self, params, lr, proximal, regularize_idxs=None):
        params = list(params)
        super().__init__(params, lr, proximal, regularize_idxs)
        self.params_pre = deepcopy(params)

    def step(self, alpha):
        # x_n
        self.params_now = deepcopy(self.params)
        for param_pre, param in zip(self.params_pre, self.params):
            # y_n = x_n + \alpha_n * (x_n - x_{n-1})
            param.data = param.data + alpha * (param.data - param_pre.data)
        # x_{n+1}
        super().step()
        self.params_pre = self.params_now


class AlphaClassic(object):

    def __init__(self):
        self.ts = [1]

    @staticmethod
    def rec_formula(t):
        return np.sqrt(t**2+1/4)+1/2

    def __call__(self, n):
        if n > len(self.ts):
            for _ in range(n-self.ts):
                self.ts.append(self.rec_formula(self.ts[-1]))
        return self.ts[n-1]


class Alpha(object):

    def __init__(self, a):
        self.a = a

    def __call__(self, n):
        return (n-1)/(n+self.a)
