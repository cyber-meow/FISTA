from copy import deepcopy
import torch.optim


class ForwardBackward(object):

    def __init__(self, params, lr, proximal):
        """__init__

        :param params: the parameters to update
        :param lr: learning rate
        :param proximal: proximal operator taking in input a tensor
        """
        self.params = params
        self.lr = lr
        self.optimizer = torch.optim.SGD(params, lr=lr)
        self.proximal = proximal

    def step(self):
        self.optimizer.step()
        for param in self.params:
            # param.data = torch.tensor(
            #     self.proximal(param.detach().numpy(), self.lr),
            #     dtype=torch.float)
            param.data = self.proximal(param, self.lr)

    def zero_grad(self):
        self.optimizer.zero_grad()


class FISTA(ForwardBackward):

    def __init__(self, params, lr, proximal):
        super().__init__(params, lr, proximal)
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
