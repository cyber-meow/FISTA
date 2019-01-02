import torch
from torch import nn
from torch.nn import functional as F

import fista
import proximal


class TVDenoising(object):

    def __init__(self, noisy_img, p0=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.noisy_img = torch.tensor(
            noisy_img,
            requires_grad=False,
            dtype=torch.float,
            device=self.device)
        self.reset(p0)

    def reset(self, p0=None):
        if p0 is not None:
            self.p = torch.tensor(
                p0, requires_grad=True,
                dtype=torch.float, device=self.device)
        else:
            p_shape = list(self.noisy_img.shape) + [2]
            self.p = torch.rand(
                p_shape, requires_grad=True, device=self.device)
        self.counter = 0
        self.p_pre = self.p.clone().detach()
        self.p_diffs = []
        self.energies = []

    def run(self, n_iters, lr, lamb, alpha=None):
        proximal_op = proximal.ProjectInf(lamb)
        if alpha is None:
            optimizer = fista.ForwardBackward([self.p], lr, proximal_op)
        else:
            optimizer = fista.FISTA([self.p], lr, proximal_op)
        for _ in range(n_iters):
            optimizer.zero_grad()
            loss = torch.sum(
                (self.noisy_img + divergence(self.p))**2)/2
            loss.backward(retain_graph=True)
            if alpha is None:
                optimizer.step()
            else:
                optimizer.step(alpha(self.counter+1))
            self.p_diffs.append(torch.sum((self.p - self.p_pre)**2).item())
            self.energies.append(loss.item())
            self.p_pre = self.p.clone().detach()
            self.counter += 1
        self.x = self.noisy_img + divergence(self.p)
        return self.x


# Seems to work better than the one below
def divergence(x):
    """Compute the divergence of a 2d vector field, naive"""
    x_h = x[..., 0]
    h_grad = torch.cat([x_h[:, None, 0], x_h[:, 1:] - x_h[:, :-1]], dim=1)
    x_v = x[..., 1]
    v_grad = torch.cat([x_v[0, None], x_v[1:] - x_v[:-1]], dim=0)
    return h_grad + v_grad


class Divergence(nn.Module):
    """Compute the divergence of a 2d vector field with Soblev filter"""

    def __init__(self):
        super(Divergence, self).__init__()
        conv_h = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float)
        conv_v = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)
        self.register_buffer('conv_h', conv_h.view(1, 1, 3, 3))
        self.register_buffer('conv_v', conv_v.view(1, 1, 3, 3))

    def forward(self, x):
        """forward

        :param x (torch.Tensor): 2d vector field, of shape (H, W, 2)
        """
        x = x.view(1, 1, *x.size())
        h_grad = F.conv2d(x[..., 0], weight=self.conv_h, padding=1).squeeze()
        v_grad = F.conv2d(x[..., 1], weight=self.conv_v, padding=1).squeeze()
        return h_grad + v_grad
