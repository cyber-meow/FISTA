import numpy as np
import torch
# import pywt

import fista
import proximal


def generate_mask(size, dropout=0.5):
    mask = np.zeros(size).reshape(-1)
    mask[:int(mask.shape[0]*dropout)] = 1
    np.random.shuffle(mask)
    return mask.reshape(size)


class Inpainting(object):

    def __init__(self, masked_img, mask, x0=None):
        """masked_img of values between 0 and 1"""
        self.masked_img = torch.tensor(
            masked_img, requires_grad=False, dtype=torch.float)
        self.mask = torch.tensor(
            mask, requires_grad=False, dtype=torch.float)
        self.reset(x0)

    def reset(self, x0=None):
        if x0 is not None:
            self.x = torch.tensor(
                x0, requires_grad=True, dtype=torch.float)
        else:
            self.x = torch.rand(self.masked_img.shape, requires_grad=True)
        self.counter = 0
        self.x_pre = self.x.clone().detach()
        self.x_diffs = []

    # def energey(self):
    #     square_err = torch.sum((self.masked_img - self.M * self.x)**2)

    def run(self, n_iters, lr, lamb, wavelet='db4', alpha=None):
        proximal_op = proximal.WaveletST(lamb, wavelet)
        if alpha is None:
            optimizer = fista.ForwardBackward([self.x], lr, proximal_op)
        else:
            optimizer = fista.FISTA([self.x], lr, proximal_op)
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = torch.sum((self.masked_img - self.mask * self.x)**2)/2
            loss.backward(retain_graph=True)
            if alpha is None:
                optimizer.step()
            else:
                optimizer.step(alpha(self.counter+1))
            self.x_diffs.append(torch.sum((self.x - self.x_pre)**2).item())
            self.x_pre = self.x.clone().detach()
            self.counter += 1
        return self.x
