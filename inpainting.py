import numpy as np
import torch
import pywt

import fista
import proximal


def generate_mask(size, dropout=0.5):
    mask = np.zeros(size).reshape(-1)
    mask[:int(mask.shape[0]*dropout)] = 1
    np.random.shuffle(mask)
    return mask.reshape(size)


class Inpainting(object):

    def __init__(self, masked_img, mask, x0=None, ergodic_weight=None):
        """masked_img of values between 0 and 1"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.masked_img = torch.tensor(
            masked_img,
            requires_grad=False,
            dtype=torch.float,
            device=self.device)
        self.mask = torch.tensor(
            mask,
            requires_grad=False,
            dtype=torch.float,
            device=self.device)
        self.reset(x0)
        self.ergodic_weight = ergodic_weight

    def reset(self, x0=None):
        if x0 is not None:
            self.x = torch.tensor(
                x0, requires_grad=True,
                dtype=torch.float, device=self.device)
        else:
            self.x = torch.rand(
                self.masked_img.shape,
                requires_grad=True, device=self.device)
        self.counter = 0
        self.x_pre = self.x.clone().detach()
        self.x_diffs = []
        self.energies = []
        self.l1s = []
        self.ergodic_acc = 0
        self.ergodic_x = np.zeros(self.masked_img.shape)

    def wavelet_l1(self, wavelet):
        res = 0
        x_wav = pywt.wavedec2(self.x.cpu().detach().numpy(), wavelet)
        res += np.sum(np.abs(x_wav[0]))
        for coeffs in x_wav[1:]:
            res += np.sum(np.abs(coeffs))
        return res

    def run(self, n_iters, lr, lamb,
            wavelet='db4', alpha=None, err_levels=None):
        proximal_op = proximal.WaveletST(lamb, wavelet)
        if alpha is None:
            optimizer = fista.ForwardBackward([self.x], lr, proximal_op)
        else:
            optimizer = fista.FISTA([self.x], lr, proximal_op)
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = torch.sum((self.masked_img - self.mask * self.x)**2)/2
            loss.backward(retain_graph=True)
            err_level = (None if err_levels is None
                         else err_levels(self.counter+1))
            if alpha is None:
                optimizer.step(err_level)
            else:
                optimizer.step(alpha(self.counter+1), err_level)
            self.x_diffs.append(torch.sum((self.x - self.x_pre)**2).item())
            self.energies.append(loss.item() + lamb*self.wavelet_l1(wavelet))
            self.l1s.append(self.wavelet_l1(wavelet))
            self.x_pre = self.x.clone().detach()
            if self.ergodic_weight is not None:
                weight = self.ergodic_weight(self.counter+1)
                self.ergodic_x += self.x_pre.cpu().numpy() * weight
                self.ergodic_acc += weight
            self.counter += 1
        return self.x
