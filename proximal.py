import numpy as np
import torch
import pywt


def soft_thresholding(x, th):
    return np.sign(x) * np.maximum(np.abs(x)-th, 0)


class SoftThresholding(object):

    def __init__(self, lamb):
        self.lamb = lamb

    def __call__(self, x, gamma):
        th = self.lamb * gamma
        x = soft_thresholding(x.detach().numpy(), th)
        return torch.tensor(x, dtype=torch.float, requires_grad=True)


class WaveletST(object):

    def __init__(self, lamb, wavelet='db4'):
        self.lamb = lamb
        self.wavelet = wavelet

    def __call__(self, x, gamma):
        th = self.lamb * gamma
        x_wav = pywt.wavedec2(x.detach().numpy(), self.wavelet)
        x_wav[0] = soft_thresholding(x_wav[0], th)
        for i, coeffs in enumerate(x_wav[1:]):
            cH = soft_thresholding(coeffs[0], th)
            cV = soft_thresholding(coeffs[1], th)
            cD = soft_thresholding(coeffs[2], th)
            x_wav[i+1] = cH, cV, cD
        x = pywt.waverec2(x_wav, self.wavelet)
        return torch.tensor(x, dtype=torch.float, requires_grad=True)
