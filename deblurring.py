import math
import numbers
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import fista
import proximal


def blur(img, kernel_size, sigma, noise_level=0):
    blurring = GaussianSmoothing(kernel_size, sigma)
    blurred = blurring(torch.tensor(img, dtype=torch.float))
    return blurred.numpy() + noise_level * np.random.randn(*img.shape)


class Deblurring(object):

    def __init__(self, blurred_img, kernel_size, sigma, x0=None):
        self.blurred_img = torch.tensor(
            blurred_img, requires_grad=False, dtype=torch.float)
        self.gaussian_blurring = GaussianSmoothing(kernel_size, sigma)
        self.reset(x0)

    def reset(self, x0=None):
        if x0 is not None:
            self.x = torch.tensor(
                x0, requires_grad=True, dtype=torch.float)
        else:
            self.x = torch.rand(self.blurred_img.shape, requires_grad=True)
        self.counter = 0
        self.x_pre = self.x.clone().detach()
        self.x_diffs = []

    def run(self, n_iters, lr, lamb, wavelet='db4', alpha=None):
        proximal_op = proximal.WaveletST(lamb, wavelet)
        if alpha is None:
            optimizer = fista.ForwardBackward([self.x], lr, proximal_op)
        else:
            optimizer = fista.FISTA([self.x], lr, proximal_op)
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = torch.sum(
                (self.blurred_img - self.gaussian_blurring(self.x))**2)/2
            loss.backward(retain_graph=True)
            if alpha is None:
                optimizer.step()
            else:
                optimizer.step(alpha(self.counter+1))
            self.x_diffs.append(torch.sum((self.x - self.x_pre)**2).item())
            self.x_pre = self.x.clone().detach()
            self.counter += 1
        return self.x


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor.
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional):
            The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, kernel_size, sigma, dim=2):

        super(GaussianSmoothing, self).__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1 / (std * math.sqrt(2 * math.pi))
                * torch.exp(-((mgrid-mean)/std)**2/2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())

        self.register_buffer('weight', kernel)
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.')
        self.padding = tuple((np.array(kernel_size)-1)//2)

    def forward(self, x):
        """
        Apply gaussian filter to input.

        :param x (torch.Tensor): Input to apply gaussian filter on.
        """
        x = x.view(1, 1, *x.size())
        return self.conv(x, weight=self.weight, padding=self.padding).squeeze()
