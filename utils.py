import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def to_img(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))*255
    return Image.fromarray(np.uint8(x))


def plot_energy(energy, min_energy, n_iters, **kwargs):
    energy = np.array(energy)
    plt.plot((energy-min_energy)[:n_iters], **kwargs)
