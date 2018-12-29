import numpy as np
from PIL import Image


def to_img(x):
    x = (x-np.min(x))/(np.max(x)-np.min(x))*255
    return Image.fromarray(np.uint8(x))
