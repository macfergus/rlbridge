import numpy as np
from scipy import signal


def concat_inplace(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    n_final = x_size + y_size
    final_shape = (n_final,) + x.shape[1:]
    x.resize(final_shape, refcheck=False)
    x[x_size:] = y


def shrink_axis(array, n):
    """Shrink the 0th axis of an array in-place."""
    new_shape = (n,) + array.shape[1:]
    array.resize(new_shape, refcheck=False)


def smooth(x, window_size, width=1):
    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')
    n_pad = (window_size - 1) // 2
    kernel = signal.gaussian(window_size, std=1)
    x_pad = np.concatenate([
        x[0] * np.ones(n_pad),
        x,
        x[-1] * np.ones(n_pad)
    ])
    x_smooth = np.convolve(x_pad, kernel, mode='valid') / np.sum(kernel)
    return x_smooth
