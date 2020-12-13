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
