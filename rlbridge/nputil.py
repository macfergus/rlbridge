def concat_inplace(x, y):
    n1 = x.shape[0]
    n2 = y.shape[0]
    n_final = n1 + n2
    final_shape = (n_final,) + x.shape[1:]
    x.resize(final_shape, refcheck=False)
    x[n1:] = y


def shrink_axis(array, n):
    """Shrink the 0th axis of an array in-place."""
    new_shape = (n,) + array.shape[1:]
    array.resize(new_shape, refcheck=False)
