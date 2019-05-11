def concat_inplace(x, y):
    n1 = x.shape[0]
    n2 = y.shape[0]
    n_final = n1 + n2
    final_shape = (n_final,) + x.shape[1:]
    x.resize(final_shape, refcheck=False)
    x[n1:] = y
