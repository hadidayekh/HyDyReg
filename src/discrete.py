from logging import fatal
import numpy as np

"""
This function takes a single trajectory of input output data (u,y) and
extracts (x_t, y_t) pairs 
"""


def extract_from_trajectory(u_array, y_array, nc, na):
    N = u_array.T.shape[0]
    if (N != y_array.T.shape[0]):
        fatal("u_array and y_array are not the same size")
    nmax = max(nc, na)

    u_dim = 1
    y_dim = 1
    if u_array.ndim == 2:
        u_dim = u_array.shape[0]
    if y_array.ndim == 2:
        y_dim = y_array.shape[0]

    N_new = N - nmax

    X_array = np.zeros((N_new, nc * u_dim + na * y_dim))

    # print(u_dim)
    # print(y_dim)
    # print(u_array)
    # print(y_array)
    # print(X_array.shape)

    for i in range(nc):
        i_inv = nc * u_dim - u_dim * i - u_dim
        if u_dim == 1:
            X_array.T[i_inv] = u_array[nmax - i - 1: N - i - 1]
        else:
            for j in range(u_dim):
                X_array.T[i_inv + j] = u_array.T[nmax - i - 1: N - i - 1].T[j]
        
    for i in range(na):
        i_inv = na * y_dim - y_dim * i - y_dim
        #X_array.T[i_inv + nc] = y_array[nmax - i - 1: N - i - 1]
        if y_dim == 1:
            X_array.T[i_inv + nc * u_dim] = y_array[nmax - i - 1: N - i - 1]
        else:
            for j in range(y_dim):
                X_array.T[i_inv + nc * u_dim + j] = y_array.T[nmax - i - 1: N - i - 1].T[j]

    y_output = y_array.T[nmax:].T
    return X_array, y_output
