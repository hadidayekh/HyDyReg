import numpy as np

def linear(X, Y, args):
    # Check if X=Y
    sym = isinstance(Y, type(None))
    if sym:
        Y = X

    c = args[0]
    return np.dot(X, Y.T) + c


def polynomial(X, Y, args):
    # Check if X=Y
    sym = isinstance(Y, type(None))
    if sym:
        Y = X

    d = args[0]
    c = args[1]
    gamma = args[2]
    return (gamma * np.dot(X, Y.T) + c)**d

def rbf(X, Y, args):
    # Check if X=Y
    sym = isinstance(Y, type(None))
    if sym:
        Y = X
    if X.ndim == 1:
        X = X.reshape(1, X.size)
    if Y.ndim == 1:
        Y = Y.reshape(1, Y.size)
    sigma = args[0]

    X_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in X]).T
    Y_norms = X_norms if sym else np.mat(
        [np.mat(np.dot(v, v.T))[0, 0] for v in Y]).T

    K1 = X_norms * np.mat(np.ones((Y.shape[0], 1), dtype=np.float64)).T
    K2 = np.mat(np.ones((X.shape[0], 1), dtype=np.float64)) * Y_norms.T

    K = K1 + K2
    K -= 2 * np.mat(np.dot(X, Y.T))
    K *= -1. / (2 * np.power(sigma, 2))

    return np.exp(K)

