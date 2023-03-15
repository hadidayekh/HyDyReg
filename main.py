import kernels
from opti import optimizeL1, optimizeL2
import diff_methods
import numpy as np
import os
from paper_utils import get_labeltest
label_test = get_labeltest(0)

def get_modes(Chis_sol):
    return np.argmin(Chis_sol, axis=0)


def get_differences_learned(X_dot, kernel_mat_list, Alphas, betas, Chis):
    n = Alphas.shape[0]
    D = Alphas.shape[1]
    N = Alphas.shape[2]

    modes = get_modes(Chis)

    for d in range(D):
        difference = np.zeros(N)
        differences = []
        for j in range(n):
            differences.append(X_dot.T[d] - kernel_mat_list[j] @ Alphas[j][d] - betas[j][d] * np.ones(N))

        for i in range(N):
            difference[i] = differences[modes[i]][i]

    return difference