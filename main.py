import kernels
from opti import optimizeL1, optimizeL2
import diff_methods
import numpy as np


def get_modes(Chis_sol):
    return np.argmin(Chis_sol, axis=0)

def learn(t_list, x_list, diff_method, kernel_list, deltas, C, norm='L2', print_on=False, write_on=True):
    assert (norm == 'L1' or norm == 'L2')
    # getting final X and final X dot by estimating derivatives
    final_X, final_X_dot = diff_method(t_list, x_list)
    # computing kernels and adding them to list
    kernel_mat_list = []
    for k_list in kernel_list:
        kernel = k_list[0]
        k_args = k_list[1]
        nb_replicas = k_list[2]
        K  = kernel(final_X, final_X, k_args)
        for i in range(nb_replicas):
            kernel_mat_list.append(K)
    # run the optimization function
    if norm == 'L1':
        sol, Alphas_sol, betas_sol, Chis_sol = optimizeL1(kernel_mat_list, final_X_dot, deltas, C, print_on=print_on)
    else:
        sol, Alphas_sol, betas_sol, Chis_sol = optimizeL2(kernel_mat_list, final_X_dot, deltas, C, print_on=print_on)

    if write_on:
        n = len(kernel_mat_list)
        for j in range(n):
            np.savetxt("output/Alphas" + str(j) + ".csv", Alphas_sol[j], delimiter=', ', fmt='%8f')
            np.savetxt("output/betas" + str(j) + ".csv", betas_sol[j], delimiter=', ', fmt='%8f')
            np.savetxt("output/Chis" + str(j) + ".csv", Chis_sol[j], delimiter=', ', fmt='%8f')
        np.savetxt("input/final_X.csv", final_X, delimiter=', ', fmt='%8f')
        np.savetxt("input/final_X_dot.csv", final_X_dot, delimiter=', ', fmt='%8f')
    return kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol