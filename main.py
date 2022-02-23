import kernels
from opti import optimize
import diff_methods
import numpy as np

def learn(t_list, x_list, diff_method, kernel_list, deltas, C, print_on=False):
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
    sol, Alphas_sol, betas_sol, Chis_sol = optimize(kernel_mat_list, final_X_dot, deltas, C, print_on=print_on)
    n = len(kernel_list)
    for j in range(n):
        np.savetxt("output/Alphas" + str(j) + ".csv", Alphas_sol[j], delimiter=', ', fmt='%8f')
        np.savetxt("output/betas" + str(j) + ".csv", betas_sol[j], delimiter=', ', fmt='%8f')
        np.savetxt("output/Chis" + str(j) + ".csv", Chis_sol[j], delimiter=', ', fmt='%8f')
    return kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol