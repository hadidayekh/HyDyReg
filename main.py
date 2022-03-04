import kernels
from opti import optimizeL1, optimizeL2
import diff_methods
import numpy as np


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

    return differences







# Main function to initiate the derivative estimation and hybrid system identification
# t_list and x_list are the given trajectories
# diff_method is the differentiation method used (a function)
# kernel_list is a list consisting of:
##### kernel_list[0]: the kernel (a function)
##### kernel_list[1]: a list of the additional arguments kargs to the kernel
##### kernel_list[2]: the number of replicas of this kernel. This is to avoid calculating the same kernel matrix more than once
# deltas is a scalar or a list of the error
# C is the hyperparameter that weighs the different parts of the optimization objective function
# norm is either 'L1' or 'L2'
def learn(t_list, x_list, diff_method, kernel_list, deltas, C, norm='L2', print_on=False, write_on=True):
    assert (norm == 'L1' or norm == 'L2')
    # getting final X and final X dot by estimating derivatives
    final_X, final_X_dot = diff_method(t_list, x_list)

    if write_on:
        np.savetxt("input/final_X.csv", final_X, delimiter=', ', fmt='%8f')
        np.savetxt("input/final_X_dot.csv", final_X_dot, delimiter=', ', fmt='%8f')

    # computing kernels and adding them to list
    kernel_mat_list = []
    # for every kernel type
    for k_list in kernel_list:
        kernel = k_list[0]
        k_args = k_list[1]
        nb_replicas = k_list[2]
        # calculate the kernel matrix
        K  = kernel(final_X, final_X, k_args)
        # add the kernel matrix to kernel_mat_list, nb_replicas of times
        for i in range(nb_replicas):
            kernel_mat_list.append(K)

    # run the optimization function, depending on which norm
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
    return kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol