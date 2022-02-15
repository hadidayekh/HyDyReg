import kernels
from opti import optimize
import diff_methods

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
    optimize(kernel_mat_list, final_X_dot, deltas, C, print_on=print_on)