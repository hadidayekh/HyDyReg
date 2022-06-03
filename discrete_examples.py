import numpy as np
import random
import matplotlib.pyplot as plt
from discrete import extract_from_trajectory
import kernels
import opti
from main import get_modes

random.seed(10)

"""
first mode:  y[t] = -0.905 y[t-1]   + 0.9 u[t-1] + e[t]
second mode: y[t] = -0.4   y^2[t-1] + 0.5 u[t-1] + e[t]
"""
def generate_disc_hyb1(y0, noise=False):
    u = np.zeros(101)
    y = np.zeros(101)
    y[0] = y0

    u[0] = random.random()
    for t in range(1, 101):
        if t < 41:
            y[t] = -0.905 * y[t-1]   + 0.9 * u[t-1]
        else:
            y[t] = -0.4 * y[t-1]**2 + 0.5 * u[t-1]
        u[t] = random.random()
    
    return(u,y)


u,y = generate_disc_hyb1(0.6)

X, y_out = extract_from_trajectory(u,y,1,1)

print(u.shape)
print(y.shape)
print(X.shape)
print(y_out.shape)
y_out = y_out.reshape(-1, 1)
print(y_out.shape)

kernel_list = [[kernels.linear, [0], 1], [kernels.polynomial, [2, 1, 1], 1]]

# computing kernels and adding them to list
kernel_mat_list = []
# for every kernel type
for k_list in kernel_list:
    kernel = k_list[0]
    k_args = k_list[1]
    nb_replicas = k_list[2]
    # calculate the kernel matrix
    K  = kernel(X, X, k_args)
    # add the kernel matrix to kernel_mat_list, nb_replicas of times
    for i in range(nb_replicas):
        kernel_mat_list.append(K)

sol, Alphas_sol, betas_sol, Chis_sol = opti.optimizeL2(kernel_mat_list, y_out, 0.001, 1)

print(Alphas_sol, betas_sol)
print(get_modes(Chis_sol))
Alphas = Alphas_sol
betas = betas_sol
N = X.shape[0]