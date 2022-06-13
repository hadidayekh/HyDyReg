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

sol, Alphas_sol, betas_sol, Chis_sol = opti.optimizeL2(kernel_mat_list, y_out, 0.0001, 1)

print(Alphas_sol, betas_sol)
modes = get_modes(Chis_sol)
print(modes)

Alphas = Alphas_sol
betas = betas_sol
N = X.shape[0]
print("NUMBER OF POINTS IS "+str(N)+"\n")



x1_squared_1  = Alphas[1][0] @ (X.T[0]**2)
x2_squared_1  = Alphas[1][0] @ (X.T[1]**2)
x1_x2_1       = Alphas[1][0] @ (X.T[0] * X.T[1]) * 2
x1_1          = Alphas[1][0] @ (X.T[0]) * 2
x2_1          = Alphas[1][0] @ (X.T[1]) * 2
one_coef_1    = Alphas[1][0] @ np.ones(N)

print("\nx1_dot\t=")
print("% .8f" % x1_squared_1, "\tx1**2\t+")
print("% .8f" % x2_squared_1, "\tx2**2\t+")
print("% .8f" % x1_x2_1, "\tx1x2\t+")
print("% .8f" % x1_1, "\tx1\t+")
print("% .8f" % x2_1, "\tx2\t+")
print("% .8f" % (one_coef_1 + betas[1][0]))


print("\nSecond mode:")
x1_0          = Alphas[0][0] @ (X.T[0])
x2_0          = Alphas[0][0] @ (X.T[1])

print("\nx1_dot\t=")
print("% .8f" % x1_0, "\tx1\t+")
print("% .8f" % x2_0, "\tx2\t+")
print("% .8f" % (betas[0][0]))



u,y = generate_disc_hyb1(0.6)

X, y_out = extract_from_trajectory(u,y,1,1)


y_predicted = np.zeros(N)
for i in range(N):
    if modes[i] == 0:
        y_predicted[i] = x1_0 * X[i][0] + x2_0 * X[i][1] + betas[0][0]
    elif modes[i] == 1:
        y_predicted[i] = x1_1 * X[i][0] + x2_1 * X[i][1] + x1_squared_1 * X[i][0]**2 + x2_squared_1 * X[i][1]**2 + x1_x2_1 * X[i][0]*X[i][1] + one_coef_1 + betas[1][0]



# plt.scatter(range(N), y_out, s = 1)
# plt.scatter(range(N), y_predicted, c='r', s= 1)

print(y_predicted.shape)
print(y_out.shape)
diff = y_predicted - y_out.reshape((N,))
print(diff.shape)
plt.plot(diff)

plt.show()