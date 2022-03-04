# ax = plt.axes()
#     for temp_y in y_list:
#         y0_list = temp_y.T[0]
#         y1_list = temp_y.T[1]
#         ax.scatter(y0_list, y1_list,c='b',s=5)
    
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     #plt.legend()
#     #plt.show()

import numpy as np
import matplotlib.pyplot as plt


t = np.genfromtxt('data/t_1.csv', delimiter=',')
t_list = np.array([t, np.copy(t)])


x0 = np.genfromtxt('data/x0_1.csv', delimiter=',')
x1 = np.genfromtxt('data/x1_1.csv', delimiter=',')

x_list = np.array([x0, x1])

ax = plt.axes()
for temp_x in x_list:
    x1_list = temp_x.T[0]
    x2_list = temp_x.T[1]
    ax.scatter(x1_list, x2_list,c='b',s=1)
    
ax.set_xlabel('x1')
ax.set_ylabel('x2')
# plt.legend()
# plt.show()

from main import learn, get_modes, get_differences_learned
import diff_methods
import kernels

kernel_list = [[kernels.linear, [0], 2]]
kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol = learn(t_list, x_list, diff_methods.diff_method_fps, kernel_list, 0.00001, 10, norm='L1', print_on=True)

n = Alphas_sol.shape[0]
D = Alphas_sol.shape[1]
N = Alphas_sol.shape[2]

modes = get_modes(Chis_sol)

X, X_dot = diff_methods.diff_method_fps(t_list, x_list)

modes_true0 = np.zeros(N)
modes_true1 = np.ones(N)

for i in range(N):
    if X[i][0] > 98.5:
        modes_true0[i] = 1
        modes_true1[i] = 0

print("\n\n\n")

print(modes - modes_true0)
print(modes - modes_true1)


print("\n\n\nNow printing differences:")
differences = get_differences_learned(X_dot, kernel_mat_list, Alphas_sol, betas_sol, Chis_sol)
print(differences)



# difference1 = X_dot.T[0] - kernel_mat_list[0] @ Alphas_sol[0][0] - betas_sol[0][0] * np.ones(N)
# difference2 = X_dot.T[0] - kernel_mat_list[1] @ Alphas_sol[1][0] - betas_sol[1][0] * np.ones(N)
# difference3 = X_dot.T[1] - kernel_mat_list[0] @ Alphas_sol[0][1] - betas_sol[0][1] * np.ones(N)
# difference4 = X_dot.T[1] - kernel_mat_list[1] @ Alphas_sol[1][1] - betas_sol[1][1] * np.ones(N)

# print(difference1)
# print(difference2)
# print(difference3)
# print(difference4)


# X, X_dot = diff_methods.diff_method_paper(t_list, x_list)

# N = X.shape[0]
# D = X.shape[1]

# X_dot_true = np.zeros((N,D))

# for i in range(N):
#     X_dot_true[i][0] = -0.026*(X[i][0]-X[i][1])
#     if(X[i][0] <= 98.5):
#         X_dot_true[i][1] = 1
#     else:
#         X_dot_true[i][1] = -1

# print(X_dot_true-X_dot)