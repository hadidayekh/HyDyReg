import numpy as np
import matplotlib.pyplot as plt


t = np.genfromtxt('data/t_3.csv', delimiter=',')
t_list = np.array([t, np.copy(t)])

x0 = np.genfromtxt('data/x0_3.csv', delimiter=',')
x1 = np.genfromtxt('data/x1_3.csv', delimiter=',')

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

kernel_list = [[kernels.polynomial, [2, 1, 1], 1]]
kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol = learn(t_list, x_list, diff_methods.diff_method_fps, kernel_list, 0.01, 50, norm='L2', print_on=True)

n = Alphas_sol.shape[0]
D = Alphas_sol.shape[1]
N = Alphas_sol.shape[2]




#print("\n\n\nNow printing differences:")
#differences = get_differences_learned(X_dot, kernel_mat_list, Alphas_sol, betas_sol, Chis_sol)
#print(differences)
