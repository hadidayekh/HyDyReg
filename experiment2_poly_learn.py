import numpy as np
import matplotlib.pyplot as plt


t = np.genfromtxt('data/t_2.csv', delimiter=',')
t_list = np.array([t, np.copy(t), np.copy(t)])


x0 = np.genfromtxt('data/x0_2.csv', delimiter=',')
x1 = np.genfromtxt('data/x1_2.csv', delimiter=',')
x2 = np.genfromtxt('data/x2_2.csv', delimiter=',')

x_list = np.array([x0, x1, x2])

ax = plt.axes()
for temp_x in x_list:
    x1_list = temp_x.T[0]
    x2_list = temp_x.T[1]
    ax.scatter(x1_list, x2_list,c='b',s=1)
    
ax.set_xlabel('x1')
ax.set_ylabel('x2')
# plt.legend()
#plt.show()
#exit()

from main import learn, get_modes, get_differences_learned
import diff_methods
import kernels

kernel_list = [[kernels.polynomial, [2, 1, 1], 3]]#, [kernels.linear, [0], 2]]
kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol = learn(t_list, x_list, diff_methods.diff_method_fps, kernel_list, 0.1, 10, norm='L2', print_on=True)

n = Alphas_sol.shape[0]
D = Alphas_sol.shape[1]
N = Alphas_sol.shape[2]

modes = get_modes(Chis_sol)

X, X_dot = diff_methods.diff_method_fps(t_list, x_list)

modes_true01 = np.zeros(N)
modes_true12 = np.ones(N)
modes_true20 = np.ones(N)
modes_true02 = np.ones(N)
modes_true10 = np.ones(N)
modes_true21 = np.ones(N)


for i in range(N):
    if X[i][1] <= 0:
        modes_true01[i] = 0
        modes_true02[i] = 0
        modes_true12[i] = 1
        modes_true10[i] = 1
        modes_true20[i] = 2
        modes_true21[i] = 2
    elif X[i][0] <= 0:
        modes_true01[i] = 1
        modes_true02[i] = 2
        modes_true12[i] = 2
        modes_true10[i] = 0
        modes_true20[i] = 0
        modes_true21[i] = 1
    else:
        modes_true01[i] = 2
        modes_true02[i] = 1
        modes_true12[i] = 0
        modes_true10[i] = 2
        modes_true20[i] = 1
        modes_true21[i] = 0


modes_true = [modes_true01, modes_true02, modes_true10, modes_true12, modes_true20, modes_true21]

print("\n\n\n")

for m in modes_true:
    print(modes - m)


print("\n\n\nNow printing differences:")
differences = get_differences_learned(X_dot, kernel_mat_list, Alphas_sol, betas_sol, Chis_sol)
print(differences)
