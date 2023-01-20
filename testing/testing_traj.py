import sys
# setting path
sys.path.append('./')
sys.path.append('../')

import kernels
import diff_methods
from main import learn, get_modes, get_differences_learned
from snds import SNDS
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os





def labelfun(x):
    x1, x2 = x
    if x2 <= 3 and x1 <= -2:
        return 1
    else:
        return 0


def event_track1(t, x):
    x1, x2 = x
    return x1 + 2


def event_track2(t, x):
    x1, x2 = x
    return x2 - 3


event_track1.terminal = True
event_track1.direction = 0
event_track2.terminal = True
event_track2.direction = 0
eventlist = [event_track1, event_track2]


y0 = [-14, 14]


k_learn = True
x0s = [[-11, 11], [-14, 14]]


a = -3
b = -2
cons1 = 0
c = 4
d = 1
cons2 = 0


if (a+d)**2 >= 4*(a*d-b*c):
    eig1 = (a+d)/2.0 + 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    eig2 = (a+d)/2.0 - 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    print("Solutions are real!")
    exit()
    print("Eigenvalue 1:", eig1)
    print("Eigenvalue 2:", eig2)
    if(eig1*eig2 >= 0):
        print("Eigenvalues have the same sign!")
        if(eig1 >= 0):
            print("It is a repelling/unstable node!")
        else:
            print("It is an attracting/stable node!")
    else:
        print("Eigenvalues have opposite signs!")
        print("It is a saddle!")
else:
    print("Solutions are complex!")
    if a+d >= 0:
        print("It is a repelling/unstable focus!")
    else:
        print("It is an attracting/stable focus!")


def eq1(t, y):
    return [a*y[0] + b*y[1] + cons1, c*y[0] + d*y[1] + cons2]


def eq2(t, y):
    return [5, -5]


eqs = [eq1, eq2]


sul = SNDS(2, 2, eqs, eventlist, labelfun)

trajectories = sul.simulate_trajectory(0.05, 5, x0s)


fig = trajectories.draw_trajectories_2D(mark_beginning=True, plot_xt=True)

fig.savefig('plot.pdf')
exit()
# for x in x_list:
#     x1 = x.T[0]
#     x2 = x.T[1]

#     plt.scatter(x1, x2, s=3,c='b')
#     plt.scatter(x1[0], x2[0], s=3, c='r')


# t_list = np.linspace(0, 10, 100)

# y_obj = solve_ivp(dy_dt, (0,10), y0, t_eval=t_list)

# y = y_obj.y

# plt.scatter(y[0], y[1], s=3,c='b')
# plt.scatter(y[0][0], y[1][0], s=3, c='r')


t_list = np.array(trajectories.t_list)
x_list = np.array(trajectories.x_list)

print(t_list.shape)
print(x_list.shape)


kernel_list = [[kernels.linear, [0], 3]]

if k_learn:
    X, X_dot = diff_methods.diff_method_fps(t_list, x_list)
    print(X.shape)
    kernel_mat_list, sol, Alphas_sol, betas_sol, Chis_sol = learn(
        t_list, x_list, diff_methods.diff_method_fps, kernel_list, 0.001, 100, norm='L2', print_on=True)

    n = Alphas_sol.shape[0]
    D = Alphas_sol.shape[1]
    N = Alphas_sol.shape[2]

    modes = get_modes(Chis_sol)

    colors = ['r', 'b', 'g']
    for i in range(X.shape[0]):
        plt.scatter(X[i][0], X[i][1], c=colors[modes[i]], s=3)
    print(modes)

    print("First mode:")
    K = kernel_list[0][0](X, X, kernel_list[0][1])
    x1 = Alphas_sol[0][0] @ (X.T[0])
    x2 = Alphas_sol[0][0] @ (X.T[1])

    print("\nx1_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas_sol[0][0]))

    x1 = Alphas_sol[0][1] @ (X.T[0])
    x2 = Alphas_sol[0][1] @ (X.T[1])

    print("\nx2_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas_sol[0][1]))

    # second linear equation
    print("\Second mode:")
    K = kernel_list[0][0](X, X, kernel_list[0][1])
    x1 = Alphas_sol[1][0] @ (X.T[0])
    x2 = Alphas_sol[1][0] @ (X.T[1])

    print("\nx1_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas_sol[1][0]))

    x1 = Alphas_sol[1][1] @ (X.T[0])
    x2 = Alphas_sol[1][1] @ (X.T[1])

    print("\nx2_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas_sol[1][1]))


    if kernel_list[0][2] == 3:
        print("\Third mode:")
        K = kernel_list[0][0](X, X, kernel_list[0][1])
        x1 = Alphas_sol[2][0] @ (X.T[0])
        x2 = Alphas_sol[2][0] @ (X.T[1])

        print("\nx1_dot\t=")
        print("% .8f" % x1, "\tx1\t+")
        print("% .8f" % x2, "\tx2\t+")
        print("% .8f" % (betas_sol[2][0]))

        x1 = Alphas_sol[2][1] @ (X.T[0])
        x2 = Alphas_sol[2][1] @ (X.T[1])

        print("\nx2_dot\t=")
        print("% .8f" % x1, "\tx1\t+")
        print("% .8f" % x2, "\tx2\t+")
        print("% .8f" % (betas_sol[2][1]))



plt.savefig('plot.pdf')
