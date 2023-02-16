import sys
# setting path
sys.path.append('./')
sys.path.append('../')

import kernels
import diff_methods
from main import learn, get_modes, get_differences_learned
from snds import SNDS, SNDS_linear
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from matplotlib.colors import ListedColormap



def labelfun(x):
    x1, x2 = x
    if x[1] >= -0.2*x[0]**2:
        return 0
    else:
        return 1
    if x2 <= 3 and x1 <= -2:
        return 0
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

eventlist = lambda t, x : x[1] + 0.2*x[0]**2


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

a1 = 0.1
a2 = 0.04
a3 = -0.9
b1 = -0.1
b2 = 0.06
b3 = -0.7

def eq1(t, y):
    #return [a*y[0] + b*y[1] + cons1, c*y[0] + d*y[1] + cons2]
    return [ a1 * y[0]**2 + a2 * y[1]**3,
            a3 * y[0]]


def eq2(t, y):
    return [ b1 * y[0]**2 + b2 * y[1]**3,
            b3 * y[0]]
    #return [8,-6]
    #return [5, -5]
    # with [8,-6] the visual looks close

eqs = [eq1, eq2]

k_learn = True
x0s = [[-11, 11], [-14, 14]]
x0s = [[0,1],[0,2],[0,3],[0,4],[0,5]]

sul = SNDS(2, 2, eqs, eventlist, labelfun)

params = np.array([[[a, b, cons1], [c, d, cons2]], [[0, 0, 8], [0,0,-6]]])

#sul = SNDS_linear(params, eventlist, labelfun)
#exit()

time_step = 0.05
time_range = 15
num_trajectories = len(x0s)
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
#segments = trajectories.segment_trajectories(0.5)
print(trajectories.t_list[0].shape)
print(trajectories.t_list[1].shape)
print(trajectories.x_list[0].shape)
print(trajectories.x_list[1].shape)
print(trajectories.x_list)
fig,ax = trajectories.draw_trajectories_2D(mark_beginning=True, plot_xt=True)#, segments=segments)

#ax['main'].axhline(y = 3, color = 'r', alpha=0.2, linestyle = '-', linewidth=0.1)
#ax['main'].axvline(x = -2, color = 'r', alpha=0.5, linestyle = '-', linewidth=0.1)

ax = ax['main']

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

x = [xmin, -2]
y1 = [ymin, ymin]
y2 = [3, 3]

d = np.arange(xmin, xmax, 0.01)
x0,x1 = np.meshgrid(d,d)

#ax.fill_between(x, y1, y2, lw=0, color='r', alpha=0.075)
im = ax.imshow( (x0 < -0.2 * x1**2).astype(int) , 
                extent=(x0.min(),x0.max(),x1.min(),x1.max()),origin="lower", cmap='Greys',alpha=0.2)

ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin, ymax)


title = 'traj=' + str(num_trajectories) + ' ts=' + str(time_step) + ' T=' + str(time_range)
ax.set_title(title)


fig.savefig('plot_traj=' + str(num_trajectories) + '_ts=' + str(time_step) + '_T=' + str(time_range) + '.pdf')

exit()
lr_models = trajectories.lin_reg(segments, diff_methods.diff_method_fps)

coefs1 = lr_models[1].coef_
coefs2 = lr_models[4].coef_

inter1 = lr_models[1].intercept_
inter2 = lr_models[4].intercept_

params1 = np.insert(coefs1, [2], [[inter1[0]], [inter1[1]]], axis=1)
params2 = np.insert(coefs2, [2], [[inter2[0]], [inter2[1]]], axis=1)

params1 = np.array([params1])
params2 = np.array([params2])

x0s = [[-11, 11], [-14, 14], [-20, 5], [2,3], [10,4]]#, [-10, -3], [-20,-20], [-20, 10], [-30,10], [-30,-10], [20, 0], [40,20], [-60,-40], [-60,-60]]


sul = SNDS_linear(params1, [], lambda x : 0)
traj = sul.simulate_trajectory(time_step, time_range, x0s)
fig,ax = traj.draw_trajectories_2D(mark_beginning=True, plot_xt=False)
#ax = ax['main']
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

x = [xmin, -2]
y1 = [ymin, ymin]
y2 = [3, 3]
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin, ymax)
ax.fill_between(x, y1, y2, lw=0, color='r', alpha=0.075)
fig.savefig("fig.pdf")
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

num_modes = 3

kernel_list = [[kernels.linear, [0], num_modes]]

if k_learn:
    fig, ax = plt.subplots()
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
        ax.scatter(X[i][0], X[i][1], c=colors[modes[i]], s=3)
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

#ax.axhline(y = 3, color = 'r', alpha=0.2, linestyle = '-', linewidth=0.1)
#ax.axvline(x = -2, color = 'r', alpha=0.5, linestyle = '-', linewidth=0.1)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

x = [xmin, -2]
y1 = [ymin, ymin]
y2 = [3, 3]

ax.fill_between(x, y1, y2, lw=0, color='r', alpha=0.075)

ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin, ymax)

title = 'traj=' + str(num_trajectories) + ' n=' + str(num_modes) + ' ts=' + str(time_step) + ' T=' + str(time_range)
ax.set_title(title)
fig.savefig('plot_res_traj=' + str(num_trajectories) + '_n=' + str(num_modes) + '_ts=' + str(time_step) + '_T=' + str(time_range) + '.pdf')
