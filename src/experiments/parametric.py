from math import floor, sqrt
import sys

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from sklearn.metrics import mean_squared_error
from opti import HybridKernelRegressor
from diff_methods import diff_method_fps
from snds import SNDS_linear
import plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

exp_title = "parametric"
k_learn = True
save_plots = False
seg_tol = 0.1

nmodes = 4
radii = 1.0/nmodes
print(radii)
def labelfun(x, nmodes=nmodes):
    d = x[0]**2 + x[1]**2
    d = sqrt(d)
    label = floor(d/radii)
    return label if label < nmodes else nmodes-1


eventlist = []
params = []
for j in range(1,nmodes):
    def event(t, x, k=j):
        return x[0]**2 + x[1]**2 - (radii*k)**2
    event.direction = 0
    event.terminal = True
    eventlist.append(event)

for j in range(1, nmodes+1):
    beta = -1 if j % 2 == 1 else -1
    k = 2

    Aorg = np.array([[0, k*beta], [-beta,  0]])
    D, V = np.linalg.eig(Aorg)
    D = np.diag(D)

    epsilon_1 = -2*j
    epsilon_2 = 0
    
    E = np.array([[epsilon_1, 0], [0, epsilon_2]])
    
    #eigenvalues
    D1 = D + E
    
    A1 = V @ D1 @ np.linalg.inv(V)
    
    A = V @ D @ np.linalg.inv(V)

    A = A1.real

    params.append([[A[0,0], A[0,1], 0],
            [A[1,0], A[1,1], 0]])

params = np.array(params)
print(params)
##### DEFINING SNDS #####
sul = SNDS_linear(params, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters

x0s = np.random.random((5, 2))*2 - 1

time_step = 0.01
time_range = 10

# Simulating trajectories
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)


# Plotting results
plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True)#, plot_xt=True)

x0 = np.linspace(-1,1,100)
x1 = np.linspace(-1,1,100)
xx, yy = np.meshgrid(x0,x1)

for j in range(1, nmodes):
    print('hooo',j)
    zz = xx**2 + yy**2 - (radii*j)**2
    ax1.contour(xx, yy, zz, levels=[0], colors=['r'])

for i in range(nmodes):    
    print(eventlist[i-1](0,[0,.7]))

plt.show()